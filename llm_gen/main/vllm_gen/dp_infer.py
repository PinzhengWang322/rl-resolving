import os
import math
import itertools
import time
from typing import List

import ray
from more_itertools import distribute
from datasets import Dataset  # 仅用于类型提示

from vllm import LLM, SamplingParams
STOP_STR = os.getenv("STOP_STR", "").strip()

NUM_CPUS = int(os.environ.get("NUM_CPUS", 16))
# 如果为空，就用默认值
if not STOP_STR:
    STOP_STR = "better to redo the question"


class VLLMDataParallelInfer:
    """Ray-based data-parallel 推理器，接口与旧版保持一致。"""

    def __init__(
        self,
        model_path: str,
        dp_size: int = 1,
        tp_size: int = 1,
        node_size: int = 1,
        node_rank: int = 0,
        ray_address: str | None = None,
    ):
        """
        Args:
            model_path: HF checkpoint 路径或名称
            dp_size: 数据并行进程总数（所有节点加起来）
            tp_size: 单进程 tensor parallel size
            node_size / node_rank: 多节点时的集群信息
            ray_address: 若已有 Ray 集群，可填其 address；否则自动 `ray.init()`
        """
        self.model_path = model_path
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.node_size = node_size
        self.node_rank = node_rank

        # 只在 rank-0 负责启动 / 关闭 Ray
        if node_rank == 0 and not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True, include_dashboard=False, num_cpus=NUM_CPUS )
            self._owns_ray = True
        else:
            self._owns_ray = False
        
        print(
            f"[Init] model={model_path}  dp={dp_size}  tp={tp_size}  "
            f"node {node_rank}/{node_size}  ray={ray.get_runtime_context().namespace}"
        )

    # --------------------------------------------------------------------- #
    # 外部调用接口
    # --------------------------------------------------------------------- #
    def inference(
        self,
        dataset: Dataset | List[dict],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 16,
        gen_num: int = 1,
    ):
        """与旧版同名同参。返回值仍然是加了 responses 列的 dataset。"""
        t0 = time.time()
        prompts = [item["prompt"] for item in dataset]
        total = len(prompts)


        # 1. 交错切分 prompt
        buckets = [list(b) for b in distribute(self.dp_size, prompts)]

        # 2. 并行跑 Ray worker
        remote_func = self._make_remote_worker()
        refs = [
            remote_func.remote(
                self.model_path,
                self.tp_size,
                bucket,
                temperature,
                top_p,
                max_tokens,
                gen_num,
            )
            for bucket in buckets
        ]
        result_buckets: List[list[str]] = ray.get(refs)

        # 3. 交错拉平，恢复与原 prompts 一一对应
        responses = [
            x
            for x in itertools.chain.from_iterable(
                itertools.zip_longest(*result_buckets)
            )
            if x is not None
        ]

        # ------------------ 写回 dataset（接口保持与旧版一致） --------------- #
        assert len(responses) == total, "responses/prompt 数量不一致"

        dataset.add_column("responses")
        dataset.update_column_values("responses", responses)

        print(f"[Infer] {total} prompts done in {time.time()-t0:.2f}s")
        return dataset

    # --------------------------------------------------------------------- #
    # 单 GPU / 单进程推理
    # --------------------------------------------------------------------- #
    def _run_llm_local(
        self,
        prompts,
        temperature,
        top_p,
        max_tokens,
        gen_num,
    ):
        llm = LLM(model=self.model_path, tensor_parallel_size=self.tp_size, trust_remote_code=True)
        params = SamplingParams(
            n=gen_num,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=0,
            stop=[STOP_STR],
            include_stop_str_in_output=True, 
        )
        outputs = llm.generate(prompts, params)
        return [[o.text for o in out.outputs] for out in outputs]

    # --------------------------------------------------------------------- #
    # 构造 Ray 远程 worker（避免闭包捕获大对象）
    # --------------------------------------------------------------------- #
    @staticmethod
    def _make_remote_worker():
        @ray.remote(num_gpus=1)
        def _worker(
            model_path: str,
            tp_size: int,
            prompts: list[str],
            temperature: float,
            top_p: float,
            max_tokens: int,
            gen_num: int,
        ):
            llm = LLM(model=model_path, tensor_parallel_size=tp_size,max_model_len=32768, trust_remote_code=True)
            params = SamplingParams(
                n=gen_num,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logprobs=0,
                stop=[STOP_STR],
                include_stop_str_in_output=True, 
            )
            outs = llm.generate(prompts, params)
            return [[o.text for o in out.outputs] for out in outs]

        return _worker


    def __del__(self):
        if self._owns_ray and ray.is_initialized():
            ray.shutdown()

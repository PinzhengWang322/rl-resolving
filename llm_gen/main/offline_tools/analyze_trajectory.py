"""评估多轮纠错数据集并绘制指标曲线。

新增：`evaluate_chain_metrics_by_difficulty` ——
    根据每条样例的 “题目正确率” (n_correct / n_total) 排序、三等分，
    对 **hard / medium / easy** 三段分别调用 `evaluate_chain_metrics`，
    每段各生成 5 张图，一共 15 张。

改进：
---------
1. **统一链长** — 如果某条样例的 `answers` / `flags` 长度小于数据集中的最大轮数 (`max_chain_len`)，
   则用该样例最后一次的 `answer` 和 `flag` 对其进行填充，直到长度与 `max_chain_len` 一致。
   这样可以保证后续统计/绘图时所有链长一致。

用法示例
---------
```python
from chain_metrics_padded import evaluate_chain_metrics_by_difficulty

evaluate_chain_metrics_by_difficulty(dataset, save_dir="plots_by_diff")
```

> `save_dir` 下会自动创建 `hard`, `medium`, `easy` 子目录存放 PNG。
"""

import sys
import os
import argparse
from typing import List, Dict, Optional
import regex as re  # 支持递归正则

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_from_disk

# 项目内部工具
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.math_eval import math_lst_eval  # noqa: E402

# ---------------- 正则：支持嵌套 \\boxed{...} ----------------

def _have_boxed(text: str) -> Optional[str]:
    """返回 `\\boxed{}` 内部内容；若无则 `None`."""
    if "boxed" in text:
        return text
    else:
        return None


def _judge(gt: str, pred: Optional[str]) -> int:
    """判定正误：无法判断/匹配返回 0。"""
    if pred is None:
        return 0  # None → 错
    try:
        return int(math_lst_eval([gt], [pred])[0])
    except Exception:
        return 0


# -----------------------------------------------------------------------------
# 基础函数：评估整个数据集（或子集）并绘制 5 张图
# -----------------------------------------------------------------------------

def evaluate_chain_metrics(dataset: List[Dict], save_dir: str = "chain_figs") -> Dict[str, List[float]]:
    os.makedirs(save_dir, exist_ok=True)

    answer_chains: List[List[Optional[str]]] = []
    correct_chains: List[List[int]] = []

    # -------- 遍历样例：抽取答案与正确性 --------
    for ex in tqdm(dataset, desc="Parsing dataset"):
        gt = ex["gt_answer"]

        # 历史 assistant 回复
        answers = [
            m["content"]
            for m in ex["conversations"][:2]
            if m["role"] == "assistant"
        ] + [
            _have_boxed(m["content"])
            for m in ex["conversations"][2:]
            if m["role"] == "assistant"
        ]

        # 最后一轮
        if ex["responses"]:
            answers.append(_have_boxed(ex["responses"][0]))

        if not answers:
            continue  # 跳过空样例

        # ---- 正误链 ----
        first_flag = _judge(gt, answers[0])
        flags = [first_flag]
        for ans in answers[1:]:
            flag = _judge(gt, ans) if ans is not None else first_flag
            flags.append(flag)

        answer_chains.append(answers)
        correct_chains.append(flags)

    # -------- 统一链长：找最大轮次并填充 --------
    if not correct_chains:
        raise ValueError("Dataset appears empty after preprocessing.")

    max_chain_len = max(len(chain) for chain in correct_chains)

    for a_chain, f_chain in zip(answer_chains, correct_chains):
        pad_len = max_chain_len - len(f_chain)
        if pad_len > 0:
            first_ans = a_chain[0]
            first_flag = f_chain[0]
            a_chain.extend([first_ans] * pad_len)
            f_chain.extend([first_flag] * pad_len)

    # -------- 初始化统计数组 --------
    correct_rate = np.zeros(max_chain_len)
    right_to_wrong = np.zeros(max_chain_len)
    wrong_to_right = np.zeros(max_chain_len)
    wrong_changed_answer = np.zeros(max_chain_len)
    pass_rate = np.zeros(max_chain_len)

    denom_correct = np.zeros(max_chain_len, dtype=int)
    denom_r2w = np.zeros(max_chain_len, dtype=int)
    denom_w2r = np.zeros(max_chain_len, dtype=int)
    denom_change = np.zeros(max_chain_len, dtype=int)
    denom_pass = np.zeros(max_chain_len, dtype=int)

    # -------- 累加统计 --------
    for answers, flags in zip(answer_chains, correct_chains):
        first_flag = flags[0]
        first_ans = answers[0]
        prefix_pass = False

        for t, (flag, ans) in enumerate(zip(flags, answers)):
            # ① 正确率
            denom_correct[t] += 1
            correct_rate[t] += flag

            # ② 对→错
            if first_flag == 1:
                denom_r2w[t] += 1
                if flag == 0:
                    right_to_wrong[t] += 1

            # ③ 错→对
            if first_flag == 0:
                denom_w2r[t] += 1
                if flag == 1:
                    wrong_to_right[t] += 1

            # ④ 错后改答案
            if first_flag == 0:
                denom_change[t] += 1
                if ans != first_ans:
                    wrong_changed_answer[t] += 1

            # ⑤ pass
            if flag == 1:
                prefix_pass = True
            denom_pass[t] += 1
            if prefix_pass:
                pass_rate[t] += 1

    # -------- 计算概率 --------
    def _div(num, den):
        return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0)

    correct_rate = _div(correct_rate, denom_correct)
    right_to_wrong = _div(right_to_wrong, denom_r2w)
    wrong_to_right = _div(wrong_to_right, denom_w2r)
    wrong_changed_answer = _div(wrong_changed_answer, denom_change)
    pass_rate = _div(pass_rate, denom_pass)

    # -------- 绘图 --------
    x = np.arange(1, max_chain_len + 1)
    plots = [
        (correct_rate, "Correct rate", "correct_rate.png"),
        (right_to_wrong, "Right→Wrong", "right_to_wrong.png"),
        (wrong_to_right, "Wrong→Right", "wrong_to_right.png"),
        (wrong_changed_answer, "Wrong changed answer", "wrong_changed_answer.png"),
        (pass_rate, "Pass rate", "pass_rate.png"),
    ]

    for y, title, fname in plots:
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker="o")
        plt.xlabel("Turn number")
        plt.ylabel("Probability / Accuracy")
        plt.title(title)
        plt.xticks(x)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
        plt.close()

    return {
        "correct_rate": correct_rate.tolist(),
        "right_to_wrong": right_to_wrong.tolist(),
        "wrong_to_right": wrong_to_right.tolist(),
        "wrong_changed_answer": wrong_changed_answer.tolist(),
        "pass_rate": pass_rate.tolist(),
    }


# -----------------------------------------------------------------------------
# 新增函数：按 difficulty 三等分评估
# -----------------------------------------------------------------------------

def evaluate_chain_metrics_by_difficulty(
    dataset: List[Dict],
    save_dir: str = "plots_by_difficulty",
    n_levels: int = 3,
    level_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    将数据集按 (n_correct / n_total) 升序分为 ``n_levels`` 份，
    各子集分别调用 ``evaluate_chain_metrics`` 并输出图表。

    Parameters
    ----------
    dataset : List[Dict]
        原始样例列表。
    save_dir : str, default "plots_by_difficulty"
        总保存目录，内部会自动创建子目录。
    n_levels : int, default 3
        要划分的层级数 (>=1)。
    level_names : list[str] | None
        每个子集的名称。如未提供，则：
          * n_levels == 3 → ["hard", "medium", "easy"]
          * 其它情况 → ["level_1", "level_2", ...] 依次递增。

    Returns
    -------
    Dict[str, Dict]
        形如 {"hard": metrics_dict, ...} 的嵌套字典。
    """
    if n_levels < 1:
        raise ValueError("n_levels 必须 ≥ 1")

    # -------- 生成子集名称 --------
    if level_names:
        if len(level_names) != n_levels:
            raise ValueError("len(level_names) 必须等于 n_levels")
        names = level_names
    else:
        if n_levels == 3:
            names = ["hard", "medium", "easy"]
        else:
            names = [f"level_{i+1}" for i in range(n_levels)]

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    # -------- 计算每条样例的准确率得分 --------
    scores = []
    for idx, ex in enumerate(dataset):
        d = ex.get("difficulty", {})
        acc = d.get("n_correct", 0) / max(d.get("n_total", 1), 1)
        scores.append((idx, acc))

    # 按准确率升序 → “越难” 越靠前
    scores.sort(key=lambda x: x[1])

    # -------- 用 numpy.array_split 按行号均匀切 ``n_levels`` 份 --------
    idx_sorted = np.array([idx for idx, _ in scores])
    idx_chunks = np.array_split(idx_sorted, n_levels)

    results = {}
    for name, idx_chunk in zip(names, idx_chunks):
        subset = [dataset[i] for i in idx_chunk.tolist()]
        print(f"\n=== Evaluating {name} subset (size={len(subset)}) ===")

        subdir = save_root / name
        metrics = evaluate_chain_metrics(subset, str(subdir))
        results[name] = metrics

    return results

def evaluate_chain_metrics_by_num_correct(
    dataset: List[Dict],
    save_dir: str = "plots_by_num_correct",
    n_total: int = 8,
) -> Dict[int, Dict]:
    from collections import defaultdict
    """
    将样例按 n_correct ∈ [0, n_total] 归为 9 个等级，分别评估并绘图。

    返回:
        {0: metrics_dict, 1: metrics_dict, …, n_total: metrics_dict}
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---------- 收集各等级样例 ----------
    buckets = defaultdict(list)          # key = n_correct, value = list[example]
    for ex in dataset:
        d = ex.get("difficulty", {})
        n_correct = d.get("n_correct", 0)
        # 防御：确保落在 [0, n_total] 区间
        n_correct = min(max(int(n_correct), 0), n_total)
        buckets[n_correct].append(ex)

    results: Dict[int, Dict] = {}
    for k in range(n_total + 1):
        subset = buckets.get(k, [])
        print(f"\n=== Evaluating subset: {k} / {n_total} correct (size={len(subset)}) ===")

        # 若该等级没有样例，跳过但返回空 dict，保持键齐全
        if not subset:
            results[k] = {}
            continue

        subdir = os.path.join(save_dir, f"correct_{k}")
        metrics = evaluate_chain_metrics(subset, subdir)
        results[k] = metrics

    return results

# ---------------- CLI (optional) ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate correction‑loop metrics (overall & by difficulty)")
    parser.add_argument("--ds_path", type=str, required=True, help="Path to HuggingFace dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save figures")
    parser.add_argument("--by_difficulty", action="store_true", help="Also split into terciles by difficulty and plot")
    parser.add_argument("--by_diffnum", action="store_true", help="Also split into terciles by difficulty and plot")
    args = parser.parse_args()

    ds = load_from_disk(args.ds_path)
    if args.by_diffnum:
        evaluate_chain_metrics_by_num_correct(ds, args.save_dir)

    if args.by_difficulty:
        evaluate_chain_metrics_by_difficulty(ds, args.save_dir)
    else:
        evaluate_chain_metrics(ds, args.save_dir)

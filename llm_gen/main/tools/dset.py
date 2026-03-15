from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from typing import Dict, List, Tuple
from omegaconf import DictConfig
from transformers import AutoTokenizer
import os
from datasets import load_from_disk
from typing import Optional
import uuid
import re
from collections import defaultdict
    
class VllmDataset:
    def __init__(
        self, 
        tokenizer: Optional[AutoTokenizer] = None, 
        data_config: Optional[DictConfig] = None, 
        format_config: Optional[DictConfig] = None,
        question_template_path: Optional[str] = None,
        dataset: Optional[Dataset] = None,
    ):
        if dataset is None:
            self.tokenizer = tokenizer
            self.data_config = data_config
            self.format_config = format_config
            self.question_template_path = question_template_path
            self.make_dataset()
        else:
            self.dataset = dataset
    
    def make_dataset(self):
        if self.data_config.path.endswith(".parquet"):
            raw_dataset = load_dataset(
                'parquet',
                data_files=self.data_config.path
            )[self.data_config.split]
        elif self.data_config.path.endswith(".jsonl") or self.data_config.path.endswith(".json"):
            raw_dataset = load_dataset(
                'json',
                data_files=self.data_config.path
            )[self.data_config.split]
        else:
            if self.data_config.subset is None:
                raw_dataset = load_dataset(
                    self.data_config.path
                )[self.data_config.split]
            else:
                raw_dataset = load_dataset(
                    self.data_config.path, self.data_config.subset
                )[self.data_config.split]


        new_data = []

        for item in raw_dataset:
            if self.format_config and not self.format_config.apply_chat_template:
                prompt = item[self.data_config.question_key]
                conversations = None
            else:
                if self.question_template_path:
                    question_template = open(self.question_template_path, 'r').read()
                    conversations = [
                        {"role": "user", "content": question_template.replace('{QUESTION}', item[self.data_config.question_key])}
                    ]
                else:
                    conversations = [
                        {"role": "user", "content": item[self.data_config.question_key]}
                    ]
                prompt = self.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
                
            new_item = {
                'id': str(uuid.uuid4()),
                'prompt': prompt,
                'gt_answer': item[self.data_config.answer_key],
                'conversations': conversations,
                'metadata': item  
            }
            new_data.append(new_item)

        self.dataset = Dataset.from_list(new_data)

    def add_column(self, key):
        if key not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(key, [[] for _ in range(len(self.dataset))])
        else:
            self.dataset = self.dataset.map(lambda _: {key: []})

    def update_column_values(self, key: str, values: list):
        if len(values) != len(self.dataset):
            raise ValueError(f"Length of values ({len(values)}) does not match dataset size ({len(self.dataset)}).")
        
        self.dataset = self.dataset.map(lambda example, idx: {key: values[idx]}, with_indices=True)


    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    
    def save(self, dir: str, name: str):
        path = os.path.join(dir, name)
        os.makedirs(path, exist_ok=True)
        self.dataset.save_to_disk(path)

    @staticmethod
    def load(dir: str, name: str):
        path = os.path.join(dir, name)
        dataset = load_from_disk(path)
        return VllmDataset(dataset=dataset)
    
    @staticmethod
    def merge_datasets(datasets: List["VllmDataset"]) -> "VllmDataset":
        """
        Merge multiple VllmDataset instances into one.
        """
        all_ds = [ds.dataset for ds in datasets]
        merged_ds = concatenate_datasets(all_ds)
        return VllmDataset(dataset=merged_ds)
    
    def unflatten_dataset(self, group_key: Optional[str] = None) -> "VllmDataset":
        if group_key is None:
            group_key = self.data_config.id_key

        grouped_dict = defaultdict(list)
        for ex in self.dataset:
            key = ex[group_key]
            response = ex["responses"][0]  
            grouped_dict[key].append(response)

        unflattened_list = []
        first_by_key = {}
        for ex in self.dataset:
            k = ex[group_key]
            if k not in first_by_key:
                first_by_key[k] = ex

        for key, responses in grouped_dict.items():
            first_ex = first_by_key[key]
            new_ex = dict(first_ex)
            new_ex["responses"] = responses
            unflattened_list.append(new_ex)

        return VllmDataset(dataset=Dataset.from_list(unflattened_list))
   

    def flatten_dataset(self) -> "VllmDataset":
        """
        Flatten the dataset so that each example has exactly one response in `responses`.
        """
        flattened_list = []
        for ex in self.dataset:
            responses = ex.get('responses', [])
            for resp in responses:
                new_ex = dict(ex)
                new_ex['responses'] = [resp]
                flattened_list.append(new_ex)
        flat_ds = Dataset.from_list(flattened_list)
        return VllmDataset(dataset=flat_ds)
    
    def split_by_detect_str(self, detect_str: str) -> Tuple["VllmDataset", "VllmDataset"]:
        """
        Split the dataset into two VllmDatasets based on presence of detect_str in responses.
        Returns (with_str, without_str).
        """
        flat = self.flatten_dataset()
        # assume responses is a list with a single string element
        has_ds = flat.dataset.filter(lambda ex: detect_str in ex.get('responses', [])[0])
        not_has_ds = flat.dataset.filter(lambda ex: detect_str not in ex.get('responses', [])[0])
        return VllmDataset(dataset=has_ds), VllmDataset(dataset=not_has_ds)


from copy import deepcopy
from datasets import Dataset

class VllmCorrectDataset:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        max_len: int,
        correct_prompt: str,
    ):
        self.tokenizer = tokenizer
        self.orig_dataset = dataset
        self.max_len = max_len
        self.correct_prompt = correct_prompt
        self.RESP_KEYS=['responses']
        self.make_dataset()

    def make_dataset(self):
        new_data = []

        for sample in self.orig_dataset:
            if sample.get("conversations"):
                base_conv = deepcopy(sample["conversations"])
            else:  
                question = sample.get("prompt")
                if question is None:
                    raise ValueError("找不到原问题文本。")
                base_conv = [{"role": "user", "content": question}]

            resp_list = None
            for k in self.RESP_KEYS:
                if k in sample:
                    resp_list = sample[k]
                    break
            if resp_list is None:
                raise ValueError(f"样本 {sample.get('id')} 缺少候选回复列。")

            for r_idx, resp in enumerate(resp_list):
                conversations = deepcopy(base_conv)
                conversations.append({"role": "assistant", "content": resp})
                conversations.append({"role": "user", "content": self.correct_prompt})

                prompt = self.tokenizer.apply_chat_template(
                    conversations,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # ---------- 4. 长度过滤 ---------- #
                if len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"]) > self.max_len:
                    continue

                new_data.append(
                    {
                        "id": f"{sample['id']}_{r_idx}",
                        "prompt": prompt,
                        "gt_answer": sample.get("gt_answer", ""),
                        "conversations": conversations,
                        "response": "",          # 下一轮模型写入
                        "metadata": sample,      # 保留原样本
                    }
                )

        self.dataset = Dataset.from_list(new_data)

    def add_column(self, key: str):
        if key not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                key, [[] for _ in range(len(self.dataset))]
            )
        else:
            self.dataset = self.dataset.map(lambda _: {key: []})

    def update_column_values(self, key: str, values: list):
        if len(values) != len(self.dataset):
            raise ValueError(
                f"Length of values ({len(values)}) "
                f"does not match dataset size ({len(self.dataset)})."
            )
        self.dataset = self.dataset.map(
            lambda example, idx: {key: values[idx]}, with_indices=True
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def save(self, dir_: str, name: str):
        path = os.path.join(dir_, name)
        os.makedirs(path, exist_ok=True)
        self.dataset.save_to_disk(path)

    @staticmethod
    def load(dir_: str, name: str):
        path = os.path.join(dir_, name)
        dataset = load_from_disk(path)
        return VllmCorrectDataset(dataset=dataset)
    


class VllmVerifyDataset:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        max_len: int,
        verify_prompt: str,
    ):
        self.tokenizer = tokenizer
        self.orig_dataset = dataset
        self.max_len = max_len
        self.verify_prompt = verify_prompt
        self.RESP_KEYS=['responses']
        self.make_dataset()

    def make_dataset(self):
        new_data = []

        for sample in self.orig_dataset:
            # ---------- 1. 取历史 conversation ---------- #
            if sample.get("conversations"):
                base_conv = deepcopy(sample["conversations"])
            else:  # 没有 conversations 就用 prompt 还原一句话历史
                question = sample.get("prompt")
                if question is None:
                    raise ValueError("找不到原问题文本。")
                base_conv = [{"role": "user", "content": question}]

            # ---------- 2. 找到回复列表 ---------- #
            resp_list = None
            for k in self.RESP_KEYS:
                if k in sample:
                    resp_list = sample[k]
                    break
            if resp_list is None:
                raise ValueError(f"样本 {sample.get('id')} 缺少候选回复列。")

            # ---------- 3. 针对每条回复复制一份历史并追加两句 ---------- #
            for r_idx, resp in enumerate(resp_list):
                conversations = deepcopy(base_conv)
                conversations.append({"role": "assistant", "content": resp})
                conversations.append({"role": "user", "content": self.verify_prompt})

                prompt = self.tokenizer.apply_chat_template(
                    conversations,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # ---------- 4. 长度过滤 ---------- #
                if len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"]) > self.max_len:
                    continue

                new_data.append(
                    {
                        "id": f"{sample['id']}_{r_idx}",
                        "prompt": prompt,
                        "gt_answer": sample.get("gt_answer", ""),
                        "conversations": conversations,
                        "response": "",          # 下一轮模型写入
                        "metadata": sample,      # 保留原样本
                    }
                )

        self.dataset = Dataset.from_list(new_data)

    def add_column(self, key: str):
        if key not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                key, [[] for _ in range(len(self.dataset))]
            )
        else:
            self.dataset = self.dataset.map(lambda _: {key: []})

    def update_column_values(self, key: str, values: list):
        if len(values) != len(self.dataset):
            raise ValueError(
                f"Length of values ({len(values)}) "
                f"does not match dataset size ({len(self.dataset)})."
            )
        self.dataset = self.dataset.map(
            lambda example, idx: {key: values[idx]}, with_indices=True
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def save(self, dir_: str, name: str):
        path = os.path.join(dir_, name)
        os.makedirs(path, exist_ok=True)
        self.dataset.save_to_disk(path)

    @staticmethod
    def load(dir_: str, name: str):
        path = os.path.join(dir_, name)
        dataset = load_from_disk(path)
        return VllmVerifyDataset(dataset=dataset)

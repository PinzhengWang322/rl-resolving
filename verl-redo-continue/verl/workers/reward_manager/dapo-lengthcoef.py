# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from typing import Dict, List, Tuple

import re
from typing import Optional
import os

REDO_STR=os.environ['REDO_STR']

def last_boxed_pos_regex(s: str) -> Optional[int]:
    """
    返回 s 中最后一个完整 'boxed{xxx}' 的起始索引，
    如果没有找到，返回 None。
    """
    _pattern = re.compile(r'boxed\{.*?\}')
    matches = list(_pattern.finditer(s))
    if not matches:
        return None
    last = matches[-1]
    return last.start() 

def check_strict_redo(response, redo_str):
    box_id = last_boxed_pos_regex(response)
    if (redo_str in response[-100:]) and ((box_id is None) or (response.rfind(redo_str) - box_id > 40)):
        return True
    return False
    # 检查模型的redo是否是经过思考，而非得到答案后直接甩出的。
    
def get_ai_prefix(tokenizer, placeholder="__UNLIKELY_PLACEHOLDER__"):
    msgs = [{"role": "user", "content": placeholder}]
    rendered = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )
    return rendered.rsplit(placeholder, 1)[-1]


@register("dapo")
class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.ai_prefix = get_ai_prefix(self.tokenizer)


        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )


    def compute_response_lengths(
        self, data
    ) -> Tuple[List[int], Dict[str, List[int]], Dict[str, List[int]]]:
        """
        根据 attention_mask 截取有效 tokens，解码后按 ai_prefix 切分，取其后的 response，
        再用 tokenizer 统计 response 的 token 数。

        参数
        ----
        data.non_tensor_batch['uid'] : List[str]
        data.non_tensor_batch['question_uid'] : List[str]
        data.batch['input_ids'] : torch.Tensor [B, T]
        data.batch['attention_mask'] : torch.Tensor [B, T]

        返回
        ----
        lengths : List[int]                      # 每条样本的 response token 长度
        uid_len : Dict[str, List[int]]           # {uid: [len1, len2, ...]}
        question_uid_len : Dict[str, List[int]]  # {question_uid: [len1, len2, ...]}
        """

        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]

        uids = data.non_tensor_batch["uid"]
        qids = data.non_tensor_batch["question_uid"]

        assert input_ids.ndim == 2 and attention_mask.ndim == 2, "input_ids/attention_mask 应为二维"
        assert input_ids.shape == attention_mask.shape, "input_ids 与 attention_mask 形状需一致"
        assert len(uids) == input_ids.size(0) and len(qids) == input_ids.size(0), "uid/question_uid 数量需与 batch 对齐"

        lengths: List[int] = []
        uid_len: Dict[str, List[int]] = {}
        question_uid_len: Dict[str, List[int]] = {}

        B = input_ids.size(0)

        # 确保在 CPU 上进行解码（有些 tokenizer 需要）
        input_ids_cpu = input_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()

        for i in range(B):
            uid = str(uids[i])
            qid = str(qids[i])

            ids_row: torch.Tensor = input_ids_cpu[i]
            mask_row: torch.Tensor = attention_mask_cpu[i].to(dtype=torch.bool)

            # 仅取有效 token
            valid_ids = ids_row[mask_row].tolist()

            # 解码为字符串；保留特殊符号（有助于确保能匹配到 ai_prefix）
            response_text = self.tokenizer.decode(valid_ids, skip_special_tokens=False).split(self.ai_prefix)[-1]

            # 用 tokenizer 再编码统计 token 数；不加特殊符号
            enc = self.tokenizer(
                response_text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            resp_len = len(enc["input_ids"])

            lengths.append(resp_len)

            # 累计到字典
            uid_len.setdefault(uid, []).append(resp_len)
            question_uid_len.setdefault(qid, []).append(resp_len)

        return lengths, uid_len, question_uid_len

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        question_uid_2_uids = defaultdict(list)
        uid_2_question_uid = {}
        uid_expectation = {}
        uid_2_acc = defaultdict(list)
        uid_2_redo = defaultdict(list)
        uid_2_redo_len = defaultdict(list)
        uid_2_noredo_len = defaultdict(list)
        uid_2_fulllen = defaultdict(list)
        uid_2_redo_rate = {}
        uid_2_redolen = {}
        uid_2_noredolen = {}
        uid_2_redo_extra_len = {}
        uid_2_mean_len = {}
        data_info_lst = []
        redo_tag = REDO_STR
        write_flag = True

        lengths, uid_2_len, quid_2_len = self.compute_response_lengths(data)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            data_len = lengths[i]

            prompt_ids = data_item.batch["prompts"].int()

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"].int()
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            data_info = {
                "score": result['score'],
                "acc": result['acc'],
                "pred": result['pred'],
                "redo": False,
                "question_uid": data_item.non_tensor_batch["question_uid"],
                "uid": data_item.non_tensor_batch["uid"]
            }

            if check_strict_redo(response_str, redo_tag):
                data_info['score'] = None
                data_info['pred'] = '[REDO]'
                data_info['redo'] = True

            uid_2_redo[data_info['uid']].append(data_info['redo'])
            if data_info['redo']:
                uid_2_redo_len[data_info['uid']].append(data_len)
            else:
                uid_2_noredo_len[data_info['uid']].append(data_len)

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                # for key, value in result.items():
                #     reward_extra_info[key].append(value)
            else:
                score = result

            
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                # reward += overlong_reward / 2
                # if self.overlong_buffer_cfg.log:
                #     reward_extra_info["overlong_reward"].append(overlong_reward)
                #     reward_extra_info["overlong"].append(overlong_reward < 0)

            data_info['overlong_reward'] = overlong_reward / 2
            data_info['valid_response_length'] = valid_response_length

            # 构建数据结构，用以估计redo概率。
            uid, question_uid = data_info['uid'], data_info['question_uid']
            question_uid_2_uids[question_uid].append(uid)
            uid_2_question_uid[uid] = question_uid
            uid_2_acc[uid].append(data_info['score'])
            data_info_lst.append(data_info)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # 计算redo正确率
        for question_uid, uids in question_uid_2_uids.items():
            # 处理uid的平均无偏正确率
            uids_group_acc = {
                uid: {
                    "uid_responses_sum": sum([score for score in uid_2_acc[uid] if score is not None]),
                    "uid_responses_num": len([score for score in uid_2_acc[uid] if score is not None]),
                }
                for uid in uids
            }

            for uid in uids_group_acc:
                len_sum = sum([uid_data["uid_responses_num"]
                            for uid_iter, uid_data in uids_group_acc.items() 
                            if uid_iter != uid])
                
                sum_sum = sum([uid_data["uid_responses_sum"]
                            for uid_iter, uid_data in uids_group_acc.items() 
                            if uid_iter != uid])
                
                if len_sum != 0:
                    uid_expectation[uid] = sum_sum / len_sum
                else:
                    uid_expectation[uid] = 0

        # 计算redo概率, redo长度与非redo长度
        for question_uid, uids in question_uid_2_uids.items():
            # 计算redo概率
            uids_group_redo = {
                uid: {
                    "uid_responses_redo_sum": sum([redo_num for redo_num in uid_2_redo[uid]]),
                    "uid_responses_num": len(uid_2_redo[uid]),
                }
                for uid in uids
            }
            
            for uid in uids_group_redo:
                len_sum = sum([uid_data["uid_responses_num"]
                            for uid_iter, uid_data in uids_group_redo.items() 
                            if uid_iter != uid])
                
                sum_sum = sum([uid_data["uid_responses_redo_sum"]
                            for uid_iter, uid_data in uids_group_redo.items() 
                            if uid_iter != uid])
                
                uid_2_redo_rate[uid] = sum_sum / len_sum

            # 计算redo len的长度
            uids_group_redo_len = {
                uid: {
                    "uid_responses_redo_len_sum": sum([redo_num for redo_num in uid_2_redo_len[uid]]),
                    "uid_responses_num": len(uid_2_redo_len[uid]),
                }
                for uid in uids
            }
            
            for uid in uids_group_redo_len:
                len_sum = sum([uid_data["uid_responses_num"]
                            for uid_iter, uid_data in uids_group_redo_len.items() 
                            if uid_iter != uid])
                
                sum_sum = sum([uid_data["uid_responses_redo_len_sum"]
                            for uid_iter, uid_data in uids_group_redo_len.items() 
                            if uid_iter != uid])

                if len_sum != 0:
                    uid_2_redolen[uid] = sum_sum / len_sum
                else:
                    uid_2_redolen[uid] = 0

            # 计算noredo len的长度
            uids_group_noredo_len = {
                uid: {
                    "uid_responses_noredo_len_sum": sum([redo_num for redo_num in uid_2_noredo_len[uid]]),
                    "uid_responses_num": len(uid_2_noredo_len[uid]),
                }
                for uid in uids
            }
            
            for uid in uids_group_noredo_len:
                len_sum = sum([uid_data["uid_responses_num"]
                            for uid_iter, uid_data in uids_group_noredo_len.items() 
                            if uid_iter != uid])
                
                sum_sum = sum([uid_data["uid_responses_noredo_len_sum"]
                            for uid_iter, uid_data in uids_group_noredo_len.items() 
                            if uid_iter != uid])

                if len_sum != 0:
                    uid_2_noredolen[uid] = sum_sum / len_sum
                else:
                    uid_2_noredolen[uid] = 1000000000000

        # 计算uid条样本选择重做时，需要额外付出的期望长度
        for uid in uid_2_redo_rate:
            redo_rate = uid_2_redo_rate[uid]
            redo_len = uid_2_redo_len[uid]
            noredo_len = uid_2_noredo_len[uid]
            if redo_rate == 1:
                uid_2_redo_extra_len[uid] = redo_len
            else:
                uid_2_redo_extra_len[uid] = (redo_rate * redo_len + (1 - redo_rate) * noredo_len)  / (1 - redo_rate)

        # 计算每个uid样本的平均长度
        for length, data_info in zip(lengths, data_info_lst): 
            if data_info['redo']:
                uid_2_fulllen[data_info['uid']].append(length + uid_2_redo_extra_len[data_info['uid']])
            else:
                uid_2_fulllen[data_info['uid']].append(length)

        for uid in uid_2_fulllen:
            uid_2_mean_len[uid] = sum(uid_2_fulllen) / len(uid_2_fulllen)

        # 根据data_info和redo_expection重新构造return_dict
        for i, (length, data_info) in enumerate(zip(lengths, data_info_lst)):
            length_coef = uid_2_mean_len[data_info['uid']] / length
            if data_info['redo']:
                score = uid_expectation[data_info['uid']] * length_coef
                data_info['score'] = score
                if write_flag:
                    print("[REDO!]", score)
                    write_flag = False
            else:
                score = data_info['score']

            valid_response_length = data_info['valid_response_length']
            overlong_reward = data_info['overlong_reward']
            reward = score + overlong_reward / 2
            reward_tensor[i, valid_response_length - 1] = reward

            for key, value in data_info.items():
                reward_extra_info[key].append(value)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
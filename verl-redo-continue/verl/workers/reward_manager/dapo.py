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

import re
from typing import Optional
import os

REDO_STR=os.environ['REDO_STR']

def last_boxed_pos_regex(s: str) -> Optional[int]:
    _pattern = re.compile(r'boxed\{.*?\}')
    matches = list(_pattern.finditer(s))
    if not matches:
        return None
    last = matches[-1]
    return last.start() 

def check_strict_redo(response, redo_str):
    box_id = last_boxed_pos_regex(response)
    if (redo_str in response[-len(response) // 3:]) and ((box_id is None) or (response.rfind(redo_str) - box_id > 40)):
        return True
    return False
    


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

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

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
        # uid_expectation = {}
        uid_2_acc = defaultdict(list)
        data_info_lst = []
        redo_tag = REDO_STR
        uid_trunc_expect_success = {} 
        uid_redo_penalty = {} 
        write_flag = True
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

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

        uid_event_counts = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'redo': 0})

        for info in data_info_lst:
            uid = info['uid']
            if info['redo']:
                uid_event_counts[uid]['redo'] += 1
            else:
                acc_val = 1 if info.get('acc') else 0  # 你的 compute_score 给的 acc（0/1）
                if acc_val == 1:
                    uid_event_counts[uid]['correct'] += 1
                else:
                    uid_event_counts[uid]['wrong'] += 1

        R_penalty = 0
        for uids in question_uid_2_uids.items():
            total_correct = sum(uid_event_counts[u]['correct'] for u in uids)
            total_wrong   = sum(uid_event_counts[u]['wrong']   for u in uids)
            total_redo    = sum(uid_event_counts[u]['redo']    for u in uids)

            for u in uids:
                corr_ex = total_correct - uid_event_counts[u]['correct']
                wrong_ex = total_wrong   - uid_event_counts[u]['wrong']
                redo_ex  = total_redo    - uid_event_counts[u]['redo']
                all_ex   = corr_ex + wrong_ex + redo_ex

                P_R = 0
                if all_ex <= 0:
                    a_hat = 0.0; b_hat = 0.0; c_hat = 1.0  
                else:
                    a_hat = corr_ex / all_ex
                    b_hat = wrong_ex / all_ex
                    c_hat = redo_ex  / all_ex

                if c_hat >= 1.0 - 1e-12:
                    P_R = -1.0
                else:
                    max_round=self.overlong_buffer_cfg.max_redo_round
                    if max_round < 1:
                        P_R = a_hat / (1.0 - c_hat) 
                    else:
                        P_R = a_hat * (1 - c_hat ** max_round) / (1.0 - c_hat) 
                    R_penalty = a_hat / (1.0 - c_hat) - P_R

                uid_trunc_expect_success[u] = P_R
                uid_redo_penalty[u] = R_penalty
        
        for i, data_info in enumerate(data_info_lst):
            if data_info['redo']:
                score = uid_trunc_expect_success[data_info['uid']]
                data_info['score'] = score
                data_info['penalty'] = uid_redo_penalty[data_info['uid']]
                if write_flag:
                    print("[REDO!]", P_R)
                    print("[penalty]",a_hat, c_hat, R_penalty)
                    write_flag = False
            else:
                score = data_info['score']
                data_info['penalty'] = 0.0

            valid_response_length = data_info['valid_response_length']
            overlong_reward = data_info['overlong_reward']
            reward = score + overlong_reward
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
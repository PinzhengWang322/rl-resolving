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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
import random

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.torch_functional import tokenize_and_postprocess_data
from verl.utils.model import compute_position_id_with_mask
import os

import re
from typing import Optional



def last_boxed_pos_regex(s: str) -> Optional[int]:
    _pattern = re.compile(r'boxed\{.*?\}')
    matches = list(_pattern.finditer(s))
    if not matches:
        return None
    last = matches[-1]
    return last.start() 

REDO_STR=os.environ['REDO_STR']

def check_strict_redo(response, redo_str):
    box_id = last_boxed_pos_regex(response)
    if (redo_str in response[-len(response) // 3:]):
        return True
    return False
    

def record_experiment(config, tokenizer, batch, reward_tensor, global_steps):
    exp_dir = config.trainer.experiences_dir
    os.makedirs(exp_dir, exist_ok=True)
    
    subdir = os.path.join(exp_dir, f"{(global_steps // 50) * 50}-{(global_steps // 50) * 50 + 49}")
    os.makedirs(subdir, exist_ok=True)
    
    file_path = os.path.join(subdir, f"exp_{global_steps}.txt")
    gen_texts = [tokenizer.decode(ids[msks], skip_special_tokens=False) for ids, msks in zip(batch.batch['input_ids'], batch.batch["attention_mask"].bool())]
    returns = batch.batch['returns'][:, 0].tolist()
    rewards = reward_tensor.sum(dim=-1).tolist()
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        for text, ret, reward in zip(gen_texts, returns, rewards):
            f.write(f"Generated Text:\n{text}\nReturn: {ret}\n" + "-" * 50 + "\n" + f"Rewards: {reward}\n")

def make_redo_gen_batch(redo_str, tokenizer, gen_batch_output, max_len=8192):
    input_ids_lst, attention_mask_lst, responses_lst = [], [], []
    prompt_len, response_len = gen_batch_output.batch['prompts'].shape[-1], gen_batch_output.batch['responses'].shape[-1]
    for i in range(len(gen_batch_output.batch['attention_mask'])):
        attention_mask = gen_batch_output.batch['attention_mask'][i]
        input_ids = gen_batch_output.batch['input_ids'][i]
        response_ids = gen_batch_output.batch['responses'][i]
        prompt_ids = gen_batch_output.batch['prompts'][i]
        
        main_response_ids = response_ids[attention_mask[-response_ids.shape[0]:] == 1]
        eos_str = tokenizer.eos_token
        decoded_str = tokenizer.decode(main_response_ids.tolist())
        if decoded_str.count(REDO_STR) > 0:
            redo_str = decoded_str.split(REDO_STR)[0] + REDO_STR + eos_str
            if not REDO_STR.endswith('.'):
                redo_str = decoded_str.split(REDO_STR)[0] + REDO_STR + '.' + eos_str
            if not check_strict_redo(redo_str, REDO_STR):
                redo_str = decoded_str
        else:
            redo_str = decoded_str
        
        redo_response_ids, redo_response_attention_mask = \
        tokenize_and_postprocess_data(redo_str, tokenizer, max_len, 
                                      tokenizer.pad_token_id, left_pad=False, truncation = 'right')
        
        redo_input_ids = torch.cat((prompt_ids, redo_response_ids[0]), dim=0)
        redo_attention_mask = torch.cat((attention_mask[:prompt_len], redo_response_attention_mask[0]), dim=0)
        input_ids_lst.append(redo_input_ids)
        attention_mask_lst.append(redo_attention_mask)
        responses_lst.append(redo_response_ids)
    
    input_ids = torch.stack(input_ids_lst).squeeze()
    attention_mask = torch.stack(attention_mask_lst).squeeze()
    responses_ids = torch.stack(responses_lst).squeeze()

    return DataProto.from_dict(
            tensors = dict(
                input_ids = input_ids.int(),
                attention_mask = attention_mask.int(),
                responses = responses_ids.int(),
                position_ids = gen_batch_output.batch['position_ids'].int(),
                prompts = gen_batch_output.batch['prompts'].int()
            )
        )

def make_continue_writing_gen_batch(tokenizer, gen_batch_output, trunc_range=(0, 0.8),max_len=8192):
    input_ids_lst, attention_mask_lst, position_ids_lst = [], [], []
    for i in range(len(gen_batch_output.batch['attention_mask'])):
        pos_ids = gen_batch_output.batch['attention_mask'][i]
        prompt_ids = gen_batch_output.batch['prompts'][i]
        response_ids = gen_batch_output.batch['responses'][i]
        non_zero_prompt_mask = pos_ids[:prompt_ids.shape[-1]] != 0
        non_zero_response_mask = pos_ids[prompt_ids.shape[-1]:] != 0
        unpad_prompt_ids = prompt_ids[non_zero_prompt_mask]
        unpad_response_ids = response_ids[non_zero_response_mask]
        unpad_response = tokenizer.decode(unpad_response_ids, skip_special_tokens=False)
        unpad_response = unpad_response.split(REDO_STR)[0]
        
        unpad_response_ids = tokenizer(unpad_response, return_tensors="pt")['input_ids'][0]
        if tokenizer.bos_token_id is not None and unpad_response_ids[0].item() == tokenizer.bos_token_id:
            unpad_response_ids = unpad_response_ids[1:]

        prob = 0.3
        if random.random() < prob:
            trunc_ratio = 0
        else:
            trunc_ratio = torch.empty(1).uniform_(*trunc_range).item()

        unpad_trunc_response_ids = unpad_response_ids[:int(unpad_response_ids.shape[-1] * trunc_ratio)]
        unpad_trunc_input_ids = torch.cat([unpad_prompt_ids, unpad_trunc_response_ids ])

        new_prompt = tokenizer.decode(unpad_trunc_input_ids.int(), skip_special_tokens=False)
        input_ids, attention_mask = \
        tokenize_and_postprocess_data(new_prompt, tokenizer, max_len, 
                                      tokenizer.pad_token_id, truncation='right')
        position_ids = compute_position_id_with_mask(attention_mask)

        input_ids_lst.append(input_ids)
        attention_mask_lst.append(attention_mask)
        position_ids_lst.append(position_ids)
    
    input_ids = torch.stack(input_ids_lst).squeeze()
    attention_mask = torch.stack(attention_mask_lst).squeeze()
    position_ids = torch.stack(position_ids_lst).squeeze()
    
    return DataProto.from_dict(
            tensors = dict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                position_ids = position_ids
            )
        )


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in (self.config.trainer.profile_steps or [])
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch.meta_info['num_samples'] = self.config.actor_rollout_ref.rollout.n
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                        continue_writing_batch = make_continue_writing_gen_batch(self.tokenizer, gen_batch_output, max_len = int(self.config.data.max_response_length * 0.8 + 1) + 2048)
                        continue_writing_batch.meta_info['num_samples'] = self.config.actor_rollout_ref.rollout.continue_writing_n
                        continue_writing_batch_out = self.actor_rollout_wg.generate_sequences(continue_writing_batch)

                        continue_writing_batch_out = make_redo_gen_batch(REDO_STR, self.tokenizer, continue_writing_batch_out, self.config.data.max_response_length)
                        new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n * self.config.actor_rollout_ref.rollout.continue_writing_n, interleave=True)
                        new_batch.non_tensor_batch['question_uid'] = np.repeat(
                            np.array(
                                [str(uuid.uuid4()) for _ in range(len(gen_batch))],
                                dtype=object
                            ),
                            repeats=self.config.actor_rollout_ref.rollout.n * self.config.actor_rollout_ref.rollout.continue_writing_n
                        )

                        new_batch.non_tensor_batch['uid'] = np.repeat(
                            np.array(
                                [str(uuid.uuid4()) for _ in range(len(gen_batch) * self.config.actor_rollout_ref.rollout.n)],
                                dtype=object
                            ),
                            repeats=self.config.actor_rollout_ref.rollout.continue_writing_n
                        )

                    new_batch = new_batch.union(continue_writing_batch_out)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        
                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n * self.config.actor_rollout_ref.rollout.continue_writing_n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    
                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.actor_rollout_ref
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    record_experiment(self.config, self.tokenizer, batch, reward_tensor, self.global_steps)
                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1


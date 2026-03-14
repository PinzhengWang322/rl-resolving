#!/usr/bin/env bash

ulimit -n 4096
project_name='YOUR-PROJECT-NAME'
exp_name='YOUR-EXP-NAME'

export WANDB_DIR="./wandb/$exp_name"
export WANDB_MODE=offline
mkdir -p "$WANDB_DIR/wandb"
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))
enable_overlong_buffer=True
overlong_buffer_len=$((2048))
overlong_penalty_factor=1.0
max_redo_round=5
safe_redo_round=4
redo_penalty=0.0




loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=score
max_num_gen_batches=10
train_prompt_bsz=32
gen_prompt_bsz=$((train_prompt_bsz * 3))
train_prompt_mini_bsz=32
n_resp_per_prompt=8
n_resp_per_prompt_continue_writing=8


# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
# Paths
RAY_DATA_HOME=${PWD}
MODEL_PATH=${MODEL_PATH:-"/path/to/YOUR_MODEL"}
EXP_DIR=${EXP_DIR:-"${RAY_DATA_HOME}/experiences/experiences-${exp_name}"}
VAL_EXP_DIR=${VAL_EXP_DIR:-"${RAY_DATA_HOME}/experiences/val_experiences-${exp_name}"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/path/to/your_train_data"}
# Algorithm
temperature=1.0
val_temperature=0.6
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
eta=0.1
max_prop=1

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length+ max_response_length))
offload=True
gen_tp=2
rollout_backend=vllm

# "${RAY_DATA_HOME}/data_redo2/aime-2024.parquet",

export REDO_STR="s better to redo the question"
export HYDRA_FULL_ERROR=1


ray stop --force && ray start --head --num-cpus=16
python3 -u -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=["path/to/your_val_data"] \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.dataloader_num_workers=4 \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    +actor_rollout_ref.rollout.continue_writing_n=${n_resp_per_prompt_continue_writing} \
    actor_rollout_ref.rollout.name=${rollout_backend} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    +actor_rollout_ref.adj_eta=${eta} \
    +actor_rollout_ref.adj_max_prop=${max_prop} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.micro_batch_size=${rollout_mbsz} \
    actor_rollout_ref.rollout.val_kwargs.n=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    +reward_model.overlong_buffer.max_redo_round=${max_redo_round} \
    +reward_model.overlong_buffer.safe_redo_round=${safe_redo_round} \
    +reward_model.overlong_buffer.redo_penalty=${redo_penalty} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    +trainer.experiences_dir=$EXP_DIR \
    +trainer.val_experiences_dir=$VAL_EXP_DIR \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=40 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \

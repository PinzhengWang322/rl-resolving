#!/bin/bash

# 定义模型路径和生成设置
declare -A MODEL_PATHS=(
    ["Qwen2.5-7B"]="/path/tp/Qwen2.5-7B"
)

# 参数设置
MAX_LEN=32768
TEMP=0.6
GEN_NUM=16
TOP_P=0.95
DP_SIZE=8

mkdir -p ./logs

export NUM_CPUS=16
for MODEL_NAME in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$MODEL_NAME]}
    ray stop --force
    sleep 15

    # 设置EXP_NAME
    EXP_NAME="${MODEL_NAME}"
    echo "Running $EXP_NAME..."

    python self_redo.py \
        data_lst=[aime24,amc] \
        task=self_redo \
        task.unsure_str="better to redo the question." \
        task.redo_prompt_path="./prompts/self_redo/simple_redo3.txt" \
        model.path=$MODEL_PATH \
        model.dp_size=$DP_SIZE \
        task.max_loop=0 \
        generation.save_dir=../results/data_model_test/temp0.6_sample16_32k_$EXP_NAME \
        generation.gen_num=$GEN_NUM \
        generation.temperature=$TEMP \
        generation.top_p=$TOP_P \
        generation.maxlen=$MAX_LEN \
        2>&1 | tee ./logs/${EXP_NAME}.log 
done


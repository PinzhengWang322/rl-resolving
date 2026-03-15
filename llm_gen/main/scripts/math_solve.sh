#!/bin/bash

# 定义模型路径和生成设置
declare -A MODEL_PATHS=(
    ["Qwen2.5-7B"]="/path/to/Qwen2.5-7B"
)
MAX_LEN=32768
TEMP=0.6
GEN_NUM=16
TOP_P=0.95
DP_SIZE=8
TP_SIZE=1

mkdir -p ./math_solve_logs
export NUM_CPUS=16
RESULTS_BASE="../results/data_ER/temp0.6_sample16_32k_"

for MODEL_NAME in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$MODEL_NAME]}

    # 设置EXP_NAME
    EXP_NAME="${MODEL_NAME}"
    echo "Running $EXP_NAME..."

    python math_solve.py \
        data_lst=[aime24,aime25,math500,amc,minerva_math,olympiadbench] \
        model.path=$MODEL_PATH \
        model.dp_size=$DP_SIZE \
        generation.save_dir=${RESULTS_BASE}${EXP_NAME} \
        generation.gen_num=$GEN_NUM \
        generation.temperature=$TEMP \
        generation.top_p=$TOP_P \
        generation.maxlen=$MAX_LEN \
        2>&1 | tee ./math_solve_logs/${EXP_NAME}.log
done

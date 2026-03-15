#!/bin/bash

# 定义数据路径和生成配置
BASE_PATH="/path/to/Qwen2.5-7B-data"

FILE_NAMES=(
    "aime24"
    "aime25"
    "amc"
    "gsm8k"
    "gpqa_diamond"
)

# 参数设置
SAMPLE_NUM=1024
SAMPLE_TIME=5

TOKENIZER="/path/tp/Qwen2.5-7B"

export NUM_CPUS=16
for FILE_NAME in "${FILE_NAMES[@]}"; do
    FILE_PATH="${BASE_PATH}/${FILE_NAME}"

    # 设置EXP_NAME
    EXP_NAME="${FILE_NAME}"
    echo "Val $EXP_NAME..."
    
    mkdir -p ./test_val_logs 
    python cal_redo_v2.py \
        --path="$FILE_PATH" \
        --tokenizer="$TOKENIZER"
        2>&1 | tee "./test_val_logs/${EXP_NAME}.log"
done


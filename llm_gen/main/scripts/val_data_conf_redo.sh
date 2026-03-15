#!/bin/bash

declare -A DATA_PATHS=(
    ["Qwen2.5-7B"]="/path/tp/Qwen2.5-7B-data"
)

declare -A MODEL_PATHS=(
    ["Qwen2.5-7B"]="/path/tp/Qwen2.5-7B"
)


FILE_NAMES=(
    "aime24"
    "aime25"
    "amc"
    "gsm8k"
    "gpqa_diamond"
)


export NUM_CPUS=8
for MODEL_NAME in "${!DATA_PATHS[@]}"; do
    for FILE_NAME in "${FILE_NAMES[@]}"; do
        FILE_PATH="${DATA_PATHS[${MODEL_NAME}]}/${FILE_NAME}"
        # 设置EXP_NAME
        EXP_NAME="${FILE_NAME}"
        echo "Val $EXP_NAME..."

        mkdir -p ./val_redo_conf/${MODEL_NAME}
        
	TOKENIZER="${MODEL_PATHS[${MODEL_NAME}]}"
        python cal_redo_conf.py \
            --path="$FILE_PATH" \
	    --tokenizer="$TOKENIZER" \
            2>&1 | tee "./val_redo_conf/${MODEL_NAME}/${EXP_NAME}.log"
    done
done


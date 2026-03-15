#!/bin/bash

# 根目录路径
BASE_DIR="/nvme1/wpz/llm_gen/results/sc"

# 数据集参数（可以按需修改或传参）
DATASETS="aime math500"

# 遍历符合条件的目录
for DIR in "$BASE_DIR"/Qwen2.5-Mit-sc-1.5b-sc_noverify; do
    # 判断是否为目录
    if [ -d "$DIR" ]; then
        echo "Processing: $DIR"

        python merge_sc_datasets.py \
            --root_dir "$DIR" \
            --output_dir "$DIR/merged_chains" \
            --datasets $DATASETS

        echo "Finished: $DIR"
        echo "------------------------"
    fi
done

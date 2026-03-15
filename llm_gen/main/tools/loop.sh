#!/usr/bin/env bash
set -eu

ROOT="/nvme1/wpz/llm_gen/results/sc"                # 顶层模型结果目录
TOKENIZER="/nvme/hf_models/Qwen2.5-Math-1.5B-Instruct"

for model_dir in "$ROOT"/*; do
  # 只遍历 initial/ 与 correct_loop8/
  for stage in initial correct_loop8; do
    stage_dir="${model_dir}/${stage}"
    [ -d "$stage_dir" ] || continue

    # 遍历 stage 下的 aime 和 math500 子目录
    for ds_dir in "$stage_dir"/*; do
      [ -d "$ds_dir" ] || continue

      ds_name=$(basename "$ds_dir")
      if [[ "$ds_name" != "aime" && "$ds_name" != "math500" ]]; then
        continue  # 跳过非目标数据集
      fi

      echo ">>> Evaluating: $ds_dir"
      python math_eval.py \
        --dataset_dir "$ds_dir" \
        --tokenizer_path "$TOKENIZER"
    done
  done
done

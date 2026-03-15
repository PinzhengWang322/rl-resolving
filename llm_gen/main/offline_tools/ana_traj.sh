#!/bin/bash

# # 设置基本路径
# base_path="/nvme1/wpz/llm_gen/results/sc"
# log_dir="logs"

# # 数据集列表
# datasets=("aime24" "aime25" "math500")

# # 创建日志目录（如果不存在）
# mkdir -p "$log_dir"

# # 遍历模型目录
# for model_dir in "$base_path"/Qwen2.5-Mit-1.5B-v_c; do
#   model_name=$(basename "$model_dir")
  
#   for dataset in "${datasets[@]}"; do
#     ds_path="$model_dir/merged_chains/$dataset"
#     save_dir="$model_dir/merged_chains/V2_math_figs_$dataset"
#     log_file="$log_dir/${model_name}-${dataset}.log"
    
#     if [ -d "$ds_path" ]; then
#       # if [ -d "$save_dir" ]; then
#       #   echo "Skipping $model_name on $dataset: $save_dir already exists."
#       # else
#       echo "Running analysis for $model_name on $dataset"
#       python analyze_trajectory.py --ds_path "$ds_path" --save_dir "$save_dir" 2>&1 | tee "$log_file"
#       # fi
#     else
#       echo "Warning: $ds_path does not exist, skipping..."
#     fi
#   done
# done

#!/bin/bash

# 设置基本路径
base_path="/nvme1/wpz/llm_gen/results/sc"
log_dir="logs"

# 数据集列表
datasets=("aime24" "aime25" "math500")

# 创建日志目录（如果不存在）
mkdir -p "$log_dir"

# 遍历模型目录
for model_dir in "$base_path"/Qwen2.5-Mit-1.5B-v_c; do
  model_name=$(basename "$model_dir")
  
  for dataset in "${datasets[@]}"; do
    ds_path="$model_dir/merged_chains/$dataset"
    save_dir="$model_dir/merged_chains/diff_math_figs_$dataset"
    log_file="$log_dir/${model_name}-${dataset}.log"
    
    if [ -d "$ds_path" ]; then
      # if [ -d "$save_dir" ]; then
      #   echo "Skipping $model_name on $dataset: $save_dir already exists."
      # else
      echo "Running analysis for $model_name on $dataset"
      python analyze_trajectory.py --ds_path "$ds_path" --save_dir "$save_dir" --by_difficulty 2>&1 | tee "$log_file"
      # fi
    else
      echo "Warning: $ds_path does not exist, skipping..."
    fi
  done
done

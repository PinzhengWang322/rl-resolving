import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from sklearn.cluster import KMeans
from omegaconf import OmegaConf
from tqdm import tqdm
import random
import math
from tools.math_eval import math_lst_eval
from lighteval.metrics.utils.extractive_match_utils import (  # noqa: F401
    ExprExtractionConfig,
    ExtractionTarget,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.utils.language import Language
from lighteval.tasks.requests import Doc
from lighteval.utils.timeout import timeout

def extract_target_from_pred_with_timeout(pred, pred_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds):
    try:
        return timeout(timeout_seconds)(extract_target_from_pred)(
            pred, pred_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds
        )
    except Exception as e:
        print(f"Error during prediction extraction: {e}")
        return timeout(timeout_seconds)(extract_target_from_pred)(
            "", pred_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds
        )



def parse_verification(response):
    """从验证响应中解析验证结果"""
    # print(response[-100:])
    # print('-' * 100)
    if "{Incorrect}" in response[-100:]:
        return 0
    else:
        return 1

def evaluate_verify_dataset(current_dir, dataset, name, tokenizer, cfg):
    # 创建目录
    figs_dir = os.path.join(current_dir, 'figs')
    figs_diff_dir = os.path.join(current_dir, f'{name}_diff')
    metrics_dir = os.path.join(current_dir, 'metrics')
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(figs_diff_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. 预处理数据：为每个样本计算正确性和验证结果
    processed_data = []
    for item in tqdm(dataset, desc="Processing samples"):
        # 提取关键信息
        prompt = item['conversations'][0]['content']
        gt_answer = str(item['gt_answer'])
        model_answer = item['conversations'][1]['content']  # 模型的原始答案
        verification_response = item['responses'][0]  # 验证响应
        
        # 计算回答是否正确
        correctness = math_lst_eval([gt_answer], [model_answer])[0]
        
        # 解析验证结果
        verification = parse_verification(verification_response)
        
        # 计算验证是否正确（如果验证结果与真实正确性一致）
        verification_correct = 1 if verification == correctness else 0 if verification is not None else None
        
        # 估算token数量（问题+回答+验证）
        tokens = len(tokenizer.encode(prompt + model_answer + verification_response))
        
        processed_data.append({
            'id': item['id'],
            'prompt': prompt,
            'gt_answer': gt_answer,
            'model_answer': model_answer,
            'correctness': correctness,
            'verification': verification,
            'verification_correct': verification_correct,
            'tokens': tokens
        })
    
    # 2. 按问题分组并计算难度
    problem_groups = defaultdict(list)
    for item in processed_data:
        problem_groups[item['prompt']].append(item)
    
    # 计算每个问题的难度（正确回答的比例）
    problem_difficulties = {}
    for prompt, items in problem_groups.items():
        total = len(items)
        correct = sum(item['correctness'] for item in items)
        problem_difficulties[prompt] = correct / total if total > 0 else 0.0
    
    # 3. 将问题分为5个难度等级
    sorted_problems = sorted(problem_difficulties.items(), key=lambda x: x[1])
    problem_buckets = np.array_split(sorted_problems, 10)
    
    # 4. 计算整体指标
    def calculate_metrics(data):
        correctness = [item['correctness'] for item in data]
        verification_correct = [item['verification_correct'] for item in data if item['verification_correct'] is not None]
        
        # 基础指标
        accuracy = np.mean(correctness) if correctness else 0.0
        verification_accuracy = np.mean(verification_correct) if verification_correct else 0.0
        
        # 条件指标
        correct_items = [item for item in data if item['correctness'] == 1]
        incorrect_items = [item for item in data if item['correctness'] == 0]
        
        correct_verification_correct = np.mean(
            [item['verification_correct'] for item in correct_items if item['verification_correct'] is not None]
        ) if correct_items else 0.0
        
        incorrect_verification_correct = np.mean(
            [item['verification_correct'] for item in incorrect_items if item['verification_correct'] is not None]
        ) if incorrect_items else 0.0
        
        return {
            'accuracy': accuracy,
            'verification_accuracy': verification_accuracy,
            'correct_is_correct': correct_verification_correct,
            'wrong_is_wrong': incorrect_verification_correct
        }
    
    # 整体指标
    overall_metrics = calculate_metrics(processed_data)
    
    # 5. 难度分桶指标
    bucket_metrics = []
    for i, bucket in enumerate(problem_buckets):
        bucket_data = []
        for prompt, _ in bucket:
            bucket_data.extend(problem_groups[prompt])
        metrics = calculate_metrics(bucket_data)
        metrics['difficulty'] = np.mean([float(d) for _, d in bucket])
        bucket_metrics.append(metrics)

    # 6. Majority Voting曲线 - 最终优化版本
    def majority_voting_curve(data):
        # 获取最大样本数量
        max_samples = max(len(items) for items in problem_groups.values())
        
        # 从2开始（n=1没有投票意义）
        n_values = [2**i for i in range(1, int(np.log2(max_samples)) + 1)] if max_samples >= 2 else []
        
        accuracies = []
        token_counts = []
        
        
        language = Language.ENGLISH
        formatted_doc = Doc(
        task_name="",
        query="",
        choices="",
        gold_index=0,
        instruction="",
    ) # dummy doc
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0))
        pred_extraction_regexes = get_extraction_regexes(formatted_doc, pred_extraction_target, language)
        
        for n in n_values:
            correct_count = 0
            total_tokens = 0
            num_problems = 0  # 有足够样本的问题数量
            
            for prompt, items in problem_groups.items():
                if len(items) < n:
                    continue
                    
                num_problems += 1
                
                # 随机采样n个样本
                sampled = random.sample(items, n)
                
                # 创建规范化答案到正确性的映射
                answer_map = {}
                extracted_answers = []
                
                
                for item in sampled:
                    # 提取规范化答案
                    try:
                        extracted = extract_target_from_pred_with_timeout(
                            item['model_answer'],
                            pred_extraction_regexes,
                            fallback_mode="first_match",
                            extraction_mode="any_match",
                            timeout_seconds=5
                        )
                        # 处理提取结果
                        if isinstance(extracted, (list, tuple)) and len(extracted) > 1:
                            normalized = re.sub(r'\s+', '', extracted[1]).lower()
                        else:
                            normalized = re.sub(r'\s+', '', str(extracted)).lower()
                    except Exception:
                        normalized = re.sub(r'\s+', '', item['model_answer']).lower()
                    
                    # 存储提取的答案
                    extracted_answers.append(normalized)
                    # 记录答案的正确性（复用预处理结果）
                    answer_map[normalized] = item['correctness']
                
                # 统计每个答案的出现次数
                answer_counts = {}
                for answer in extracted_answers:
                    answer_counts[answer] = answer_counts.get(answer, 0) + 1
                
                # 找出票数最多的答案
                max_votes = 0
                selected_answer = None
                for answer, count in answer_counts.items():
                    if count > max_votes:
                        max_votes = count
                        selected_answer = answer
                    elif count == max_votes and selected_answer is not None:
                        # 平票时随机选择
                        if random.random() < 0.5:
                            selected_answer = answer
                
                # 检查选择的答案是否正确（复用预处理结果）
                is_correct = answer_map.get(selected_answer, 0) if selected_answer else 0
                correct_count += is_correct
                
                # 计算token消耗
                total_tokens += sum(item['tokens'] for item in sampled)
            
            # 计算该n值下的准确率
            accuracy = correct_count / num_problems if num_problems > 0 else 0
            accuracies.append(accuracy)
            token_counts.append(total_tokens)
        
        return n_values, accuracies, token_counts
    
    # 整体MV曲线 - 修改绘图部分
    n_values, mv_acc, mv_tokens = majority_voting_curve(processed_data)
    if n_values:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 准确率折线图
        ax1.plot(n_values, mv_acc, 'b-o', label='Accuracy')
        ax1.set_xlabel('Number of Samples')
        ax1.set_ylabel('Accuracy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xscale('log')
        ax1.set_xticks(n_values)
        ax1.set_xticklabels([str(n) for n in n_values], rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 创建第二个y轴用于token数量
        ax2 = ax1.twinx()
        
        # 柱状图：token数量
        bar_width = 0.4 * np.array(n_values)  # 调整柱子宽度
        ax2.bar(n_values, mv_tokens, width=bar_width, alpha=0.6, color='r')
        ax2.set_ylabel('Total Tokens', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 添加图例
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
        
        plt.title('Majority Voting Performance')
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, f'mv_{name}.png'), dpi=150)
        plt.close(fig)
    
    # 7. 每个难度分桶的MV曲线 - 修改绘图部分
    for i, metrics in enumerate(bucket_metrics):
        bucket_data = []
        for prompt, _ in problem_buckets[i]:
            bucket_data.extend(problem_groups[prompt])
        
        n_values, mv_acc, mv_tokens = majority_voting_curve(bucket_data)
        if n_values:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # 准确率折线图
            ax1.plot(n_values, mv_acc, 'b-o', label='Accuracy')
            ax1.set_xlabel('Number of Samples')
            ax1.set_ylabel('Accuracy', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_xscale('log')
            ax1.set_xticks(n_values)
            ax1.set_xticklabels([str(n) for n in n_values], rotation=45)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 创建第二个y轴用于token数量
            ax2 = ax1.twinx()
            
            # 柱状图：token数量
            bar_width = 0.4 * np.array(n_values)  # 调整柱子宽度
            ax2.bar(n_values, mv_tokens, width=bar_width, alpha=0.6, color='r')
            ax2.set_ylabel('Total Tokens', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # 添加图例
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
            
            plt.title(f'Majority Voting Performance (Difficulty Bucket {i+1})')
            plt.tight_layout()
            plt.savefig(os.path.join(figs_diff_dir, f'mv_diff{i+1}.png'), dpi=150)
            plt.close(fig)
    
    # 8. 验证采样模拟
    def verification_simulation(data, max_rounds, num_simulations=10):
        results = []
        
        for prompt, items in problem_groups.items():
            # 对每个问题进行多次独立模拟
            for sim_idx in range(num_simulations):
                rounds_used = 0
                selected_answer = None
                tokens_used = 0
                found_correct = False
                
                # 有放回采样 - 创建采样池
                sample_pool = items.copy()
                
                for j in range(max_rounds):
                    # 如果采样池为空，重新填充（有放回）
                    if not sample_pool:
                        sample_pool = items.copy()
                    
                    # 随机选择一个样本
                    item = random.choice(sample_pool)
                    rounds_used += 1
                    tokens_used += item['tokens']
                    
                    # 如果验证为正确，使用当前答案
                    if item['verification'] == 1:
                        selected_answer = item['model_answer']
                        found_correct = True
                        break
                    
                    # 从采样池中移除当前样本（有放回采样不需要移除）
                    # 保持采样池不变以实现有放回采样
                    
                # 如果达到最大轮数仍未找到验证正确的样本
                if not found_correct:
                    # 使用最后一个采样样本的答案
                    selected_answer = item['model_answer']
                
                # 检查答案是否正确
                is_correct = math_lst_eval([items[0]['gt_answer']], [selected_answer])[0]
                results.append((is_correct, rounds_used, tokens_used // num_simulations))
        
        return results
    
    # 整体验证模拟 - 修改绘图部分
    max_rounds_list = [2**i for i in range(0, 7)]  # 1, 2, 4, 8, 16, 32
    sim_acc = []
    sim_tokens = []
    
    for max_rounds in max_rounds_list:
        results = verification_simulation(processed_data, max_rounds)
        accuracy = np.mean([r[0] for r in results]) if results else 0
        total_tokens = np.sum([r[2] for r in results])
        sim_acc.append(accuracy)
        sim_tokens.append(total_tokens)
    
    if max_rounds_list:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 准确率折线图
        ax1.plot(max_rounds_list, sim_acc, 'b-o', label='Accuracy')
        ax1.set_xlabel('Max Rounds')
        ax1.set_ylabel('Accuracy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xticks(max_rounds_list)
        ax1.set_xticklabels([str(x) for x in max_rounds_list])
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 创建第二个y轴用于token数量
        ax2 = ax1.twinx()
        
        # token数量折线图
        ax2.plot(max_rounds_list, sim_tokens, 'r--s', label='Total Tokens')
        ax2.set_ylabel('Total Tokens', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 添加图例
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
        
        plt.title('Verification Sampling Simulation')
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, f'verification_{name}.png'), dpi=150)
        plt.close(fig)
    
    # 9. 每个难度分桶的验证模拟 - 修改绘图部分
    for i, metrics in enumerate(bucket_metrics):
        bucket_data = []
        for prompt, _ in problem_buckets[i]:
            bucket_data.extend(problem_groups[prompt])
        
        sim_acc = []
        sim_tokens = []
        
        for max_rounds in max_rounds_list:
            results = verification_simulation(bucket_data, max_rounds)
            accuracy = np.mean([r[0] for r in results]) if results else 0
            total_tokens = sum(r[2] for r in results)
            sim_acc.append(accuracy)
            sim_tokens.append(total_tokens)
        
        if max_rounds_list:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # 准确率折线图
            ax1.plot(max_rounds_list, sim_acc, 'b-o', label='Accuracy')
            ax1.set_xlabel('Max Rounds')
            ax1.set_ylabel('Accuracy', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_xticks(max_rounds_list)
            ax1.set_xticklabels([str(x) for x in max_rounds_list])
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 创建第二个y轴用于token数量
            ax2 = ax1.twinx()
            
            # token数量折线图
            ax2.plot(max_rounds_list, sim_tokens, 'r--s', label='Total Tokens')
            ax2.set_ylabel('Total Tokens', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # 添加图例
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
            
            plt.title(f'Verification Sampling Simulation (Difficulty Bucket {i+1})')
            plt.tight_layout()
            plt.savefig(os.path.join(figs_diff_dir, f'verification_diff{i+1}.png'), dpi=150)
            plt.close(fig)
    
    # 10. 汇总结果
    record_str = "===== Evaluation Results =====\n"
    record_str += f"Overall Accuracy: {overall_metrics['accuracy']:.4f}\n"
    record_str += f"Overall Verification Accuracy: {overall_metrics['verification_accuracy']:.4f}\n"
    record_str += f"Correct is Correct: {overall_metrics['correct_is_correct']:.4f}\n"
    record_str += f"Wrong is Wrong: {overall_metrics['wrong_is_wrong']:.4f}\n\n"
    
    for i, metrics in enumerate(bucket_metrics):
        record_str += f"----- Difficulty Bucket {i+1} -----\n"
        record_str += f"Avg Difficulty: {metrics['difficulty']:.4f}\n"
        record_str += f"Accuracy: {metrics['accuracy']:.4f}\n"
        record_str += f"Verification Accuracy: {metrics['verification_accuracy']:.4f}\n"
        record_str += f"Correct is Correct: {metrics['correct_is_correct']:.4f}\n"
        record_str += f"Wrong is Wrong: {metrics['wrong_is_wrong']:.4f}\n\n"
    
    print(record_str)
    
    # 保存结果
    results = {
        'overall': overall_metrics,
        'buckets': bucket_metrics,
        'record_str': record_str
    }
    
    with open(os.path.join(metrics_dir, f'verify_{name}.json'), 'w') as f:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        save_results = dict(
            config=cfg_dict,
            metrics=results
        )
        json.dump(save_results, f, indent=2)
    
    with open(os.path.join(metrics_dir, f'record_{name}.log'), 'w') as f:
        f.write(record_str)
    
    return results
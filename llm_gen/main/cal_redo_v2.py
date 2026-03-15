#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datasets import load_from_disk
from statistics import mean
from tqdm import tqdm
from tools.math_eval import math_lst_eval
import random

# tokenizer 用于长度统计（原功能保留）
from transformers import AutoTokenizer

REDO_STR = "s better to redo the question"

random.seed(42)


def is_correct_by_math_eval(pred: str | None, gt: str) -> float:
    """
    返回 math_lst_eval 打分（1.0 表示完全正确）
    """
    pred_str = pred if pred is not None else ""
    try:
        score = math_lst_eval([str(gt)], [pred_str])[0]
    except Exception as e:
        print(f"[警告] math_lst_eval 出错，pred={pred_str!r}, gt={gt!r}: {e}")
        score = 0.0
    return score


def needs_redo(resp: str) -> bool:
    """判断是否需要 redo"""
    if resp is None:
        return True
    tail = resp[-100:].lower()
    # 按你给的版本：包含 "better to redo"（不限定在尾部，区分大小写由你代码决定），或最后100字无 "boxed"
    return ("better to redo" in resp) or ("boxed" not in tail)


def power_of_two_sizes(n: int):
    """返回 1,2,4,...,<=n 的列表，确保包含 n"""
    sizes = []
    s = 1
    while s < n:
        sizes.append(s)
        s *= 2
    if not sizes or sizes[-1] != n:
        sizes.append(n)
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="load_from_disk 数据集路径")
    parser.add_argument("--num", type=int, default=1024, help="每题最多使用多少条 response（会在洗牌后截断）")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="\pah\to\your\model",
        help="用于统计长度的 tokenizer 路径"
    )
    parser.add_argument(
        "--sample_times",
        type=int,
        default=10,
        help="按每个采样规模重复采样的次数（例如 10）"
    )
    args = parser.parse_args()

    ds = load_from_disk(args.path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    acc_redo = []
    acc_naive = []
    acc_choose_no_redo = []
    acc_choose_redo = []

    # 长度统计容器（原功能保留）
    all_token_lengths = []
    required_avg_lengths = []
    avg_rounds = []

    redo_num = 0
    all_num = 0

    # 为“按采样规模多次采样”准备：每题的全量分数与 redo 标记
    per_item_scores_all = []     # List[List[float]]
    per_item_redo_all = []       # List[List[bool]]
    per_item_N = []              # 每题实际使用的响应数（洗牌+截断后）

    for idx, ex in enumerate(tqdm(ds, desc="Evaluating dataset (full)")):
        gt = str(ex["gt_answer"])
        responses = list(ex["responses"])  # 拷贝，避免原地修改

        # 洗牌 + 截断；对含 REDO_STR 的做截断处理（与原逻辑一致）
        rnd = random.Random(42 + idx)
        rnd.shuffle(responses)
        responses = responses[:args.num]
        responses = [(i if REDO_STR not in i else i.split(REDO_STR)[0] + REDO_STR) for i in responses]

        if not responses:
            acc_redo.append(0.0)
            acc_naive.append(0.0)
            per_item_scores_all.append([])
            per_item_redo_all.append([])
            per_item_N.append(0)
            continue

        # 对该题全体 responses 计算分数与 redo 标记（后续采样重用，避免重复评测）
        scores_all = [is_correct_by_math_eval(r, gt) for r in responses]
        redo_flags_all = [needs_redo(r) for r in responses]
        per_item_scores_all.append(scores_all)
        per_item_redo_all.append(redo_flags_all)
        per_item_N.append(len(responses))

        # ====== 原始全量统计（保留）======
        #长度统计（token 数）
        resp_token_lens = []
        for r in responses:
            text = r if r is not None else ""
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            L = len(ids)
            resp_token_lens.append(L)
            all_token_lengths.append(L)

        # redo 概率与所需平均轮数
        redo_flags = redo_flags_all
        if sum(redo_flags) == len(redo_flags):
            avg_round = len(redo_flags)
        else:
            avg_round = 1 / (1 - (sum(redo_flags) / len(redo_flags)))
        avg_rounds.append(avg_round)

        # “所需平均长度”
        if sum(redo_flags) == len(redo_flags):
            required_len = sum(resp_token_lens)
        else:
            required_len = avg_round * mean(resp_token_lens)
        required_avg_lengths.append(required_len)

        redo_num += sum(redo_flags)
        all_num += len(redo_flags)

        # 按 redo 规则的准确率
        no_redo_scores = [s for s, r in zip(scores_all, redo_flags_all) if not r]
        redo_scores = [s for s, r in zip(scores_all, redo_flags_all) if r]
        if no_redo_scores:
            acc_choose_no_redo.append(mean(no_redo_scores))
            acc_redo.append(mean(no_redo_scores))
        else:
            acc_redo.append(mean(scores_all))
        if redo_scores:
            acc_choose_redo.append(mean(redo_scores))

        # naive
        acc_naive.append(mean(scores_all))

    # ===== 全量输出（保留）=====
    print(f"平均准确率（按 redo 规则）：{mean(acc_redo):.6f}")
    print(f"平均准确率(naive):{mean(acc_naive):.6f}")
    print(f"平均准确率(no redo):{mean(acc_choose_no_redo):.6f}")
    print(f"平均准确率(redo):{mean(acc_choose_redo):.6f}")
    print(f"redo概率:", (redo_num / all_num) if all_num else 0.0)
    print(f"redo所需平均轮数: ", (sum(avg_rounds) / len(avg_rounds)) if avg_rounds else 0.0)
    if avg_rounds:
        valid_avg_rounds = [i for i in avg_rounds if i != max(avg_rounds)]
        print(
            f"有效redo比例，所需平均轮数: ",
            (len(valid_avg_rounds) / len(avg_rounds)) if avg_rounds else 0.0,
            ",",
            (sum(valid_avg_rounds) / len(valid_avg_rounds)) if valid_avg_rounds else 0.0,
        )
    else:
        print("有效redo比例，所需平均轮数: 0.0 , 0.0")

    # ===== 多次采样：按 1,2,4,8,...,N 输出四个指标 =====
    global_max_N = max(per_item_N) if per_item_N else 0
    sizes = power_of_two_sizes(global_max_N) if global_max_N > 0 else []

    if sizes:
        print("\n===== 按采样规模统计（多次采样）=====")
        print("规模\t平均准确率(redo)\tredu所需平均轮数\t有效redo比例\t有效redo所需平均轮数")
        # 使用固定随机源以保证复现；不同规模复用同一源也没问题
        rnd = random.Random(12345)

        for k in sizes:
            per_q_acc_means = []     # 每题在该规模下，sample_times 次准确率的均值
            per_q_round_means = []   # 每题在该规模下，sample_times 次轮数的均值

            for scores_all, redo_all, n_i in zip(per_item_scores_all, per_item_redo_all, per_item_N):
                if n_i == 0:
                    continue
                kk = min(k, n_i)

                acc_trials, round_trials = [], []
                for _ in range(args.sample_times):
                    # 不放回随机采样 kk 个索引
                    idxs = rnd.sample(range(n_i), kk)
                    s_sub = [scores_all[i] for i in idxs]
                    r_sub = [redo_all[i] for i in idxs]

                    # redo 规则准确率
                    no_redo_scores = [s for s, r in zip(s_sub, r_sub) if not r]
                    if no_redo_scores:
                        acc_k = mean(no_redo_scores)
                    else:
                        acc_k = mean(s_sub)
                    acc_trials.append(acc_k)

                    # redo 轮数
                    if sum(r_sub) == len(r_sub):
                        avg_round_k = len(r_sub)
                    else:
                        avg_round_k = 1 / (1 - (sum(r_sub) / len(r_sub)))
                    round_trials.append(avg_round_k)

                # 对该题取 sample_times 的均值
                per_q_acc_means.append(mean(acc_trials))
                per_q_round_means.append(mean(round_trials))

            # 聚合到数据集层面
            avg_acc_k = mean(per_q_acc_means) if per_q_acc_means else 0.0
            avg_rounds_k = mean(per_q_round_means) if per_q_round_means else 0.0

            # 有效 redo（去掉该规模下的最大轮数作为异常）
            if per_q_round_means:
                max_round_k = max(per_q_round_means)
                valid_rounds_k = [x for x in per_q_round_means if x != max_round_k]
                effective_ratio_k = (len(valid_rounds_k) / len(per_q_round_means)) if per_q_round_means else 0.0
                effective_mean_rounds_k = mean(valid_rounds_k) if valid_rounds_k else 0.0
            else:
                effective_ratio_k = 0.0
                effective_mean_rounds_k = 0.0

            print(
                f"{k}\t{avg_acc_k:.6f}\t\t{avg_rounds_k:.6f}\t\t{effective_ratio_k:.6f}\t{effective_mean_rounds_k:.6f}"
            )
    else:
        print("\n[提示] 数据集中没有可用的 responses，无法进行按采样规模统计。")


if __name__ == "__main__":
    main()

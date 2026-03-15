#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from collections import Counter
from statistics import mean
from typing import List, Optional, Tuple

from lighteval.utils.timeout import timeout
from datasets import load_from_disk
from tqdm import tqdm

from tools.math_eval import math_lst_eval

random.seed(42)

@timeout(5)
def math_lst_eval_with_timeout(golds, predictions):
    scores = math_lst_eval(golds, predictions)
    # for score, pred in zip(scores, predictions):
    #     if score == 0:
    #         print(golds, pred[-50:])
    return scores 

# --------- 评测函数（你指定的用法） ---------
def is_correct_by_math_eval(pred: Optional[str], gt: str) -> float:
    pred_str = pred if pred is not None else ""
    try:
        score = math_lst_eval_with_timeout([str(gt)], [pred_str])[0]
    except Exception as e:
        print(f"[警告] math_lst_eval_with_timeout 出错，pred={pred_str!r}, gt={gt!r}: {e}")
        score = 0.0
    return score


def extract_last_boxed(text: str) -> str | None:
    """
    从整段文本中按大括号配对提取“最后一个” \\boxed{...} 的内容。
    支持嵌套花括号（例如 \\boxed{\\frac{a}{b}}）。
    若找不到或不闭合，则返回 None。
    """
    s = text or ""
    target = r"\boxed{"
    i = 0
    last_content = None

    while True:
        start = s.find(target, i)
        if start == -1:
            break
        j = start + len(target)
        depth = 1
        # 逐字符扫描，配对花括号
        while j < len(s) and depth > 0:
            ch = s[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth == 0:
            # 匹配到一段完整的 \boxed{ ... } ，记录下来并继续向后找，取“最后一个”
            last_content = s[start + len(target) : j - 1]
            i = j
        else:
            # 未闭合，终止
            break

    return last_content


def normalize_for_vote(s: str) -> str:
    return s.replace(" ", "").strip()


def choose_mode(candidates_norm: List[str]) -> str:
    if not candidates_norm:
        raise ValueError("No candidates to choose from.")
    ctr = Counter(candidates_norm)
    max_freq = ctr.most_common(1)[0][1]
    tied = [s for s, c in ctr.items() if c == max_freq]
    tied.sort(key=lambda x: (len(x), x))
    return tied[0]


# --------- 纯查缓存的子集评估（不再调用 math_lst_eval_with_timeout） ---------
def evaluate_subset_cached(
    idxs: List[int],
    boxed_norm_list: List[Optional[str]],
    indiv_scores: List[float],
    cand_score_map: dict[str, float],
) -> Tuple[float, float, str, bool]:
    """
    返回:
      final_score: 众数投票得到的最终预测 与 gt 的得分（来自缓存）
      avg_individual_score: 子集内逐 response 的个体预测平均得分（来自缓存）
      final_pred_for_eval: 用于评测的最终预测（格式为 \boxed{...} 或 ""）
      had_voted_answer: 子集是否存在至少一个 boxed 候选
    """
    if not idxs:
        # 空子集的处理放到外层（需要 gt）；这里返回占位
        return 0.0, 0.0, "", False

    # 个体平均分
    avg_individual_score = mean(indiv_scores[i] for i in idxs)

    # 收集有 boxed 的候选用于投票
    candidates = [boxed_norm_list[i] for i in idxs if boxed_norm_list[i] is not None]
    if candidates:
        voted_norm = choose_mode(candidates)
        final_pred_for_eval = f"\\boxed{{{voted_norm}}}"
        had_voted = True
        # 众数对应的分数直接查表（若万一缺失，回退 0.0）
        final_score = cand_score_map.get(voted_norm, 0.0)
    else:
        final_pred_for_eval = ""
        had_voted = False
        # 无 boxed：按原逻辑，用个体平均分作为最终分
        final_score = avg_individual_score

    return final_score, avg_individual_score, final_pred_for_eval, had_voted


# --------- 规模生成 ---------
def power_of_two_sizes(n: int) -> List[int]:
    if n <= 0:
        return []
    sizes = []
    s = 1
    while s < n:
        sizes.append(s)
        s *= 2
    if not sizes or sizes[-1] != n:
        sizes.append(n)
    return sizes


# --------- 主流程 ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="load_from_disk 数据集路径")
    parser.add_argument("--num", type=int, default=10240, help="每个样本最多使用多少条 response（先洗牌再截断）")
    parser.add_argument("--sample_times", type=int, default=100, help="每个采样规模的重复采样次数")
    parser.add_argument("--verbose", action="store_true", help="打印每条样本的详细信息")
    args = parser.parse_args()

    ds = load_from_disk(args.path)

    # ---- 全局缓存容器 ----
    per_item_responses: List[List[str]] = []
    per_item_gt: List[str] = []
    per_item_N: List[int] = []

    # 缓存（关键）
    per_item_boxed_norm: List[List[Optional[str]]] = []   # 每题每条resp的 boxed 去空格(用于投票) 或 None
    per_item_pred_for_score: List[List[str]] = []         # 每题每条resp用于个体评分的预测串
    per_item_indiv_scores: List[List[float]] = []         # 每题每条resp的个体分缓存
    per_item_cand_score_map: List[dict] = []              # 每题 {voted_norm -> 分数} 缓存

    # 汇总指标（全量一次）
    overall_scores_full: List[float] = []
    overall_avg_individual_scores_full: List[float] = []

    for i in tqdm(range(len(ds)), desc="全量(≤num)一次评估（构建缓存 + 众数投票）"):
        item = ds[i]
        # gt 兼容 metadata.answer / gt_answer
        gt = None
        try:
            md = item.get("metadata", {})
            if isinstance(md, dict) and "answer" in md:
                gt = md["answer"]
        except Exception:
            gt = None
        if gt is None:
            gt = item.get("gt_answer", "") or ""

        responses = list(item.get("responses", []) or [])
        rnd = random.Random(42 + i)  # 题内可复现
        rnd.shuffle(responses)
        if args.num > 0:
            responses = responses[:args.num]

        per_item_responses.append(responses)
        per_item_gt.append(gt)
        per_item_N.append(len(responses))

        if not responses:
            # 与原逻辑一致：无预测→记空串评测
            score = is_correct_by_math_eval("", gt)
            overall_scores_full.append(score)
            overall_avg_individual_scores_full.append(0.0)

            # 空题也要填充占位缓存，方便后续流程
            per_item_boxed_norm.append([])
            per_item_pred_for_score.append([])
            per_item_indiv_scores.append([])
            per_item_cand_score_map.append({})
            if args.verbose:
                print(f"\n[id={i}] 无 responses，final_score={score:.3f}")
            continue

        # ---- 为该题构建缓存 ----
        boxed_norm_list: List[Optional[str]] = []
        pred_for_score_list: List[str] = []

        for r in responses:
            boxed = extract_last_boxed(r)
            if boxed is not None:
                pred_for_score_list.append(f"\\boxed{{{boxed}}}")
                boxed_norm_list.append(normalize_for_vote(boxed))
            else:
                pred_for_score_list.append(r)
                boxed_norm_list.append(None)

        per_item_boxed_norm.append(boxed_norm_list)
        per_item_pred_for_score.append(pred_for_score_list)

        # 批量个体评分
        gt_list = [str(gt)]
        try:
            indiv_scores = math_lst_eval_with_timeout(gt_list, pred_for_score_list)
        except Exception as e:
            print(f"[警告] 批量个体评分出错(id={i}): {e}")
            indiv_scores = [0.0] * len(pred_for_score_list)
        per_item_indiv_scores.append(indiv_scores)

        # 为所有“可能成为众数”的候选（去重后的 boxed_norm）批量评分并缓存
        unique_norm_candidates = sorted({c for c in boxed_norm_list if c is not None})
        preds_for_vote = [f"\\boxed{{{c}}}" for c in unique_norm_candidates]
        gt_list_vote = [str(gt)]
        cand_score_map = {}
        if preds_for_vote:
            try:
                scores_vote = math_lst_eval_with_timeout(gt_list_vote, preds_for_vote)
                cand_score_map = {c: s for c, s in zip(unique_norm_candidates, scores_vote)}
            except Exception as e:
                print(f"[警告] 候选众数批量评分出错(id={i}): {e}")
                cand_score_map = {c: 0.0 for c in unique_norm_candidates}
        per_item_cand_score_map.append(cand_score_map)

        # 使用缓存版评估全量 idxs
        idxs_all = list(range(len(responses)))

        final_score, avg_ind, final_pred, had_voted = evaluate_subset_cached(
            idxs_all,
            per_item_boxed_norm[-1],
            per_item_indiv_scores[-1],
            per_item_cand_score_map[-1],
        )

        overall_scores_full.append(final_score)
        overall_avg_individual_scores_full.append(avg_ind)

        if args.verbose:
            print(f"\n[id={i}]")
            print(f"GT: {gt!r}")
            print(f"responses used: {len(responses)}")
            print(f"had_voted_answer: {had_voted}")
            print(f"final_pred_for_eval: {final_pred!r}")
            print(f"final_score: {final_score:.3f}")
            print(f"avg_individual_score: {avg_ind:.3f}")

    # 汇总（全量一次）
    overall_mean_full = mean(overall_scores_full) if overall_scores_full else 0.0
    overall_mean_indiv_full = mean(overall_avg_individual_scores_full) if overall_avg_individual_scores_full else 0.0

    print("\n==== 全量(≤num) 一次评估（众数投票，已缓存） ====")
    print(f"样本数: {len(overall_scores_full)}")
    print(f"最终众数答案的平均得分: {overall_mean_full:.4f}")
    print(f"个体平均正确率(参考): {overall_mean_indiv_full:.4f}")

    # 多规模 + 多次采样（纯查缓存）
    global_max_N = max(per_item_N) if per_item_N else 0
    sizes = power_of_two_sizes(global_max_N)
    if not sizes:
        print("\n[提示] 无可用 responses，跳过多规模评估。")
        return

    print("\n==== 多规模 + 多次采样（众数投票，纯查缓存） ====")
    print("规模\t众数答案平均得分\t个体平均正确率(参考)")
    rnd_global = random.Random(12345)

    for k in sizes:
        per_q_final_means: List[float] = []
        per_q_indiv_means: List[float] = []

        for i in range(len(ds)):
            n_i = per_item_N[i]
            if n_i == 0:
                continue

            kk = min(k, n_i)
            trial_final_scores: List[float] = []
            trial_indiv_scores: List[float] = []

            for _ in range(args.sample_times):
                idxs = rnd_global.sample(range(n_i), kk)  # 无放回采样 kk 条
                final_score, avg_ind, _, _ = evaluate_subset_cached(
                    idxs,
                    per_item_boxed_norm[i],
                    per_item_indiv_scores[i],
                    per_item_cand_score_map[i],
                )
                trial_final_scores.append(final_score)
                trial_indiv_scores.append(avg_ind)

            per_q_final_means.append(mean(trial_final_scores))
            per_q_indiv_means.append(mean(trial_indiv_scores))

        avg_final_k = mean(per_q_final_means) if per_q_final_means else 0.0
        avg_indiv_k = mean(per_q_indiv_means) if per_q_indiv_means else 0.0

        print(f"{k}\t{avg_final_k:.6f}\t\t{avg_indiv_k:.6f}")


if __name__ == "__main__":
    main()

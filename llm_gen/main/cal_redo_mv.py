#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import re
from statistics import mean
from typing import List, Optional, Tuple

from lighteval.utils.timeout import timeout
from datasets import load_from_disk
from tqdm import tqdm
from tools.math_eval import math_lst_eval

# ---------- 常量 ----------
REDO_STR = "better to redo the question"
random.seed(42)
@timeout(5)
def math_lst_eval_with_timeout(golds, predictions):
    return math_lst_eval(golds, predictions)


# ---------- 你指定的打分函数 ----------
def is_correct_by_math_eval(pred: Optional[str], gt: str) -> float:
    """
    返回 math_lst_eval_with_timeout 打分（1.0 表示完全正确）
    """
    pred_str = pred[-100:] if pred is not None else ""
    try:
        score = math_lst_eval_with_timeout([str(gt)], [pred_str])[0]
    except Exception as e:
        print(f"[警告] math_lst_eval_with_timeout 出错，pred={pred_str!r}, gt={gt!r}: {e}")
        score = 0.0
    return score

# ---------- redo 判定 ----------
def needs_redo(resp: Optional[str]) -> bool:
    """
    规则与之前一致：
    - 文本中包含 "better to redo" 视为 redo（不区分大小写请按需要自行修改）
    - 或者最后 100 个字符里不含 "boxed"
    """
    if resp is None:
        return True
    tail = resp[-100:].lower()
    return ("better to redo" in resp) or ("boxed" not in tail)

# ---------- 提取 \boxed{...} 与投票 ----------
def extract_last_boxed(text: str) -> str | None:
    """
    从整段文本中按大括号配对提取“最后一个” \boxed{...} 的内容。
    支持嵌套花括号（例如 \boxed{\frac{a}{b}}）。
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

def choose_mode(cands: List[str]) -> str:
    """
    在已归一化的候选中选众数；并列时：长度升序，再字典序升序
    """
    from collections import Counter
    if not cands:
        raise ValueError("No candidates to choose from.")
    ctr = Counter(cands)
    max_freq = max(ctr.values())
    tied = [s for s, c in ctr.items() if c == max_freq]
    tied.sort(key=lambda x: (len(x), x))
    return tied[0]

# ---------- 规模列表 ----------
def power_of_two_sizes(n: int) -> List[int]:
    """返回 1,2,4,...,<=n 的列表，确保包含 n"""
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

# ---------- 在子集上进行“MV 优先、失败再平均”的 redo 规则打分 ----------
def redo_accuracy_mv_first_on_subset(
    idxs: List[int],
    redo_flags_all: List[bool],
    boxed_norm_all: List[Optional[str]],
    scores_all: List[float],
    gt: str,
) -> float:
    """
    给定子集索引 idxs：
    1) 取其中 no-redo 子集；若非空 -> 用 no-redo 子集，否则用 redo 子集
    2) 在所选子集上：若存在 boxed 候选 -> 进行众数投票并对 \boxed{voted} 打分
       否则 -> 回退到对所选子集 scores 的均值
    """
    if not idxs:
        return is_correct_by_math_eval(REDO_STR, gt)

    no_redo = [j for j in idxs if not redo_flags_all[j]]
    chosen = no_redo

    # MV 优先
    cands = [boxed_norm_all[j] for j in chosen if boxed_norm_all[j] is not None]
    if not cands: cands = [boxed_norm_all[j] for j in idxs if boxed_norm_all[j] is not None]
    if cands:
        voted = choose_mode(cands)
        final_pred = f"\\boxed{{{voted}}}"
        return is_correct_by_math_eval(final_pred, gt)

    # 没有任何 boxed 候选，回退到平均
    # print("no boxed:", mean(scores_all[j] for j in idxs))
    return mean(scores_all[j] for j in idxs)

# ---------- redo 轮数指标 ----------
def redo_rounds_for_subset(r_sub: List[bool]) -> float:
    """
    所需平均轮数的口径与之前一致：
    - 若全是 redo（p=1），返回 len(r_sub)
    - 否则返回 1 / (1 - p)，其中 p = redo 比例
    """
    if not r_sub:
        return 0.0
    p = sum(r_sub) / len(r_sub)
    if p >= 1.0:
        return float(len(r_sub))
    return 1.0 / (1.0 - p)

# ============================================================
#                         主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="load_from_disk 数据集路径")
    parser.add_argument("--num", type=int, default=1024, help="每题最多使用多少条 response（先洗牌再截断）")
    parser.add_argument("--sample_times", type=int, default=500, help="多规模下每个规模的重复采样次数")
    parser.add_argument("--verbose", action="store_true", help="打印每条样本的详细信息")
    args = parser.parse_args()

    ds = load_from_disk(args.path)
    print(f"[信息] 数据集共 {len(ds)} 条，features: {ds.features}")

    # 逐题预处理并做“全量(≤num)一次评估”
    per_item_responses: List[List[str]] = []
    per_item_gt: List[str] = []
    per_item_N: List[int] = []
    per_item_scores_all: List[List[float]] = []
    per_item_redo_all: List[List[bool]] = []
    per_item_boxed_norm_all: List[List[Optional[str]]] = []

    # 汇总（全量一次性）的两个口径
    overall_acc_redo_mv_first_full: List[float] = []   # 按 redo 规则且 MV 优先
    overall_acc_naive_mean_full: List[float] = []      # naive（对全部 responses 平均）

    for i in tqdm(range(len(ds)), desc="预处理 & 全量(≤num)一次评估"):
        item = ds[i]
        # 兼容两种 gt 字段：metadata.answer 优先，其次 gt_answer
        gt = None
        try:
            if "metadata" in item and isinstance(item["metadata"], dict) and "answer" in item["metadata"]:
                gt = str(item["metadata"]["answer"])
        except Exception:
            gt = None
        if gt is None:
            gt = str(item.get("gt_answer", "")) or ""

        # responses 洗牌+截断
        responses = list(item.get("responses", []) or [])
        rnd = random.Random(42 + i)
        rnd.shuffle(responses)
        if args.num > 0:
            responses = responses[:args.num]

        per_item_responses.append(responses)
        per_item_gt.append(gt)
        per_item_N.append(len(responses))

        if not responses:
            # 空题：redo-MV 口径 → REDO_STR；naive → 0
            overall_acc_redo_mv_first_full.append(is_correct_by_math_eval(REDO_STR, gt))
            overall_acc_naive_mean_full.append(0.0)
            per_item_scores_all.append([])
            per_item_redo_all.append([])
            per_item_boxed_norm_all.append([])
            if args.verbose:
                print(f"[id={i}] 无 responses")
            continue

        # 为该题预计算 per-response：
        redo_flags_all = [needs_redo(r) for r in responses]
        boxed_norm_all = [extract_last_boxed(r) for r in responses]
        scores_all = [is_correct_by_math_eval(r, gt) for r in responses]

        per_item_scores_all.append(scores_all)
        per_item_redo_all.append(redo_flags_all)
        per_item_boxed_norm_all.append(boxed_norm_all)

        # 全量(≤num)一次评估：
        idxs_all = list(range(len(responses)))
        acc_redo_mv = redo_accuracy_mv_first_on_subset(
            idxs_all, redo_flags_all, boxed_norm_all, scores_all, gt
        )
        overall_acc_redo_mv_first_full.append(acc_redo_mv)
        overall_acc_naive_mean_full.append(mean(scores_all))

        if args.verbose:
            print(f"[id={i}] used={len(responses)}, redo_mv={acc_redo_mv:.3f}, naive={mean(scores_all):.3f}")

    # —— 全量(≤num)一次评估：汇总 ——
    print("\n==== 全量(≤num) 一次评估 ====")
    print(f"样本数: {len(per_item_responses)}")
    print(f"平均准确率（redo规则，MV优先）: {mean(overall_acc_redo_mv_first_full):.6f}")
    print(f"平均准确率（naive，均值）     : {mean(overall_acc_naive_mean_full):.6f}")

    # —— 多规模 + 多次采样（1,2,4,8,...,N） —— 
    global_max_N = max(per_item_N) if per_item_N else 0
    sizes = power_of_two_sizes(global_max_N)
    if not sizes:
        print("\n[提示] 无可用 responses，跳过多规模评估。")
        return

    print("\n==== 多规模 + 多次采样（redo规则：MV优先，不行再平均） ====")
    print("规模\t平均准确率(redo+MV)\tredo所需平均轮数\t有效redo比例\t有效redo所需平均轮数")

    rnd_global = random.Random(12345)

    for k in sizes:
        per_q_acc_means: List[float] = []     # 每题在规模 k 下，sample_times 次 redo+MV 的均值
        per_q_round_means: List[float] = []   # 每题在规模 k 下，sample_times 次 redo 轮数的均值

        for i in range(len(per_item_responses)):
            n_i = per_item_N[i]
            if n_i == 0:
                continue
            kk = min(k, n_i)

            redo_all = per_item_redo_all[i]
            boxed_norm_all = per_item_boxed_norm_all[i]
            scores_all = per_item_scores_all[i]
            gt = per_item_gt[i]

            acc_trials: List[float] = []
            round_trials: List[float] = []

            for _ in range(args.sample_times):
                idxs = rnd_global.sample(range(n_i), kk)

                # redo + MV
                acc_k = redo_accuracy_mv_first_on_subset(
                    idxs, redo_all, boxed_norm_all, scores_all, gt
                )
                acc_trials.append(acc_k)

                # redo 轮数（仅由 r_sub 决定，与 MV 无关）
                r_sub = [redo_all[j] for j in idxs]
                round_trials.append(redo_rounds_for_subset(r_sub))

            per_q_acc_means.append(mean(acc_trials))
            per_q_round_means.append(mean(round_trials))

        # 聚合到数据集层面
        avg_acc_k = mean(per_q_acc_means) if per_q_acc_means else 0.0
        avg_rounds_k = mean(per_q_round_means) if per_q_round_means else 0.0

        # 有效 redo（去掉该规模下 per-question 轮数均值的最大值）
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

if __name__ == "__main__":
    main()

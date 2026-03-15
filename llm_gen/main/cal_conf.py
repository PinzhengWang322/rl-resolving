#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import math
from statistics import mean, stdev
from typing import List, Optional, Tuple

from datasets import load_from_disk
from lighteval.utils.timeout import timeout

from tools.math_eval import math_lst_eval

random.seed(42)


@timeout(5)
def math_lst_eval_with_timeout(golds, predictions):
    return math_lst_eval(golds, predictions)


# --------- 与原脚本保持一致的辅助函数 ---------
def extract_last_boxed(text: str) -> Optional[str]:
    """
    从整段文本中按大括号配对提取“最后一个” \\boxed{...} 的内容。
    支持嵌套花括号。若找不到或不闭合，则返回 None。
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
        while j < len(s) and depth > 0:
            ch = s[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last_content = s[start + len(target) : j - 1]
            i = j
        else:
            break

    return last_content


def prepare_predictions_for_scoring(responses: List[str]) -> List[str]:
    """
    对每条 response，如果包含 \\boxed{...}，只拿最后一个 boxed 作为预测；
    否则用原始字符串。
    """
    preds = []
    for r in responses:
        boxed = extract_last_boxed(r)
        if boxed is not None:
            preds.append(f"\\boxed{{{boxed}}}")
        else:
            preds.append(r)
    return preds


def mean_confidence_interval(
    values: List[float],
    alpha: float = 0.05,
    clamp_01: bool = True,
) -> Tuple[float, float, float]:
    """
    对一组数值计算均值和 (1 - alpha) 置信区间（正态近似）。
    返回: (mean, lower, upper)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    m = mean(values)
    if n == 1:
        lower = upper = m
    else:
        s = stdev(values)
        z = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96  # 简单写死 95% 对应的 z
        margin = z * s / math.sqrt(n)
        lower = m - margin
        upper = m + margin

    if clamp_01:
        lower = max(0.0, min(1.0, lower))
        upper = max(0.0, min(1.0, upper))

    return m, lower, upper


# --------- 主流程：只算均值和置信区间，不做 voting ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="load_from_disk 数据集路径")
    parser.add_argument("--num", type=int, default=0,
                        help="每题最多使用多少条 response（0 表示用全部；先洗牌再截断）")
    parser.add_argument("--ci_alpha", type=float, default=0.05,
                        help="置信区间的 alpha（默认 0.05，即 95% CI）")
    parser.add_argument("--verbose", action="store_true", help="打印每个样本的统计信息")
    args = parser.parse_args()

    ds = load_from_disk(args.path)

    # 所有“按 response 粒度”的评分
    all_response_scores: List[float] = []

    # “按题目粒度”的平均分（每题把自己的多条 response 先平均，再在题目之间平均）
    per_question_avg_scores: List[float] = []

    for i in range(len(ds)):
        item = ds[i]

        # 取 GT（兼容 metadata.answer / gt_answer）
        gt = None
        try:
            md = item.get("metadata", {})
            if isinstance(md, dict) and "answer" in md:
                gt = str(md["answer"])
        except Exception:
            gt = None
        if gt is None:
            gt = str(item.get("gt_answer", "")) or ""

        # responses
        responses = list(item.get("responses", []) or [])
        rnd = random.Random(42 + i)  # 题内可复现
        rnd.shuffle(responses)
        if args.num > 0:
            responses = responses[: args.num]

        if not responses:
            # 没有预测的话，默认用空串评测（与原脚本保持一致感觉）
            try:
                score_empty = math_lst_eval_with_timeout([gt], [""])[0]
            except Exception as e:
                print(f"[警告] 空预测评分失败(id={i}): {e}")
                score_empty = 0.0
            per_question_avg_scores.append(score_empty)
            if args.verbose:
                print(f"\n[id={i}] 无 responses，per_question_avg_score={score_empty:.4f}")
            continue

        preds = prepare_predictions_for_scoring(responses)

        # 这里沿用你原来的使用方式：gt_list 长度为 1，math_lst_eval 内部广播
        try:
            indiv_scores = math_lst_eval_with_timeout([gt], preds)
        except Exception as e:
            print(f"[警告] 批量评分出错(id={i}): {e}")
            indiv_scores = [0.0] * len(preds)

        # 累积到全局
        all_response_scores.extend(indiv_scores)

        # 按题目取平均
        avg_score_this_q = mean(indiv_scores) if indiv_scores else 0.0
        per_question_avg_scores.append(avg_score_this_q)

        if args.verbose:
            print(f"\n[id={i}]")
            print(f"GT: {gt!r}")
            print(f"responses used: {len(responses)}")
            print(f"per_question_avg_score: {avg_score_this_q:.4f}")

    # --------- 统计 + 置信区间 ---------
    print("\n==== 评测结果（不含 majority voting）====")

    # 1) 按 response 粒度
    n_resp = len(all_response_scores)
    m_resp, lo_resp, hi_resp = mean_confidence_interval(
        all_response_scores, alpha=args.ci_alpha, clamp_01=True
    )
    print(f"\n[按 response 粒度]")
    print(f"  总 response 数: {n_resp}")
    print(f"  平均得分: {m_resp:.6f}")
    print(
        f"  {(1 - args.ci_alpha) * 100:.1f}% 置信区间: "
        f"+-{(hi_resp - m_resp):.6f}"
    )

    # 2) 按题目粒度（先对每题求平均，再在题目之间做均值和 CI）
    n_q = len(per_question_avg_scores)
    m_q, lo_q, hi_q = mean_confidence_interval(
        per_question_avg_scores, alpha=args.ci_alpha, clamp_01=True
    )
    # print(f"\n[按题目粒度（每题平均后再统计）]")
    # print(f"  题目数: {n_q}")
    # print(f"  平均得分: {m_q:.6f}")
    # print(
    #     f"  {(1 - args.ci_alpha) * 100:.1f}% 置信区间: "
    #     f"+-{(hi_q - m_q):.6f}"
    # )


if __name__ == "__main__":
    main()

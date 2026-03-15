# import argparse
# from datasets import load_from_disk
# from statistics import mean
# from tqdm import tqdm
# from tools.math_eval import math_lst_eval_with_timeout

# REDO_STR="better to redo the question"
# def is_correct_by_math_eval(pred: str | None, gt: str) -> float:
#     """
#     返回 math_lst_eval_with_timeout 打分（1.0 表示完全正确）
#     """
#     pred_str = pred if pred is not None else ""
#     try:
#         score = math_lst_eval_with_timeout([gt], [pred_str])[0]
#     except Exception as e:
#         print(f"[警告] math_lst_eval_with_timeout 出错，pred={pred_str!r}, gt={gt!r}: {e}")
#         score = 0.0
#     return score


# def needs_redo(resp: str) -> bool:
#     """判断是否需要 redo"""
#     if resp is None:
#         return True
#     tail = resp[-100:].lower()
#     # if ("better to redo" in resp) or ("boxed" not in tail):
#     #     print(("better to redo" in resp), ("boxed" not in tail))
#     return ("better to redo" in resp) 


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path", type=str, help="load_from_disk 数据集路径")
#     args = parser.parse_args()

#     ds = load_from_disk(args.path)

#     acc_redo = []
#     acc_naive = []
#     acc_choose_no_redo = []
#     acc_choose_redo = []

#     # 按题处理
#     redo_num = 0
#     all_num = 0
#     avg_rounds = []
#     for ex in tqdm(ds, desc="Evaluating dataset"):
#         gt = ex["gt_answer"]
#         responses = ex["responses"]
#         responses = [(i if REDO_STR not in i else i.split(REDO_STR)[0] + REDO_STR) for i in responses]

#         if not responses:
#             acc_redo.append(0.0)
#             acc_naive.append(0.0)
#             continue

#         scores = [is_correct_by_math_eval(r, gt) for r in responses]
#         redo_flags = [needs_redo(r) for r in responses]

#         # for redo_flag, resp in zip(redo_flags, responses):
#         #     if redo_flag:
#         #         print(resp[-100:]) 
#         if sum(redo_flags) == len(redo_flags):
#             avg_round = len(redo_flags)
#         else:
#             avg_round = 1 / (1 - (sum(redo_flags) / len(redo_flags)))
#         avg_rounds.append(avg_round)
#         redo_num += sum(redo_flags)
#         all_num += len(responses)

#         # 按 redo 规则
#         no_redo_scores = [s for s, r in zip(scores, redo_flags) if not r]
#         redo_scores = [s for s, r in zip(scores, redo_flags) if r]
#         if no_redo_scores:
#             acc_choose_no_redo.append(mean(no_redo_scores))
#         if redo_scores:
#             acc_choose_redo.append(mean(redo_scores))
#         if no_redo_scores:
#             acc_redo.append(mean(no_redo_scores))
#         else:
#             acc_redo.append(mean(scores))

#         # naive
#         acc_naive.append(mean(scores))

#     print(f"平均准确率（按 redo 规则）：{mean(acc_redo):.6f}")
#     print(f"平均准确率(naive):{mean(acc_naive):.6f}")
#     print(f"平均准确率(no redo):{mean(acc_choose_no_redo):.6f}")
#     print(f"平均准确率(redo):{mean(acc_choose_redo):.6f}")
#     print(f"redo概率:", redo_num / all_num)
#     print(f"redo所需平均轮数: ", sum(avg_rounds) / len(avg_rounds))
#     valid_avg_rounds = [i for i in avg_rounds if i != max(avg_rounds)]
#     print(f"有效redo比例，所需平均轮数: ", len(valid_avg_rounds) / len(avg_rounds), ",", sum(valid_avg_rounds) / len(valid_avg_rounds))
#     # print(avg_rounds)


# if __name__ == "__main__":
#     main()

import argparse
from datasets import load_from_disk
from statistics import mean
from tqdm import tqdm
from tools.math_eval import math_lst_eval
import random

# 新增：transformers tokenizer
from transformers import AutoTokenizer

REDO_STR = "better to redo the question"

random.seed(42)
@timeout(5)
def math_lst_eval_with_timeout(golds, predictions):
    return math_lst_eval(golds, predictions)



def is_correct_by_math_eval(pred: str | None, gt: str) -> float:
    """
    返回 math_lst_eval_with_timeout 打分（1.0 表示完全正确）
    """
    pred_str = pred if pred is not None else ""
    try:
        score = math_lst_eval_with_timeout([gt], [pred_str])[0]
    except Exception as e:
        print(f"[警告] math_lst_eval_with_timeout 出错，pred={pred_str!r}, gt={gt!r}: {e}")
        score = 0.0
    return score


def needs_redo(resp: str) -> bool:
    """判断是否需要 redo"""
    if resp is None:
        return True
    tail = resp[-100:].lower()
    return ("better to redo" in resp) or ('boxed' not in tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="load_from_disk 数据集路径")
    parser.add_argument("--num", type=int, default=1024)
    # 新增：tokenizer 路径（默认按你的要求）
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="\path\to\your\model",
        help="用于统计长度的 tokenizer 路径"
    )
    args = parser.parse_args()

    ds = load_from_disk(args.path)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    acc_redo = []
    acc_naive = []
    acc_choose_no_redo = []
    acc_choose_redo = []

    # 用于长度统计的容器
    all_token_lengths = []          # 所有单条回复的长度
    required_avg_lengths = []       # 按题的“所需平均长度”（类比所需平均轮数）
    avg_rounds = []                 # 保留你原始的所需平均轮数

    redo_num = 0
    all_num = 0

    for ex in tqdm(ds, desc="Evaluating dataset"):
        gt = ex["gt_answer"]
        responses = ex["responses"]
        random.shuffle(responses)
        responses = responses[:args.num]
        responses = [(i if REDO_STR not in i else i.split(REDO_STR)[0] + REDO_STR) for i in responses]

        if not responses:
            acc_redo.append(0.0)
            acc_naive.append(0.0)
            continue

        # 计算正确率 & redo 标记
        scores = [is_correct_by_math_eval(r, gt) for r in responses]
        redo_flags = [needs_redo(r) for r in responses]

        # 统计长度（token 数）
        # 注意：不加 special tokens，仅按文本本身长度
        resp_token_lens = []
        for r in responses:
            text = r if r is not None else ""
            # 对 None 的回复做空串处理
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            L = len(ids)
            resp_token_lens.append(L)
            all_token_lengths.append(L)

        # 期望轮数（与你原代码一致）
        if sum(redo_flags) == len(redo_flags):
            avg_round = len(redo_flags)
        else:
            avg_round = 1 / (1 - (sum(redo_flags) / len(redo_flags)))
        avg_rounds.append(avg_round)

        # “所需平均长度”的对应计算：
        # - 若全是 redo（p=1），用该题所有回复长度之和作为上界
        # - 否则用 期望轮数 × 该题单轮平均长度
        if sum(redo_flags) == len(redo_flags):
            required_len = sum(resp_token_lens)
        else:
            required_len = avg_round * mean(resp_token_lens)
        required_avg_lengths.append(required_len)

        # 汇总 redo 概率分母
        redo_num += sum(redo_flags)
        all_num += len(responses)

        # 按 redo 规则的准确率（保留原逻辑）
        no_redo_scores = [s for s, r in zip(scores, redo_flags) if not r]
        redo_scores = [s for s, r in zip(scores, redo_flags) if r]
        if no_redo_scores:
            acc_choose_no_redo.append(mean(no_redo_scores))
        if redo_scores:
            acc_choose_redo.append(mean(redo_scores))
        if no_redo_scores:
            acc_redo.append(mean(no_redo_scores))
        else:
            acc_redo.append(mean(scores))

        # naive
        acc_naive.append(mean(scores))

    # 原有打印
    print(f"平均准确率（按 redo 规则）：{mean(acc_redo):.6f}")
    print(f"平均准确率(naive):{mean(acc_naive):.6f}")
    print(f"平均准确率(no redo):{mean(acc_choose_no_redo):.6f}")
    print(f"平均准确率(redo):{mean(acc_choose_redo):.6f}")
    print(f"redo概率:", redo_num / all_num)
    print(f"redo所需平均轮数: ", sum(avg_rounds) / len(avg_rounds))
    valid_avg_rounds = [i for i in avg_rounds if i != max(avg_rounds)]
    print(f"有效redo比例，所需平均轮数: ", len(valid_avg_rounds) / len(avg_rounds), ",", sum(valid_avg_rounds) / len(valid_avg_rounds))

    # ===== 新增三项：平均长度 & 有效 redo 的长度版指标 =====
    # 1) 回复平均长度（所有单条回复的 token 平均数）
    avg_reply_tokens = mean(all_token_lengths) if all_token_lengths else 0.0
    print(f"回复平均长度(词元): {avg_reply_tokens:.2f}")

    # 2) 有效 redo 比例（在长度视角下，和轮数视角一致：排除 required_avg_lengths 的最大值）
    if required_avg_lengths:
        max_required_len = max(required_avg_lengths)
        valid_required_lengths = [x for x in required_avg_lengths if x != max_required_len]
        effective_ratio = len(valid_required_lengths) / len(required_avg_lengths) if required_avg_lengths else 0.0
        print(f"有效redo比例(长度版): {effective_ratio:.6f}")

        # 3) 有效 redo 所需平均长度（去掉异常最大值后的平均“所需长度”）
        if valid_required_lengths:
            print(f"有效redo所需平均长度(词元): {mean(valid_required_lengths):.2f}")
        else:
            print("有效redo所需平均长度(词元): 0.00")
    else:
        print(f"有效redo比例(长度版): 0.000000")
        print(f"有效redo所需平均长度(词元): 0.00")


if __name__ == "__main__":
    main()

from lighteval.metrics.utils.extractive_match_utils import (  # noqa: F401
    ExprExtractionConfig,
    ExtractionTarget,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.metrics.utils.math_comparison import compare_gold_target
from collections import Counter, defaultdict
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import os
from transformers import AutoTokenizer
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
from lighteval.utils.timeout import timeout

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import concurrent.futures
import signal
from contextlib import contextmanager



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



def math_lst_eval(golds: list[str], predictions: list[str]) -> list[float]:
   language = Language.ENGLISH
   gold_extraction_target=(LatexExtractionConfig(),)
   fallback_mode="first_match"
   extraction_mode="any_match"
   timeout_seconds = 5
   precision=5
   aggregation_function = max
   formatted_doc = Doc(
       task_name="",
       query="",
       choices="",
       gold_index=0,
       instruction="",
   ) # dummy doc
   
   golds = ['\\boxed{' + gold + '}' if '\\boxed' not in gold else gold for gold in golds]
   pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0))
   gold_extraction_regexes = get_extraction_regexes(formatted_doc, gold_extraction_target, language)
   pred_extraction_regexes = get_extraction_regexes(formatted_doc, pred_extraction_target, language)
   
   extracted_predictions = [
       extract_target_from_pred_with_timeout(pred, pred_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds)
       for pred in predictions
   ]
   extracted_golds = [
       extract_target_from_pred_with_timeout(gold, gold_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds)
       for gold in golds
   ]
   
   # Assert on empty gold and warn on empty pred
   if any(len(g) == 0 for g in extracted_golds):
       print(f"We did not manage to extract a gold in the correct format. Gold: {golds}")
       extracted_golds = [[gold] for gold in golds]
    
   # if all(len(p) == 0 for p in extracted_predictions):
   #     print(
   #         f"We did not manage to extract a prediction in the correct format. Gold: {golds}, Pred: {predictions}"
   #     )
   
   def add_to_specifics_with_timeout(
       formatted_doc: Doc, extracted_predictions: list[list[str]], extracted_golds: list[list[str]]
   ) -> None:
       if formatted_doc.specific is None:
           formatted_doc.specific = {}

       formatted_doc.specific["extracted_predictions"] = [
           str(pred) for preds in extracted_predictions for pred in preds
       ]
       formatted_doc.specific["extracted_golds"] = [str(gold) for golds in extracted_golds for gold in golds]

   # We have to use timeout because the sypmy to str conversion can be very slow
   try:
       add_to_specifics_with_timeout(formatted_doc, extracted_predictions, extracted_golds)
   except Exception:  # noqa: E722
       print("Timeout when adding extracted predictions and golds to specific")
   return [
               (
                   1.0
                   if any(
                       compare_gold_target(gold, pred, precision, timeout_seconds=timeout_seconds)
                       for gold in extracted_golds
                   )
                   else 0.0
               )
               for pred in extracted_predictions
           ]


def evaluate_dataset(
    dataset,
    tokenizer=None,
    fig_name: str = "length_histogram.png",
    ban_str: str = None,
) -> Tuple[str, dict, Optional[str]]:
    """
    Evaluate a HuggingFace-style dataset with multiple sampled responses per item.
    如果传入 tokenizer，则额外统计 response 长度并绘制
    “响应长度直方图 + 区段平均正确率” 双 y 轴图。

    Args
    ----
    dataset : Iterable[dict]
        每条样例格式::
            {
                "responses": [str, ...],
                "gt_answer": str
            }
    tokenizer : transformers.PreTrainedTokenizerBase | None
        用来计算 token 数；为 None 时跳过长度分析。
    fig_name : str
        图像文件名（含扩展名）；会保存在当前工作目录。

    Returns
    -------
    record_str : str
        可打印的评测摘要
    results : dict
        指标结果
    fig_path : str | None
        生成的 png 路径；若 tokenizer 为 None 则为 None
    """
    correct_total     = 0
    majority_correct  = 0
    pass_count        = 0
    dist_correct      = defaultdict(int)
    all_correct_flags = []

    # 响应长度统计
    lengths_all   = []        # 与 flags 对齐的一维列表
    lengths_bins  = None      # 边界数组，稍后生成
    bin_stats     = None      # [count, correct_sum] per bin

    # 一些常量
    n_resp  = len(dataset[0]["responses"])
    n_items = len(dataset)

    record_lines = []

    for example in tqdm(dataset, desc="Evaluating"):
        gold        = example["gt_answer"]
        predictions = example["responses"]
        if ban_str is not None:
            test_predictions = [i if ban_str not in i else "[BANNED RESPONSE]" for i in example['responses']]
        else:
            test_predictions = predictions

        # if "[BANNED RESPONSE]" in test_predictions: print('BANNED IT !')
        # ---- correctness ----
        flags     = math_lst_eval([gold], test_predictions)  # 0/1 array
        k_correct = int(np.sum(flags))

        correct_total     += k_correct
        all_correct_flags += flags
        dist_correct[k_correct] += 1
        if k_correct > 0:
            pass_count += 1

        # majority vote
        majority_pred    = Counter(predictions).most_common(1)[0][0]
        majority_correct += math_lst_eval([gold], [majority_pred])[0]

        # ---- response length ----
        if tokenizer is not None:
            # ⬇️ add_special_tokens=False 避免 [CLS], [SEP] 等影响
            resp_lens = [
                len(tokenizer.encode(r, add_special_tokens=False)) for r in predictions
            ]
            lengths_all.extend(resp_lens)

            # 先记录长度，后续一次性生成 bins
            # flags 与 resp_lens 长度一致，可 later zip
    # ──────────────────────────────────────────────────────────────────────────
    # 基本指标
    total_preds       = n_items * n_resp
    accuracy          = correct_total / total_preds
    variance          = float(np.var(all_correct_flags, ddof=0))
    pass_rate         = pass_count / n_items
    majority_accuracy = majority_correct / n_items

    results = dict(
        accuracy=accuracy,
        variance=variance,
        pass_rate=pass_rate,
        majority_accuracy=majority_accuracy,
        num_correct_distribution=dict(sorted(dist_correct.items())),
    )

    # ───────── 评测摘要字符串 ─────────
    record_lines.append("\n===== Evaluation Results =====")
    record_lines.append(f"Number of samples         : {n_items}")
    record_lines.append(f"Responses per sample (n)  : {n_resp}")
    record_lines.append(f"Overall Accuracy          : {accuracy:.4%}")
    record_lines.append(f"Variance of Correctness   : {variance:.6f}")
    record_lines.append(f"Pass Rate (≥1 correct)    : {pass_rate:.4%}")
    record_lines.append(f"Majority Vote Accuracy    : {majority_accuracy:.4%}")

    record_lines.append("\n-- Distribution of Exactly k Correct Responses (0 – n) --")
    max_cnt   = max(dist_correct.values())
    bar_width = 50
    scale     = bar_width / max_cnt if max_cnt else 1
    for k in range(n_resp + 1):
        cnt = dist_correct.get(k, 0)
        bar = "█" * int(cnt * scale)
        record_lines.append(f"{k:>2} | {bar:<{bar_width}} {cnt}")

    # ==============  长度分析 + 绘图  ==============
    fig_path = None
    if tokenizer is not None and lengths_all:
        # -- 构造 10 个区段边界 --
        p10, p90 = np.percentile(lengths_all, [10, 90])
        width    = (p90 - p10) / 8 if p90 > p10 else 1  # 防止 0 除
        # 左无穷, 右无穷
        lengths_bins = np.concatenate(
            (
                [-np.inf],
                np.linspace(p10, p90, 9),  # 8 等分产生 9 个点
                [np.inf],
            )
        )  # 共 11 个边界 -> 10 个区段

        # 统计 count & correct
        bin_stats = np.zeros((10, 2), dtype=float)  # [:,0]=count, [:,1]=correct_sum

        # 重新遍历一次 dataset 来对齐 flags / lengths
        idx = 0
        for example in dataset:
            gold        = example["gt_answer"]
            predictions = example["responses"]
            flags       = math_lst_eval([gold], predictions)

            for resp, flag in zip(predictions, flags):
                length = len(tokenizer.encode(resp, add_special_tokens=False))
                # 找到所属区段
                bin_idx = np.searchsorted(lengths_bins, length, side="right") - 1
                bin_stats[bin_idx, 0] += 1
                bin_stats[bin_idx, 1] += flag
                idx += 1

        # 计算平均正确率
        avg_acc = np.divide(
            bin_stats[:, 1],
            np.maximum(bin_stats[:, 0], 1),  # 防 0 除
            where=bin_stats[:, 0] > 0,
        )

        # -- 绘图 --
        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = np.arange(10)

        # 柱状图：数量
        ax1.bar(x, bin_stats[:, 0], width=0.7, alpha=0.6)
        ax1.set_xlabel("Response length bins")
        ax1.set_ylabel("Count")
        ax1.set_xticks(x)
        # 自定义 x-tick 标签
        xtick_labels = []
        for i in range(10):
            if i == 0:
                xtick_labels.append(f"<{p10:.0f}")
            elif i == 9:
                xtick_labels.append(f">{p90:.0f}")
            else:
                start = p10 + (i - 1) * width
                end   = start + width
                xtick_labels.append(f"{start:.0f}-{end:.0f}")
        ax1.set_xticklabels(xtick_labels, rotation=30, ha="right")

        # 折线：平均正确率
        ax2 = ax1.twinx()
        ax2.plot(x, avg_acc, marker="o")
        ax2.set_ylabel("Average accuracy")

        ax1.set_title("Response Length Distribution & Accuracy per Bin")
        fig.tight_layout()

        # 保存
        fig.savefig(fig_name, dpi=150)
        plt.close(fig)

        fig_path = os.path.abspath(fig_name)
        results["length_bins"]         = xtick_labels
        results["length_bin_counts"]   = bin_stats[:, 0].astype(int).tolist()
        results["length_bin_accuracy"] = avg_acc.round(4).tolist()
        results["histogram_png"]       = fig_path

        record_lines.append("\n-- Length Analysis --")
        record_lines.append(f"Histogram saved to: {fig_path}")

    record_lines.append("=" * 28 + "\n")
    record_str = "\n".join(record_lines)

    # 打印 & 返回
    print(record_str)
    return record_str, results

def evaluate_dataset_with_difficulty(
    dataset,
    tokenizer=None,
    fig_dir: str = "./temp",
) -> Tuple[str, dict, Optional[List[str]]]:
    """
    评估 + 去掉 prop∈{0,1} 做均匀五档分桶 + 画 5 张长度-准确率双 y 图
    """
    n_resp  = len(dataset[0]["responses"])
    n_items = len(dataset)
    os.makedirs(fig_dir, exist_ok=True)

    per_item_correct, per_item_flags, per_item_lens = [], [], []
    all_flags, pass_cnt = [], 0

    # ---------- 单遍统计 ----------
    for ex in tqdm(dataset, desc="scoring"):
        flags = np.asarray(
            math_lst_eval([ex["gt_answer"]] * n_resp, ex["responses"]), dtype=int
        )
        per_item_flags.append(flags)
        per_item_correct.append(flags.sum())
        all_flags.extend(flags)
        pass_cnt += (flags.sum() > 0)
        if tokenizer:
            per_item_lens.append(
                [len(tokenizer.encode(r, add_special_tokens=False))
                 for r in ex["responses"]]
            )
        else:
            per_item_lens.append([])

    # ---------- 构造 bucket ----------
    prop        = np.array(per_item_correct, dtype=float) / n_resp
    sorted_idx  = np.argsort(prop)
    trimmed_idx = [i for i in sorted_idx if 0 < prop[i] < 1]
    base, extra = divmod(len(trimmed_idx), 5)
    bucket_of   = np.full(n_items, -1, int)

    s = 0
    for b in range(5):
        size = base + (1 if b < extra else 0)
        idxs = trimmed_idx[s:s + size]
        bucket_of[idxs] = b
        s += size
    bucket_of[prop == 0] = 0
    bucket_of[prop == 1] = 4

    # ---------- 按 bucket 聚合 ----------
    bucket_stats = [defaultdict(list) for _ in range(5)]
    for i in range(n_items):
        b = bucket_of[i]
        bucket_stats[b]['flags'].extend(per_item_flags[i])
        if tokenizer:
            bucket_stats[b]['lens'].extend(per_item_lens[i])

    results   = {"overall": {
        "accuracy" : float(np.mean(all_flags)),
        "variance" : float(np.var(all_flags, ddof=0)),
        "pass_rate": pass_cnt / n_items,
    }}
    fig_paths = [] if tokenizer else None
    rec = [
        "===== Eval (difficulty buckets) =====",
        f"samples: {n_items}  resp/ℹ️: {n_resp}",
        f"accuracy : {results['overall']['accuracy']:.4%}",
        f"variance : {results['overall']['variance']:.6f}",
        f"passRate : {results['overall']['pass_rate']:.4%}\n",
    ]

    for b in range(5):
        flags = np.array(bucket_stats[b]['flags'])
        acc   = flags.mean() if flags.size else 0.0
        results[f"bucket{b}"] = {
            "num_items": int((bucket_of == b).sum()),
            "accuracy" : float(acc),
        }
        rec.append(f"bucket{b}: items {results[f'bucket{b}']['num_items']:>4}  acc {acc:.4%}")

        if not tokenizer or not bucket_stats[b].get('lens'):
            continue

        # ---------- 绘图 ----------
        lens  = np.array(bucket_stats[b]['lens'])
        p10, p90 = np.percentile(lens, [10, 90])
        bins = np.concatenate(([-np.inf], np.linspace(p10, p90, 9), [np.inf]))
        cnt_acc = np.zeros((10, 2))
        for ln, fl in zip(lens, flags):
            idx = np.searchsorted(bins, ln, side="right") - 1
            cnt_acc[idx, 0] += 1
            cnt_acc[idx, 1] += fl
        avg = np.divide(cnt_acc[:, 1], np.maximum(cnt_acc[:, 0], 1))

        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = np.arange(10)
        ax1.bar(x, cnt_acc[:, 0], 0.7, alpha=0.6)
        ax2 = ax1.twinx(); ax2.plot(x, avg, marker='o')
        ax1.set_xticks(x)
        labels = [f"<{p10:.0f}"] + \
                 [f"{p10+(i-1)*(p90-p10)/8:.0f}-{p10+i*(p90-p10)/8:.0f}" for i in range(1,9)] + \
                 [f">{p90:.0f}"]
        ax1.set_xticklabels(labels, rotation=30, ha="right")
        ax1.set_xlabel("length"); ax1.set_ylabel("count"); ax2.set_ylabel("avg acc")
        ax1.set_title(f"bucket{b} length vs acc"); fig.tight_layout()

        path = os.path.join(fig_dir, f"bucket{b}.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        fig_paths.append(os.path.abspath(path))

        results[f"bucket{b}"].update(
            length_bins         = labels,
            length_bin_counts   = cnt_acc[:, 0].astype(int).tolist(),
            length_bin_accuracy = avg.round(4).tolist(),
            histogram_png       = path,
        )

def evaluate_dataset2(
    dataset,
    tokenizer=None,
    mv_powers: Optional[List[int]] = None,
) -> Tuple[str, dict]:
    """简化版评测（无绘图）并 **改用表达式抽取后的文本进行 MV**。

    主要变动
    ----------
    * 在 MV 计算（普通 Majority Vote 与 MV‑by‑Length）中，
      先调用 `extract_target_from_pred_with_timeout` 对每个回复做解析，
      再对返回值的 **第二个元素** 进行多数投票。
    * 其它逻辑与上一版保持一致，只调用一次 `math_lst_eval` 获取正确标志，
      并避免重复的表达式解析。
    """

    # ================= 预备：Regex 缓存 =================
    language = Language.ENGLISH
    dummy_doc = Doc(task_name="", query="", choices="", gold_index=0, instruction="")
    pred_extraction_target = (
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    )
    pred_extraction_regexes = get_extraction_regexes(dummy_doc, pred_extraction_target, language)
    fallback_mode = "first_match"
    extraction_mode = "any_match"
    timeout_seconds = 5

    # —— 基础统计变量 ——
    n_resp  = len(dataset[0]["responses"])
    n_items = len(dataset)

    # 自动生成 mv_powers
    if mv_powers is None:
        mv_powers = [1]
        while mv_powers[-1] * 2 <= n_resp:
            mv_powers.append(mv_powers[-1] * 2)
        if mv_powers[-1] != n_resp:
            mv_powers.append(n_resp)
    else:
        mv_powers = sorted({k for k in mv_powers if 1 <= k <= n_resp})
        if mv_powers[-1] != n_resp:
            mv_powers.append(n_resp)

    # —— 全局计数 ——
    correct_total     = 0
    majority_correct  = 0
    pass_count        = 0
    dist_correct      = defaultdict(int)
    all_correct_flags = []
    mv_len_correct    = {k: 0 for k in mv_powers}
    prefix_correct_ct = {k: 0 for k in mv_powers}  # 用于平均值

    record_lines = []

    # —— 主循环 ——
    for ex in tqdm(dataset, desc="Evaluating"):
        gold        = ex["gt_answer"]
        predictions = ex["responses"]

        # 1️⃣ 一次性计算所有 flags（耗时操作）
        flags = math_lst_eval([gold], predictions)  # len == n_resp, 0/1

        # —— 解析每个回复，得到用于投票的 processed_pred ——
        processed_preds = []  # 与 predictions、flags 对齐
        for pred in predictions:
            try:
                extracted = extract_target_from_pred_with_timeout(
                    pred,
                    pred_extraction_regexes,
                    fallback_mode,
                    extraction_mode,
                    timeout_seconds,
                )
                processed = extracted[1] if isinstance(extracted, (list, tuple)) and len(extracted) > 1 else str(extracted)
            except Exception:
                processed = ""
            processed_preds.append(str(processed))

        # —— 基本计数 ——
        k_correct = int(np.sum(flags))
        correct_total     += k_correct
        all_correct_flags += flags
        dist_correct[k_correct] += 1
        if k_correct > 0:
            pass_count += 1

        # 2️⃣ Majority‑Vote（全部回复）——基于 processed_preds
        maj_val = Counter(processed_preds).most_common(1)[0][0]
        # 在原列表中找到对应 index，使用其 flag 判断正确
        for idx, val in enumerate(processed_preds):
            if val == maj_val:
                majority_correct += flags[idx]
                break

        # 3️⃣ MV‑by‑Length（需要 tokenizer）——也同样使用 processed_preds
        if tokenizer is not None:
            resp_lens   = [len(tokenizer.encode(r, add_special_tokens=False)) for r in predictions]
            sorted_idx  = np.argsort(resp_lens)  # 升序 indices

            for k in mv_powers:
                subset_idx   = sorted_idx[:k]
                subset_vals  = [processed_preds[i] for i in subset_idx]
                subset_flags  = [flags[i] for i in subset_idx]
                sub_maj_val  = Counter(subset_vals).most_common(1)[0][0]

                mv_flag = 0
                for i in subset_idx:
                    if processed_preds[i] == sub_maj_val:
                        mv_flag = flags[i]
                        break
                mv_len_correct[k] += mv_flag
                prefix_correct_ct[k] += sum(subset_flags) / len(subset_flags)

    # —— 汇总指标 ——
    total_preds       = n_items * n_resp
    accuracy          = correct_total / total_preds
    variance          = float(np.var(all_correct_flags, ddof=0))
    pass_rate         = pass_count / n_items
    majority_accuracy = majority_correct / n_items

    results = dict(
        accuracy=accuracy,
        variance=variance,
        pass_rate=pass_rate,
        majority_accuracy=majority_accuracy,
        num_correct_distribution=dict(sorted(dist_correct.items())),
        mv_by_length={k: mv_len_correct[k] / n_items for k in mv_powers},
        acc_by_length={k:prefix_correct_ct[k] / n_items for k in mv_powers},
    )

    # —— 构造摘要 ——
    record_lines.append("===== Evaluation Results =====")
    record_lines.append(f"Number of samples         : {n_items}")
    record_lines.append(f"Responses per sample (n)  : {n_resp}")
    record_lines.append(f"Overall Accuracy          : {accuracy:.4%}")
    record_lines.append(f"Variance of Correctness   : {variance:.6f}")
    record_lines.append(f"Pass Rate (≥1 correct)    : {pass_rate:.4%}")
    record_lines.append(f"Majority Vote Accuracy    : {majority_accuracy:.4%}")

    record_lines.append("-- Distribution of Exactly k Correct Responses (0 – n) --")
    max_cnt   = max(dist_correct.values())
    bar_width = 50
    scale     = bar_width / max_cnt if max_cnt else 1
    for k in range(n_resp + 1):
        cnt = dist_correct.get(k, 0)
        bar = "█" * int(cnt * scale)
        record_lines.append(f"{k:>2} | {bar:<{bar_width}} {cnt}")

    if tokenizer is not None:
        record_lines.append("-- MV and mean Accuracy by Shortest-k Responses (after extraction, length ascending) --")
        for k in mv_powers:
            record_lines.append(f"k={k:<3} mv: {results['mv_by_length'][k]:.4%}")
        print('-----------------------------------')
        for k in mv_powers:
            record_lines.append(f"k={k:<3} mean: {results['acc_by_length'][k]:.4%}")

    record_lines.append("=" * 28 + "\n")
    record_str = "\n".join(record_lines)

    print(record_str)
    return record_str, results




if __name__ == "__main__":
    import argparse
    import json
    import os
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(
        description="Evaluate a math dataset and save length-accuracy figures."
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help="Path to a single dataset folder (e.g. .../initial/math500)"
    )
    parser.add_argument(
        "--tokenizer_path",
        default="\path\to\your\model",
        help="HuggingFace tokenizer checkpoint (default: Qwen2.5-Math-1.5B-Instruct)"
    )
    args = parser.parse_args()

    # ----- Load -------------------------------------------------------------
    print(f"[INFO] Loading dataset:    {args.dataset_dir}")
    dataset   = load_from_disk(args.dataset_dir)

    print(f"[INFO] Loading tokenizer:  {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # ----- Figure / output dir ---------------------------------------------
    # <dataset_dir_parent>/eval_figs/
    parent_dir = os.path.dirname(args.dataset_dir)
    fig_dir    = os.path.join(parent_dir, "eval_figs")
    os.makedirs(fig_dir, exist_ok=True)

    ds_name  = os.path.basename(args.dataset_dir.rstrip("/"))
    fig_path = os.path.join(fig_dir, f"{ds_name}_length_histogram.png")

    # ----- Run evaluation ---------------------------------------------------
    record_str, results = evaluate_dataset2(dataset, tokenizer, mv_powers=[4,8,16,24,32])

    # ----- Save summary & json ---------------------------------------------
    txt_path = os.path.join(fig_dir, f"{ds_name}_eval.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(record_str)

    json_path = os.path.join(fig_dir, f"{ds_name}_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Figure saved to:    {fig_path}")
    print(f"[INFO] Summary saved to:   {txt_path}")
    print(f"[INFO] JSON saved to:      {json_path}")


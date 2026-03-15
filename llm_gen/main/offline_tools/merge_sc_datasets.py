#!/usr/bin/env python3
"""
Merge & deduplicate self‑correction datasets **by conversation chain** and annotate difficulty
===========================================================================================
This version fixes the earlier bug where deduplication relied only on the first
user question.  Now two chains that start with the same question but diverge
later are treated as distinct unless one is a *strict prefix* of the other.

Example
-------
* chain‑A : **Q1 → R1 → R2**
* chain‑B : **Q1 → R1 → R3**
* chain‑C : **Q1 → R1 → R3 → R4**  (appears in a deeper loop)

The algorithm will keep **chain‑A** and **chain‑C**.  Because chain‑B is a strict
prefix of chain‑C it is dropped, but chain‑A survives since it diverges at the
second assistant turn.

Difficulty estimation and usage stay exactly the same as before:

```bash
python merge_dedup_compute_difficulty.py \
    /nvme1/wpz/llm_gen/results/sc/Qwen2.5-Mit-1.5B-sc \
    --output_dir /nvme1/wpz/llm_gen/merged_dataset
```
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Dict, List
from tools.math_eval import math_lst_eval 
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from datasets import Features, Sequence, Value

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def chain_len(example: Dict) -> int:
    """Return number of conversation turns (user + assistant)."""
    return len(example["conversations"])


def question_key(example: Dict) -> str:
    """Use the first user turn as the *question* key (for difficulty table)."""
    return example["conversations"][0]["content"].strip()


def is_prefix(short: List[Dict], long: List[Dict]) -> bool:
    """True iff *short* conversation is an exact prefix of *long*."""
    if len(short) >= len(long):
        return False
    for a, b in zip(short, long):
        if a["role"] != b["role"] or a["content"].strip() != b["content"].strip():
            return False
    return True

# ---------------------------------------------------------------------------
# Difficulty estimation from *initial*
# ---------------------------------------------------------------------------

def annotate_difficulty(ds_initial):
    """Return dict: question → difficulty dict (easy / medium / hard)."""

    diffs = {}
    n_total = len(ds_initial[0]["responses"])  # responses per item

    for ex in tqdm(ds_initial, desc="Scoring difficulty"):
        gold = ex["gt_answer"]
        flags = math_lst_eval([gold] * n_total, ex["responses"])
        n_correct = int(sum(flags))
        ratio = n_correct / n_total
        if ratio >= 0.75:
            level = "easy"
        elif ratio >= 0.375:
            level = "medium"
        else:
            level = "hard"
        diffs[question_key(ex)] = {
            "level": level,
            "n_correct": n_correct,
            "n_total": n_total,
        }
    return diffs

# ---------------------------------------------------------------------------
# Main merge logic – prefix‑aware deduplication
# ---------------------------------------------------------------------------

def merge_loops(root: str, output_dir: str, ds_name: str):
    """Merge all loops under *root* and write consolidated dataset to *output_dir*."""

    # Determine all sub‑dataset names via *initial*
    initial_dir = os.path.join(root, "initial")

    # 1. Build difficulty lookup table
    print("Build difficulty lookup table...")
    difficulty_map = {}
    ds_init = load_from_disk(os.path.join(initial_dir, ds_name))
    difficulty_map.update(annotate_difficulty(ds_init))

    # 2. Folder traversal order: deepest loop → shallowest → initial
    loop_dirs = [d for d in os.listdir(root) if d.startswith("correct_loop")]
    loop_dirs.sort(key=lambda s: int(re.findall(r"\d+", s)[0]))
    folder_order = loop_dirs

    merged: Dict[str, List[Dict]] = {ds_name: []}

    for folder in folder_order:
        print(f"merge from {folder}...")
        ds_path = os.path.join(root, folder, ds_name)
        ds = load_from_disk(ds_path)
        for ex in ds:
            ex['metadata'] = json.dumps(ex['metadata'])
            chains = merged[ds_name]
            conv = ex["conversations"]

            replaced = False
            i = 0
            while i < len(chains):
                keep_conv = chains[i]["conversations"]
                if is_prefix(keep_conv, conv):
                    # existing chain is shorter prefix – replace it
                    chains[i] = ex
                    replaced = True
                    break
                elif is_prefix(conv, keep_conv):
                    raise ValueError("This should not happen")
                i += 1

            if not replaced:
                chains.append(ex)

    # 3. Inject difficulty & build DatasetDict

    for ex in merged[ds_name]:
        diff = difficulty_map.get(question_key(ex), {
            "level": "unknown", "n_correct": None, "n_total": None,
        })
        ex["difficulty"] = diff
        
    merged_ds = Dataset.from_list(merged[ds_name])
        

    os.makedirs(output_dir, exist_ok=True)
    merged_ds.save_to_disk(os.path.join(output_dir, ds_name))
    print(f"✔ Merged dataset written → {output_dir}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge self‑correction loops (prefix‑aware)")
    p.add_argument("--root_dir", help="Directory containing initial/ and correct_loop*/ folders")
    p.add_argument("--output_dir", default="merged_dataset", help="Target directory")
    p.add_argument('--datasets', nargs='+', help='输入一个或多个项目')
    args = p.parse_args()
    
    for dataset in args.datasets:
        print(f"Processing {dataset}")
        merge_loops(args.root_dir, args.output_dir, dataset)

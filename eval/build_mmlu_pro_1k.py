#!/usr/bin/env python3
"""Build a `mmlu_pro_1k` lm-evaluation-harness task: the canonical MMLU-Pro
protocol (per-subject descriptions + subject-matched 5-shot CoT + exact_match),
but with each subject's TEST split randomly subsampled (fixed seed) so the total
is ~1000 questions. The same 1000 questions are used for every model, enabling a
fair, cheaper comparison than the full ~12k set.

Usage:
    python build_mmlu_pro_1k.py [--out ~/tasks/mmlu_pro_1k_dir] [--target 1000] [--seed 42]

Then run with:  lm_eval ... --tasks mmlu_pro_1k --include_path <parent-of-out>
"""
import argparse, os, shutil, re
from collections import Counter

import lm_eval
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/tasks/mmlu_pro_1k_dir"))
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = os.path.join(os.path.dirname(lm_eval.__file__), "tasks", "mmlu_pro")
    dst = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(base, dst)

    # proportional per-subject counts that sum to --target
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    cnt = Counter(ds["category"]); total = sum(cnt.values())
    K = {c: max(1, round(n / total * args.target)) for c, n in cnt.items()}
    diff = args.target - sum(K.values())
    for c in sorted(K, key=lambda x: -cnt[x]):
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        K[c] += step; diff -= step
    print("per-subject sample counts:", K, "=> total", sum(K.values()))

    # rename task/group ids: mmlu_pro -> mmlu_pro_1k (leaves 'TIGER-Lab/MMLU-Pro' untouched)
    for f in os.listdir(dst):
        if f.endswith(".yaml"):
            p = os.path.join(dst, f)
            open(p, "w").write(open(p).read().replace("mmlu_pro", "mmlu_pro_1k"))

    # patch process_docs: seed-subsample only the large TEST split (the small
    # validation/few-shot pool is < K per subject, so `len(ds) > k` leaves it intact)
    up = os.path.join(dst, "utils.py"); s = open(up).read()
    inject = (
        "\n_SAMPLE = %r\n_SEED = %d\n\n"
        "def process_docs(dataset, subject):\n"
        "    ds = dataset.filter(lambda x: x[\"category\"] == subject)\n"
        "    k = _SAMPLE.get(subject)\n"
        "    if k is not None and len(ds) > k:\n"
        "        ds = ds.shuffle(seed=_SEED).select(range(k))\n"
        "    return ds\n"
    ) % (K, args.seed)
    s = re.sub(
        r"\ndef process_docs\(dataset, subject\):\n    return dataset\.filter\(lambda x: x\[.category.\] == subject\)\n",
        inject, s)
    open(up, "w").write(s)
    print(f"built mmlu_pro_1k at {dst}")


if __name__ == "__main__":
    main()

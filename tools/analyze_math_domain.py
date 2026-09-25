#!/usr/bin/env python3
"""Answer the math-domain question: does math-adapted BPB rank-agree with
GSM8K accuracy better than the paper's existing news-adapted-BPB baseline (rho=0.727)?

Reuses the exact statistics (ranks, Pearson, Kendall tau_a) from
tools/analyze_static_benchmarks.py so the two are directly comparable -- same formulas,
same accuracy source, only the BPB domain differs (results/domain_transfer/math__*.json
instead of news__*.json).

Only the Python standard library plus tools/analyze_static_benchmarks.py is required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_static_benchmarks import (  # noqa: E402
    BENCHMARKS, MODEL_IDS, DISPLAY, ROOT, DOMAIN_DIR, STATIC_PATH,
    ranks, pearson, kendall_tau_a,
)

OUTPUT_JSON = ROOT / "results" / "math_domain_benchmark_analysis.json"

# The paper's existing news-adapted-BPB baseline (paper_sota.tex Table tab:rank_alignment /
# results/static_benchmark_analysis.json), reproduced here as the comparison point.
NEWS_BASELINE = {
    "mmlu_pro": {"spearman": 0.763636363636, "kendall_tau": 0.563636363636, "exact_ranks": 3},
    "hellaswag": {"spearman": 0.981818181818, "kendall_tau": 0.927272727273, "exact_ranks": 7},
    "gsm8k": {"spearman": 0.727272727273, "kendall_tau": 0.563636363636, "exact_ranks": 0},
}


def load_math_bpb() -> dict[str, float]:
    by_id = {}
    for path in sorted(DOMAIN_DIR.glob("math__*.json")):
        row = json.loads(path.read_text())
        by_id[row["model_id"]] = row["adapted_bpb"]
    expected = set(MODEL_IDS.values())
    missing = expected - set(by_id)
    if missing:
        raise SystemExit(f"math BPB coverage incomplete: missing {sorted(missing)} "
                          f"({len(by_id)}/{len(expected)} cells present)")
    extra = set(by_id) - expected
    if extra:
        raise SystemExit(f"math BPB has unexpected model ids: {sorted(extra)}")
    return {label: by_id[model_id] for label, model_id in MODEL_IDS.items()}


def main() -> None:
    static = json.loads(STATIC_PATH.read_text())
    if set(static) != set(MODEL_IDS):
        raise SystemExit("static benchmark model set does not match the 11-model cohort")

    bpb = load_math_bpb()
    bpb_rank = ranks(bpb, lower_is_better=True)
    task_ranks = {
        task: ranks({m: static[m][task] for m in static}, lower_is_better=False)
        for task in BENCHMARKS
    }

    print("=== Math-adapted BPB (this experiment) ===")
    for model in sorted(bpb, key=bpb.get):
        print(f"  {DISPLAY[model]:<18} adapted_bpb={bpb[model]:.4f}  rank={bpb_rank[model]:.1f}")

    alignment = {}
    models = list(static)
    bpb_values = [bpb[m] for m in models]
    bpb_ranks = [bpb_rank[m] for m in models]

    print("\n=== Rank alignment: math-adapted BPB vs. each benchmark ===")
    print(f"{'task':<10} {'pearson_r':>10} {'spearman':>10} {'kendall_tau':>12} "
          f"{'exact_ranks':>12} {'news_baseline_spearman':>24}")
    for task, metadata in BENCHMARKS.items():
        accuracies = [static[m][task] for m in models]
        accuracy_ranks = [task_ranks[task][m] for m in models]
        tau = kendall_tau_a(bpb_values, [-x for x in accuracies])
        shifts = [abs(bpb_rank[m] - task_ranks[task][m]) for m in models]
        spearman = pearson(bpb_ranks, accuracy_ranks)
        alignment[task] = {
            "label": metadata["label"],
            "pearson_bpb_vs_accuracy": pearson(bpb_values, accuracies),
            "spearman_rank_alignment": spearman,
            "kendall_tau_bpb_vs_error": tau,
            "identical_ranks": sum(shift == 0 for shift in shifts),
            "mean_absolute_rank_shift": sum(shifts) / len(shifts),
            "max_absolute_rank_shift": max(shifts),
            "news_baseline_spearman": NEWS_BASELINE[task]["spearman"],
            "spearman_delta_vs_news": spearman - NEWS_BASELINE[task]["spearman"],
        }
        print(f"{task:<10} {alignment[task]['pearson_bpb_vs_accuracy']:>10.3f} "
              f"{spearman:>10.3f} {tau:>12.3f} "
              f"{alignment[task]['identical_ranks']:>12d} "
              f"{NEWS_BASELINE[task]['spearman']:>24.3f}")

    gsm8k = alignment["gsm8k"]
    print(f"\n=== The question docs/PLAN.md asked ===")
    print(f"News-adapted BPB <-> GSM8K rank agreement (existing baseline): rho=0.727")
    print(f"Math-adapted BPB <-> GSM8K rank agreement (this experiment):   "
          f"rho={gsm8k['spearman_rank_alignment']:.3f}")
    delta = gsm8k["spearman_delta_vs_news"]
    if delta > 0.05:
        verdict = "IMPROVED: domain-matched adaptation narrows the GSM8K residual."
    elif delta < -0.05:
        verdict = "WORSE: math-domain adaptation did not help (or hurt) GSM8K alignment."
    else:
        verdict = ("FLAT: no material change -- consistent with a capability/elicitation "
                   "residual, not a domain-mismatch artifact.")
    print(f"Delta: {delta:+.3f}  ->  {verdict}")

    OUTPUT_JSON.write_text(json.dumps({
        "schema_version": 1,
        "bpb_source": "results/domain_transfer/math__*.json",
        "accuracy_source": "results/combined_bpb_vs_static.json",
        "news_baseline": NEWS_BASELINE,
        "alignment": alignment,
        "adapted_bpb": bpb,
    }, indent=2) + "\n")
    print(f"\nwrote {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

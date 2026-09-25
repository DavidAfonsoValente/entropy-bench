#!/usr/bin/env python3
"""Two discriminating checks flagged by advisor review before the math-domain
result (rho=0.727 -> 0.936 for GSM8K) gets treated as settled:

1. Spearman(math-adapted-BPB ranks, news-adapted-BPB ranks) -- if this is already ~0.95+,
   the math corpus mostly reproduced the same general-quality ordering more sharply, rather
   than reordering models toward math-specific ability. That would mean the GSM8K gain is
   substantially a byproduct of a cleaner general signal, not a math-domain-specific one.

2. Leave-one-out sensitivity of math-adapted-BPB <-> GSM8K Spearman: recompute the
   correlation 11 times, each time dropping one model, to check whether the +0.209 delta over
   the news baseline is being carried by one or two point movers (Gemma-4-31B moved from
   rank 1 under news-BPB to rank 4 under math-BPB, and Gemma is the family the paper says
   underperforms on GSM8K specifically).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_static_benchmarks import (  # noqa: E402
    BENCHMARKS, MODEL_IDS, DISPLAY, ROOT, DOMAIN_DIR, STATIC_PATH,
    ranks, pearson,
)

NEWS_GSM8K_BASELINE = 0.727272727273


def load_domain_bpb(tag: str) -> dict[str, float]:
    by_id = {}
    for path in sorted(DOMAIN_DIR.glob(f"{tag}__*.json")):
        row = json.loads(path.read_text())
        by_id[row["model_id"]] = row["adapted_bpb"]
    expected = set(MODEL_IDS.values())
    missing = expected - set(by_id)
    if missing:
        raise SystemExit(f"{tag} BPB coverage incomplete: missing {sorted(missing)}")
    return {label: by_id[model_id] for label, model_id in MODEL_IDS.items()}


def main() -> None:
    static = json.loads(STATIC_PATH.read_text())
    math_bpb = load_domain_bpb("math")
    news_bpb = load_domain_bpb("news")
    models = list(static)
    if set(models) != set(math_bpb) or set(models) != set(news_bpb):
        raise SystemExit("model set mismatch across static/math/news")

    math_rank = ranks(math_bpb, lower_is_better=True)
    news_rank = ranks(news_bpb, lower_is_better=True)

    print("=== Check 1: does math-adapted BPB just sharpen the same ordering as news? ===")
    print(f"{'model':<18} {'news_rank':>10} {'math_rank':>10} {'shift':>6}")
    for m in sorted(models, key=lambda x: math_rank[x]):
        print(f"{DISPLAY[m]:<18} {news_rank[m]:>10.1f} {math_rank[m]:>10.1f} "
              f"{news_rank[m]-math_rank[m]:>+6.1f}")
    rho_news_math = pearson([news_rank[m] for m in models], [math_rank[m] for m in models])
    print(f"\nSpearman(news-adapted-BPB rank, math-adapted-BPB rank) = {rho_news_math:.3f}")
    if rho_news_math >= 0.90:
        print("-> HIGH: math corpus mostly reproduces the news ordering, sharper. The GSM8K "
              "gain may be substantially a general-signal-cleanliness effect, not purely "
              "math-domain-specific reordering. Report both numbers, don't claim pure "
              "domain-matching.")
    else:
        print("-> Math-adapted BPB meaningfully reorders models relative to news-adapted BPB: "
              "consistent with a genuine domain-specific signal.")

    print("\n=== Check 2: leave-one-out sensitivity of math-BPB <-> GSM8K Spearman ===")
    gsm8k_acc = {m: static[m]["gsm8k"] for m in models}
    full_acc_rank = ranks(gsm8k_acc, lower_is_better=False)
    full_rho = pearson([math_rank[m] for m in models], [full_acc_rank[m] for m in models])
    print(f"Full 11-model Spearman: {full_rho:.3f} (news baseline: {NEWS_GSM8K_BASELINE:.3f})")
    print(f"{'dropped_model':<18} {'loo_spearman':>13} {'delta_vs_full':>14}")
    loo = {}
    for drop in models:
        subset = [m for m in models if m != drop]
        sub_math_bpb = {m: math_bpb[m] for m in subset}
        sub_acc = {m: gsm8k_acc[m] for m in subset}
        sub_math_rank = ranks(sub_math_bpb, lower_is_better=True)
        sub_acc_rank = ranks(sub_acc, lower_is_better=False)
        r = pearson([sub_math_rank[m] for m in subset], [sub_acc_rank[m] for m in subset])
        loo[drop] = r
        print(f"{DISPLAY[drop]:<18} {r:>13.3f} {r-full_rho:>+14.3f}")

    worst_drop = min(loo, key=loo.get)
    best_drop = max(loo, key=loo.get)
    print(f"\nRange across leave-one-out: [{loo[worst_drop]:.3f}, {loo[best_drop]:.3f}]")
    print(f"Most-influential single model (dropping it changes rho the most): "
          f"{DISPLAY[max(loo, key=lambda m: abs(loo[m]-full_rho))]}")
    gemma_models = [m for m in models if "Gemma" in DISPLAY[m] or "gemma" in m.lower()]
    print("\nGemma-specific check (paper's own claim: Gemma family underperforms GSM8K):")
    for g in gemma_models:
        print(f"  drop {DISPLAY[g]:<18} -> loo_rho={loo[g]:.3f} "
              f"(vs full={full_rho:.3f}, vs news baseline={NEWS_GSM8K_BASELINE:.3f})")


if __name__ == "__main__":
    main()

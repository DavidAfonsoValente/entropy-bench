#!/usr/bin/env python3
"""The full BPB-to-benchmark alignment matrix: every corpus, both cost tiers, all benchmarks.

The paper previously reported 6 of the 24 available (corpus x tier x benchmark) alignments --
news-adapted and math-adapted only -- which left the two controls a reader needs unstated:
what the *unadapted* ranking already achieves, and whether a non-matched second corpus would
have moved GSM8K just as well. Both were already computable from the committed cells; this
script computes all of them.

It also replaces rank correlation as the headline statistic. Spearman rho over 11 models has
very wide intervals and answers a question nobody asks. Selection regret -- "pick the top model
by this metric, how much benchmark accuracy do you give up against the oracle?" -- is the
decision a practitioner actually makes, and it is reported here alongside pairwise accuracy and
bootstrap intervals on the differences that carry claims.

Reuses tools/analyze_static_benchmarks.py's statistics verbatim so every number here is
directly comparable with the ones already in the paper. Standard library only.

    python tools/analyze_alignment_matrix.py [--check]

``--check`` regenerates into memory and fails if the committed artifacts differ, which is what
CI and tools/verify_paper_numbers.py use.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_static_benchmarks import (  # noqa: E402
    BENCHMARKS, DISPLAY, DOMAIN_DIR, MODEL_IDS, ROOT, STATIC_PATH,
    kendall_tau_a, pearson, ranks,
)

OUTPUT_JSON = ROOT / "results" / "alignment_matrix.json"
OUTPUT_TEX = ROOT / "alignment_matrix_table.tex"
OUTPUT_REGRET_TEX = ROOT / "selection_regret_table.tex"

# Ordered so the table reads free-tier first, then paid-tier, and news (the primary corpus)
# leads each block.
CORPORA = ["news", "reddit", "hackernews", "math"]
CORPUS_LABEL = {"news": "News", "reddit": "Reddit", "hackernews": "Hacker News",
                "math": "Math (arXiv)"}
TIERS = [("zero_shot_bpb", "zero-shot"), ("adapted_bpb", "adapted")]

BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 20260827

# Empirical BPB sensitivity to available byte-context near the 512-token block length, read
# off the committed context-length diagnostic (data/context_length/*.json): Gemma-4-12B base
# BPB falls 0.6465 -> 0.6054 from 512 to 2048 tokens, i.e. ~3.2% per doubling. Used only to
# bound the residual token-blocking bias; the Qwen curve gives ~1.3%, so this is the
# conservative end.
BPB_PER_CONTEXT_DOUBLING = 0.032


def load_bpb_matrix() -> dict[str, dict[str, dict[str, float]]]:
    """{corpus: {tier_key: {display_label: value}}} plus per-corpus tokenizer metadata."""
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    meta: dict[str, dict[str, dict[str, float]]] = {}
    by_model_id = {model_id: label for label, model_id in MODEL_IDS.items()}
    for corpus in CORPORA:
        cells = sorted(DOMAIN_DIR.glob(f"{corpus}__*.json"))
        if len(cells) != len(MODEL_IDS):
            raise SystemExit(
                f"{corpus}: expected {len(MODEL_IDS)} cells, found {len(cells)}"
            )
        tiers: dict[str, dict[str, float]] = {key: {} for key, _ in TIERS}
        tokenizer: dict[str, dict[str, float]] = {}
        for path in cells:
            row = json.loads(path.read_text())
            label = by_model_id.get(row["model_id"])
            if label is None:
                raise SystemExit(f"{path.name}: unexpected model id {row['model_id']}")
            for key, _ in TIERS:
                tiers[key][label] = row[key]
            tokenizer[label] = {
                "avg_bytes_per_token": row["avg_bytes_per_token"],
                "n_test_blocks_scored": row["n_test_blocks_scored"],
                "n_train_blocks": row["n_train_blocks"],
            }
        matrix[corpus] = tiers
        meta[corpus] = tokenizer
    return matrix, meta


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(a)
    return pearson([a[k] for k in keys], [b[k] for k in keys])


def alignment(bpb: dict[str, float], accuracy: dict[str, float]) -> dict[str, float]:
    """All alignment statistics for one (BPB variant, benchmark) pair."""
    keys = sorted(bpb)
    bpb_rank = ranks(bpb, lower_is_better=True)
    acc_rank = ranks(accuracy, lower_is_better=False)
    shifts = [abs(bpb_rank[k] - acc_rank[k]) for k in keys]

    concordant = sum(
        (bpb[x] < bpb[y]) == (accuracy[x] > accuracy[y])
        for x, y in combinations(keys, 2)
        if accuracy[x] != accuracy[y]
    )
    comparable = sum(1 for x, y in combinations(keys, 2) if accuracy[x] != accuracy[y])

    # Selection regret: the metric's top pick versus the benchmark's own best model.
    pick = min(keys, key=lambda k: bpb[k])
    oracle = max(keys, key=lambda k: accuracy[k])
    return {
        "spearman": spearman(bpb_rank, acc_rank),
        "kendall_tau": kendall_tau_a([bpb_rank[k] for k in keys],
                                     [acc_rank[k] for k in keys]),
        "exact_ranks": sum(1 for s in shifts if s == 0),
        "mean_shift": sum(shifts) / len(shifts),
        "max_shift": max(shifts),
        "pairs_correct": concordant,
        "pairs_comparable": comparable,
        "top1_pick": pick,
        "top1_oracle": oracle,
        "top1_regret_pp": (accuracy[oracle] - accuracy[pick]) * 100.0,
    }


def bootstrap_delta(bpb_a: dict[str, float], bpb_b: dict[str, float],
                    accuracy: dict[str, float]) -> dict[str, float]:
    """Paired bootstrap over the model cohort for spearman(B) - spearman(A).

    Resamples models, not observations: the cohort is the sampling unit and the benchmark
    ranking is recomputed within each resample, so this measures how much of the difference
    survives a different draw of 11 models. Deterministic given BOOTSTRAP_SEED.
    """
    keys = sorted(bpb_a)
    rng = random.Random(BOOTSTRAP_SEED)
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [rng.choice(keys) for _ in keys]
        if len(set(sample)) < 4:
            continue
        sub_acc = {f"{k}#{i}": accuracy[k] for i, k in enumerate(sample)}
        sub_a = {f"{k}#{i}": bpb_a[k] for i, k in enumerate(sample)}
        sub_b = {f"{k}#{i}": bpb_b[k] for i, k in enumerate(sample)}
        try:
            delta = (spearman(ranks(sub_b, lower_is_better=True),
                              ranks(sub_acc, lower_is_better=False))
                     - spearman(ranks(sub_a, lower_is_better=True),
                                ranks(sub_acc, lower_is_better=False)))
        except ZeroDivisionError:
            continue
        deltas.append(delta)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    point = (spearman(ranks(bpb_b, lower_is_better=True),
                      ranks(accuracy, lower_is_better=False))
             - spearman(ranks(bpb_a, lower_is_better=True),
                        ranks(accuracy, lower_is_better=False)))
    return {
        "delta_spearman": point,
        "ci_low": lo,
        "ci_high": hi,
        "resamples_used": len(deltas),
        "excludes_zero": lo > 0 or hi < 0,
    }


def benchmark_redundancy(static: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """How much independent signal the three benchmarks actually carry in this cohort."""
    out: dict[str, dict[str, float]] = {}
    for a, b in combinations(sorted(BENCHMARKS), 2):
        keys = sorted(static)
        agree = sum(
            (static[x][a] > static[y][a]) == (static[x][b] > static[y][b])
            for x, y in combinations(keys, 2)
        )
        total = math.comb(len(keys), 2)
        out[f"{a}__vs__{b}"] = {
            "pairs_agree": agree,
            "pairs_total": total,
            "kendall_tau": (agree - (total - agree)) / total,
        }
    return out


def tokenizer_bias_bound(meta: dict[str, dict[str, float]],
                         bpb: dict[str, float]) -> dict[str, object]:
    """Bound the residual bias from token-aligned (rather than byte-aligned) eval blocks.

    Blocks are 512 *tokens*, so a model whose tokenizer packs fewer bytes per token scores
    slightly less text and pays block-opening cost over fewer bytes. Early block positions are
    much more expensive per byte, so this is a real if small penalty. Bound it by treating the
    bytes-per-block ratio as a change in available byte-context and applying the measured
    context sensitivity.
    """
    ordered = sorted(bpb, key=bpb.get)
    worst: dict[str, object] = {"bias_over_gap": 0.0}
    rows = []
    for lower, higher in zip(ordered, ordered[1:]):
        gap = (bpb[higher] - bpb[lower]) / bpb[lower]
        bpt_lower = meta[lower]["avg_bytes_per_token"]
        bpt_higher = meta[higher]["avg_bytes_per_token"]
        bpt_delta = abs(bpt_higher - bpt_lower) / bpt_lower
        bias = BPB_PER_CONTEXT_DOUBLING * math.log2(1 + bpt_delta)
        ratio = bias / gap if gap else float("inf")
        row = {"better": lower, "worse": higher, "bpb_gap_frac": gap,
               "bytes_per_token_delta_frac": bpt_delta, "bias_bound_frac": bias,
               "bias_over_gap": ratio}
        rows.append(row)
        if ratio > worst["bias_over_gap"]:
            worst = dict(row)
    scored_mb = {
        m: meta[m]["n_test_blocks_scored"] * 512 * meta[m]["avg_bytes_per_token"] / 1e6
        for m in bpb
    }
    return {
        "adjacent_pairs": rows,
        "worst_case": worst,
        "max_bias_over_gap": worst["bias_over_gap"],
        "scored_text_mb": scored_mb,
        "scored_text_spread_frac": (max(scored_mb.values()) - min(scored_mb.values()))
        / min(scored_mb.values()),
        "context_sensitivity_per_doubling": BPB_PER_CONTEXT_DOUBLING,
        "overturns_any_rank": worst["bias_over_gap"] >= 1.0,
    }


def build() -> dict[str, object]:
    static = json.loads(STATIC_PATH.read_text())
    if set(static) != set(MODEL_IDS):
        raise SystemExit("static benchmark model set does not match the 11-model cohort")
    matrix, meta = load_bpb_matrix()

    cells: dict[str, dict[str, dict[str, object]]] = {}
    for corpus in CORPORA:
        for tier_key, tier_label in TIERS:
            name = f"{corpus}__{tier_label.replace('-', '_')}"
            bpb = matrix[corpus][tier_key]
            cells[name] = {
                "corpus": corpus,
                "tier": tier_label,
                "bpb": bpb,
                "alignment": {
                    task: alignment(bpb, {m: static[m][task] for m in static})
                    for task in BENCHMARKS
                },
            }

    # The three comparisons that carry claims in the paper, each against the baseline it
    # should be judged against rather than the one that flatters it.
    contrasts = {
        "math_adapted_vs_news_adapted": ("news", "adapted_bpb", "math", "adapted_bpb"),
        "math_adapted_vs_news_zero_shot": ("news", "zero_shot_bpb", "math", "adapted_bpb"),
        "news_adapted_vs_news_zero_shot": ("news", "zero_shot_bpb", "news", "adapted_bpb"),
        "math_adapted_vs_hackernews_adapted": ("hackernews", "adapted_bpb",
                                               "math", "adapted_bpb"),
        "math_adapted_vs_reddit_adapted": ("reddit", "adapted_bpb", "math", "adapted_bpb"),
    }
    deltas = {
        name: {
            task: bootstrap_delta(matrix[ca][ka], matrix[cb][kb],
                                  {m: static[m][task] for m in static})
            for task in BENCHMARKS
        }
        for name, (ca, ka, cb, kb) in contrasts.items()
    }

    return {
        "schema_version": 1,
        "bpb_source": "results/domain_transfer/{corpus}__*.json",
        "accuracy_source": "results/combined_bpb_vs_static.json",
        "note": (
            "Spearman is retained for continuity with the paper's earlier tables, but the "
            "headline statistics are selection regret and pairwise accuracy: rank correlation "
            "over an 11-model cohort has intervals too wide to carry a claim on its own."
        ),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "unit": "model"},
        "cells": cells,
        "contrasts": deltas,
        "benchmark_redundancy": benchmark_redundancy(static),
        "tokenizer_bias_bound": tokenizer_bias_bound(meta["news"],
                                                     matrix["news"]["adapted_bpb"]),
    }


def _portable(value: object) -> object:
    """Round floats to 12 significant digits so the artifact is byte-identical across machines.

    math.log2 (the tokenizer-bias bound) is not correctly rounded by every libm: CI's glibc and
    Leonardo's differ in the 17th digit, which made the committed JSON "stale" on every CI run.
    """
    if isinstance(value, float):
        return float(f"{value:.12g}")
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    return value


def render_matrix_tex(report: dict[str, object]) -> str:
    """Plain rows only: \\multicolumn after a booktabs rule is fragile inside \\input."""
    lines = ["% generated by tools/analyze_alignment_matrix.py"]
    for corpus in CORPORA:
        for index, (_, tier_label) in enumerate(TIERS):
            cell = report["cells"][f"{corpus}__{tier_label.replace('-', '_')}"]
            a = cell["alignment"]
            row = " & ".join([
                CORPUS_LABEL[corpus] if index == 0 else "",
                tier_label,
                f"{a['gsm8k']['spearman']:.3f}",
                f"{a['mmlu_pro']['spearman']:.3f}",
                f"{a['hellaswag']['spearman']:.3f}",
                # pairs (of 55) ordered the same way as each benchmark, in the same column order
                *(str(a[b]["pairs_correct"]) for b in ("gsm8k", "mmlu_pro", "hellaswag")),
            ])
            lines.append(row + " \\\\")
        if corpus != CORPORA[-1]:
            lines.append("\\addlinespace")
    return "\n".join(lines) + "\n"


def render_regret_tex(report: dict[str, object]) -> str:
    lines = ["% generated by tools/analyze_alignment_matrix.py"]
    for corpus in CORPORA:
        for _, tier_label in TIERS:
            cell = report["cells"][f"{corpus}__{tier_label.replace('-', '_')}"]
            a = cell["alignment"]
            pick = DISPLAY[a["gsm8k"]["top1_pick"]]
            row = " & ".join([
                f"{CORPUS_LABEL[corpus]} {tier_label}",
                pick,
                f"{a['gsm8k']['top1_regret_pp']:.1f}",
                f"{a['mmlu_pro']['top1_regret_pp']:.1f}",
                f"{a['hellaswag']['top1_regret_pp']:.1f}",
            ])
            lines.append(row + " \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if committed artifacts differ from a fresh computation")
    args = parser.parse_args()

    report = build()
    matrix_tex = render_matrix_tex(report)
    regret_tex = render_regret_tex(report)
    payload = json.dumps(_portable(report), indent=2, sort_keys=True) + "\n"

    if args.check:
        problems = []
        for path, fresh in ((OUTPUT_JSON, payload), (OUTPUT_TEX, matrix_tex),
                            (OUTPUT_REGRET_TEX, regret_tex)):
            if not path.exists():
                problems.append(f"{path.relative_to(ROOT)} is missing")
            elif path.read_text() != fresh:
                # Show where it differs: a bare "stale" is undebuggable when it only fails in CI.
                import difflib
                diff = [line for line in difflib.unified_diff(
                    path.read_text().splitlines(), fresh.splitlines(), "committed", "regenerated",
                    lineterm="", n=0) if not line.startswith("@@")][:14]
                problems.append(f"{path.relative_to(ROOT)} is stale:\n    " + "\n    ".join(diff))
        if problems:
            raise SystemExit("alignment matrix out of date:\n  " + "\n  ".join(problems))
        print("alignment matrix artifacts are up to date")
        return

    OUTPUT_JSON.write_text(payload)
    OUTPUT_TEX.write_text(matrix_tex)
    OUTPUT_REGRET_TEX.write_text(regret_tex)

    print("=== Rank alignment, every corpus and cost tier (Spearman rho) ===")
    print(f"{'variant':<26}{'GSM8K':>9}{'MMLU-Pro':>10}{'HellaSwag':>11}{'GSM8K pairs':>14}")
    for corpus in CORPORA:
        for _, tier in TIERS:
            a = report["cells"][f"{corpus}__{tier.replace('-', '_')}"]["alignment"]
            print(f"{corpus + ' ' + tier:<26}"
                  f"{a['gsm8k']['spearman']:>9.3f}{a['mmlu_pro']['spearman']:>10.3f}"
                  f"{a['hellaswag']['spearman']:>11.3f}"
                  f"{a['gsm8k']['pairs_correct']:>10}/{a['gsm8k']['pairs_comparable']}")

    print("\n=== Selection regret: accuracy given up by taking the metric's top pick ===")
    print(f"{'variant':<26}{'picks':<18}{'GSM8K':>8}{'MMLU':>8}{'HSwag':>8}")
    for corpus in CORPORA:
        for _, tier in TIERS:
            a = report["cells"][f"{corpus}__{tier.replace('-', '_')}"]["alignment"]
            print(f"{corpus + ' ' + tier:<26}{DISPLAY[a['gsm8k']['top1_pick']]:<18}"
                  f"{a['gsm8k']['top1_regret_pp']:>7.1f}{a['mmlu_pro']['top1_regret_pp']:>8.1f}"
                  f"{a['hellaswag']['top1_regret_pp']:>8.1f}")

    print("\n=== Bootstrap intervals on the contrasts that carry claims (GSM8K) ===")
    for name, per_task in report["contrasts"].items():
        d = per_task["gsm8k"]
        flag = "  CI excludes 0" if d["excludes_zero"] else "  CI includes 0"
        print(f"  {name:<40} {d['delta_spearman']:+.3f}  "
              f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{flag}")

    red = report["benchmark_redundancy"]["gsm8k__vs__mmlu_pro"]
    print(f"\nGSM8K and MMLU-Pro agree on {red['pairs_agree']}/{red['pairs_total']} model "
          f"pairs (Kendall tau = {red['kendall_tau']:.3f}) -- not independent evidence.")
    bias = report["tokenizer_bias_bound"]
    print(f"Token-blocking bias bound: at most {bias['max_bias_over_gap']:.2f} x the adjacent "
          f"rank gap; overturns a rank: {bias['overturns_any_rank']}")
    for path in (OUTPUT_JSON, OUTPUT_TEX, OUTPUT_REGRET_TEX):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

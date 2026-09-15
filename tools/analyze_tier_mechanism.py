#!/usr/bin/env python3
"""WHY the free tier fails, not just that it does.

Section~\\ref{sec:cloze} demotes zero-shot BPB from selector to screen on statistical grounds: its
interval spans chance among comparable-size candidates. That is a reason to stop recommending it
and not an explanation, and an unexplained retraction is the kind a reader is right to distrust.

This tests the mechanism we think is responsible. Zero-shot BPB measures two things at once: how
well a model predicts the target domain, and how far its output conventions sit from that corpus's
surface form. Adaptation removes the second -- Section~\\ref{sec:tokengain} shows roughly 83% of the
gain landing on formatting, punctuation and function words -- so if the mechanism is right, the
models zero-shot MIS-RANKS should be exactly the models that need the most adaptation, and that
relationship should disappear once they have had it.

The statistic is deliberately simple, because the cohort cannot support anything elaborate: rank
error against the in-domain cloze criterion, per model, before and after adaptation, correlated
with each model's own BPB reduction. Eleven models, so this is a described pattern with named
instances and not an estimate -- the correlation is reported to place the pattern, not to test it.

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _artifact_check import _diff  # noqa: E402
from analyze_cloze_validity import ROOT, load_bpb_matrix, load_cloze  # noqa: E402

OUTPUT_JSON = ROOT / "results" / "tier_mechanism.json"
SCHEMA_VERSION = 1
CORPUS = "news"
METRIC = "lenient_accuracy"


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else None


def _ranks(values, lower_is_better=True):
    order = sorted(values, key=lambda m: values[m] if lower_is_better else -values[m])
    return {m: i + 1 for i, m in enumerate(order)}


def build() -> dict:
    bpb, _ = load_bpb_matrix()
    cloze = load_cloze(ROOT / "results" / "cloze", METRIC)
    zero, adapt = bpb[CORPUS]["zero_shot_bpb"], bpb[CORPUS]["adapted_bpb"]
    models = sorted(cloze)

    zr, ar = _ranks({m: zero[m] for m in models}), _ranks({m: adapt[m] for m in models})
    cr = _ranks({m: -cloze[m] for m in models})

    per = {}
    for m in models:
        per[m] = {
            "reduction_pct": 100.0 * (zero[m] - adapt[m]) / zero[m],
            "zero_shot_bpb": zero[m], "adapted_bpb": adapt[m], "cloze": cloze[m],
            "rank_zero_shot": zr[m], "rank_adapted": ar[m], "rank_cloze": cr[m],
            # Positive means the tier ranks the model WORSE than the criterion says it is.
            "rank_error_zero_shot": zr[m] - cr[m],
            "rank_error_adapted": ar[m] - cr[m],
        }

    red = [per[m]["reduction_pct"] for m in models]
    ez = [per[m]["rank_error_zero_shot"] for m in models]
    ea = [per[m]["rank_error_adapted"] for m in models]
    mean_abs = lambda v: sum(abs(x) for x in v) / len(v)   # noqa: E731

    # The pairs the two tiers actually disagree on, named. A mechanism claim should be checkable
    # against instances rather than only against a correlation over eleven points.
    from itertools import combinations
    flips = []
    for x, y in combinations(models, 2):
        if zero[x] == zero[y] or adapt[x] == adapt[y] or cloze[x] == cloze[y]:
            continue
        z_ok = (zero[x] < zero[y]) == (cloze[x] > cloze[y])
        a_ok = (adapt[x] < adapt[y]) == (cloze[x] > cloze[y])
        if a_ok and not z_ok:
            flips.append([x, y])
    return {
        "schema_version": SCHEMA_VERSION,
        "question": "Are the models zero-shot BPB mis-ranks the ones that need the most adaptation?",
        "corpus": CORPUS,
        "criterion": f"in-domain cloze, {METRIC}",
        "caveat": "eleven models; a described pattern with named instances, not an estimate. The "
                  "correlations place the pattern and are not tested.",
        "per_model": per,
        "mean_abs_rank_error": {"zero_shot": mean_abs(ez), "adapted": mean_abs(ea)},
        "models_ranked_exactly_right": {
            "zero_shot": sorted(m for m in models if per[m]["rank_error_zero_shot"] == 0),
            "adapted": sorted(m for m in models if per[m]["rank_error_adapted"] == 0)},
        "pearson_reduction_vs_rank_error": {
            "zero_shot": _pearson(red, ez), "adapted": _pearson(red, ea)},
        "pairs_zero_shot_gets_wrong_that_adapted_gets_right": flips,
        "n_such_pairs": len(flips),
        "most_underrated_by_zero_shot": sorted(
            models, key=lambda m: -per[m]["rank_error_zero_shot"])[:3],
        "largest_reductions": sorted(models, key=lambda m: -per[m]["reduction_pct"])[:3],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    r = build()
    if a.check:
        if not OUTPUT_JSON.exists():
            raise SystemExit(f"missing {OUTPUT_JSON.relative_to(ROOT)}; run without --check")
        d = _diff(json.loads(OUTPUT_JSON.read_text()), r)
        if d:
            for x in d[:20]:
                print("  " + x)
            raise SystemExit(f"{OUTPUT_JSON.relative_to(ROOT)} is stale ({len(d)} diffs)")
        print("tier mechanism artifact is up to date")
        return
    OUTPUT_JSON.write_text(json.dumps(r, indent=2) + "\n")
    print(f"{'model':<18}{'red%':>7}{'zs err':>8}{'ad err':>8}")
    for m in sorted(r["per_model"], key=lambda m: -r["per_model"][m]["reduction_pct"]):
        p = r["per_model"][m]
        print(f"{m:<18}{p['reduction_pct']:>6.1f}%{p['rank_error_zero_shot']:>8}"
              f"{p['rank_error_adapted']:>8}")
    pr = r["pearson_reduction_vs_rank_error"]
    print(f"\nr(reduction, rank error): zero-shot {pr['zero_shot']:+.3f}  "
          f"adapted {pr['adapted']:+.3f}")
    print(f"mean |rank error|: zero-shot {r['mean_abs_rank_error']['zero_shot']:.2f}  "
          f"adapted {r['mean_abs_rank_error']['adapted']:.2f}")
    print(f"pairs zero-shot gets wrong that adapted gets right: {r['n_such_pairs']}")
    print(f"most under-rated by zero-shot: {r['most_underrated_by_zero_shot']}")
    print(f"largest reductions:            {r['largest_reductions']}")
    print(f"\nwrote {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does discarding each block's cold start make the free reading as good as the adapted one?

E10. The paper argues that block-averaged zero-shot BPB mis-ranks because it is largely scoring cold
starts. The competing explanation is that the in-domain criterion scores a warmed-up regime and
adapted BPB is warm too, so the two simply meet there. This decides between them without training
anything: re-rank the cohort by zero-shot BPB computed only over token positions >= K, and score
that ranking against the same criterion.

Reads the per-position cells written by ``tools/late_position_bpb.py`` (one JSON per model, produced
by ``slurm/late_position.sbatch``) and reports, for each K, in-band pairwise selection accuracy
against the news criterion, beside the published adapted and zero-shot figures.

**The falsifying outcome is stated in advance:** if any K brings the free reading to within one
in-band pair of adapted BPB's accuracy, adaptation is not doing the work the paper says it does and
Section "Why it works" has to be rewritten. That verdict is computed here, not argued.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_alignment_matrix import load_bpb_matrix  # noqa: E402
from analyze_cloze_validity import MODEL_IDS  # noqa: E402  (the authoritative label -> id map)
from pairwise_significance import _load_params  # noqa: E402

CELLS = Path(os.environ.get("E10_DIR", str(Path(os.environ.get("SCRATCH", "/tmp")) /
                                           "e10_late_position")))
OUTPUT_JSON = ROOT / "results" / "late_position.json"
SCHEMA_VERSION = 1
BAND = 2.0
CORPUS = "news"
# One in-band pair out of 40 is 0.025; "within one pair" is the tolerance the verdict uses.
EQUIVALENCE_PAIRS = 1


def _load_cells() -> dict[str, dict]:
    if not CELLS.is_dir():
        raise SystemExit(f"{CELLS} not found; run slurm/late_position.sbatch first")
    out = {}
    for path in sorted(CELLS.glob("*.json")):
        d = json.loads(path.read_text())
        out[d["model_id"]] = d
    if not out:
        raise SystemExit(f"no cells in {CELLS}")
    return out


def _bpb_from(cell: dict, k: int) -> float:
    ls, tc = cell["loss_sum_by_position"], cell["token_count_by_position"]
    s, n = sum(ls[k:]), sum(tc[k:])
    return (s / n / math.log(2)) / cell["avg_bytes_per_token"]


def _accuracy(scores: dict[str, float], criterion: dict[str, float], params: dict[str, float],
              lower_is_better: bool) -> tuple[float, int]:
    right = total = 0
    for a, b in combinations(sorted(scores), 2):
        if max(params[a], params[b]) / min(params[a], params[b]) > BAND:
            continue
        if criterion[a] == criterion[b] or scores[a] == scores[b]:
            continue
        pick = (a if scores[a] < scores[b] else b) if lower_is_better else (
            a if scores[a] > scores[b] else b)
        best = a if criterion[a] > criterion[b] else b
        right += pick == best
        total += 1
    return (right / total if total else float("nan")), total


def build() -> dict:
    cells = _load_cells()
    unc = json.loads((ROOT / "results" / "criterion_uncertainty.json").read_text())
    params = _load_params()
    bpb, _ = load_bpb_matrix()

    # Label the cells via the repo's own map. A string heuristic was used here first and it
    # silently failed to match Qwen3.5-35B-MoE to Qwen/Qwen3.5-35B-A3B-Base, which the completeness
    # guard caught as INCOMPLETE -- exactly the kind of near-miss a fuzzy match is good at hiding.
    by_label = {label: (MODEL_IDS[label] if MODEL_IDS.get(label) in cells else None)
                for label in unc["cohort_11"]}
    missing = sorted(l for l, m in by_label.items() if m is None)

    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "question": "Does zero-shot BPB computed only over late block positions match adapted BPB "
                    "as a selector? If it does, adaptation is not doing the work.",
        "preregistered_falsifier": "any K within %d in-band pair(s) of adapted BPB's accuracy "
                                   "overturns the mechanism section" % EQUIVALENCE_PAIRS,
        "cells_found": sorted(cells),
        "models_missing": missing,
        "corpus": CORPUS, "band": BAND,
    }
    if missing:
        result["verdict"] = "INCOMPLETE"
        return result

    labels = sorted(by_label)
    for conv in ("lenient", "strict"):
        crit = {m: unc[conv]["per_model"][m]["accuracy"] for m in labels}
        published_adapted = {m: bpb[CORPUS]["adapted_bpb"][m] for m in labels}
        published_zero = {m: bpb[CORPUS]["zero_shot_bpb"][m] for m in labels}
        adapted_acc, n_pairs = _accuracy(published_adapted, crit, params, True)
        zero_acc, _ = _accuracy(published_zero, crit, params, True)

        # These cells are re-measured, not the published ones, and they come in 0.1-0.9% low --
        # systematic but not uniform, and the cohort's smallest adjacent BPB gap is 0.29%. So the
        # load-bearing comparison is INTERNAL: every K is scored against this tool's own K=0, which
        # carries the same offset. The published columns are context, not the baseline.
        repro = {}
        for m in labels:
            mine = _bpb_from(cells[by_label[m]], 0)
            repro[m] = {"mine_k0": mine, "published_zero_shot": published_zero[m],
                        "delta_frac": (mine - published_zero[m]) / published_zero[m]}

        rows = []
        for k in (0, 8, 16, 32, 64, 128, 256):
            scores = {m: _bpb_from(cells[by_label[m]], k) for m in labels}
            acc, _ = _accuracy(scores, crit, params, True)
            rows.append({"k": k, "accuracy": acc,
                         "gap_to_adapted": adapted_acc - acc,
                         "within_equivalence": abs(adapted_acc - acc) <= (
                             EQUIVALENCE_PAIRS / n_pairs if n_pairs else 0)})
        best = max(rows, key=lambda r: r["accuracy"])
        k0 = next(r for r in rows if r["k"] == 0)
        # The test only means something where adapted BEATS zero-shot, i.e. where there is a gap
        # for the cold start to be accused of causing. Under strict scoring at eleven models the
        # two readings already tie (0.727 vs 0.727), so "late-position matches adapted" is true at
        # K=0 -- with no cutoff applied at all -- and says nothing about cold starts. Scoring that
        # as a refutation is the same degenerate-condition error as reading a collapsed interval
        # as a significant one.
        gap_to_close = adapted_acc - zero_acc
        informative = gap_to_close > 1e-9
        closed = ((best["accuracy"] - zero_acc) / gap_to_close) if informative else None
        result[conv] = {
            "n_in_band_pairs": n_pairs,
            "published_adapted_accuracy": adapted_acc,
            "published_zero_shot_accuracy": zero_acc,
            "remeasured_k0_accuracy": k0["accuracy"],
            # If the re-measured cold reading does not reproduce the published zero-shot SELECTOR,
            # the K>0 rows are measuring something else and the verdict below is void.
            "k0_reproduces_published_selector": abs(k0["accuracy"] - zero_acc) <= 1e-9,
            "reproduction": repro,
            "max_abs_repro_delta_frac": max(abs(v["delta_frac"]) for v in repro.values()),
            "by_threshold": rows,
            "best_late_position": best,
            "gap_adapted_minus_zero_shot": gap_to_close,
            "test_is_informative": informative,
            "fraction_of_gap_closed": closed,
            "mechanism_overturned": bool(informative and best["within_equivalence"]),
        }

    bad_repro = [c for c in ("lenient", "strict")
                 if not result[c]["k0_reproduces_published_selector"]]
    over = [c for c in ("lenient", "strict") if result[c]["mechanism_overturned"]]
    uninformative = [c for c in ("lenient", "strict") if not result[c]["test_is_informative"]]
    result["uninformative_conventions"] = uninformative
    if bad_repro:
        result["verdict"] = ("VOID: re-measured K=0 does not reproduce the published zero-shot "
                             "selector under " + ", ".join(bad_repro) +
                             "; the late-position rows are not comparable")
    elif over:
        result["verdict"] = "MECHANISM OVERTURNED under " + ", ".join(over)
    else:
        note = ""
        if uninformative:
            note = ("; the test is uninformative under " + ", ".join(uninformative) +
                    " because adapted BPB does not beat the free reading there to begin with")
        result["verdict"] = ("mechanism survives: no late-position cutoff matches adapted BPB "
                             "where there is a gap to close" + note)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if the artifact is stale")
    args = ap.parse_args()
    built = build()
    if args.check:
        problems = []
        if not OUTPUT_JSON.exists():
            problems.append("results/late_position.json is missing")
        elif json.dumps(json.loads(OUTPUT_JSON.read_text()), sort_keys=True) != json.dumps(
                built, sort_keys=True):
            problems.append("results/late_position.json is stale")
        for p in problems:
            print("  " + p)
        print("late position: PROBLEMS", len(problems))
        sys.exit(1 if problems else 0)
    OUTPUT_JSON.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n")
    print("wrote", OUTPUT_JSON.relative_to(ROOT))
    print("VERDICT:", built["verdict"])
    for conv in ("lenient", "strict"):
        if conv in built:
            b = built[conv]
            print("  %-8s adapted %.3f | zero-shot %.3f | best late-position %.3f at K=%d"
                  % (conv, b["published_adapted_accuracy"], b["published_zero_shot_accuracy"],
                     b["best_late_position"]["accuracy"], b["best_late_position"]["k"]))


if __name__ == "__main__":
    main()

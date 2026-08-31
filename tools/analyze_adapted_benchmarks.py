#!/usr/bin/env python3
"""Experiment E2: what does adaptation do to task ACCURACY, not just to rank?

``paper_sota.tex`` Section 5.5 says domain-matched adaptation trades general
conditional-continuation calibration for task-relevant calibration. That sentence is currently an
inference from rank movement -- no adapted model was ever scored on a task benchmark, because the
published runs did not retain adapters. This script consumes the runs produced by
``tools/gcp_adapted_benchmark_run.sh`` and answers the question directly.

The distinction it resolves:

  PROBE       adaptation raises (or holds) absolute accuracy on the matched benchmark.
              Unsupervised domain adaptation improving task performance, predicted in advance by
              BPB, is a stronger result than the paper currently claims.

  STRESS TEST adaptation lowers accuracy for everyone while preserving the ordering. The corpus
              is not revealing a latent capability, it is degrading all models unevenly -- and
              Section 5.5's mechanism sentence has to be rewritten.

Reads the harness's own ``results_*.json`` so the adapted number is computed by exactly the code
and metric that produced the base-model numbers.

Base accuracies come from E2 phase 2 -- the released weights re-scored on the adapted run's OWN
host and engine -- whenever that cell exists, falling back to the committed July A100 numbers and
marking the row when it does not. The fallback is flagged rather than silently mixed because the
measured cross-host drift on GSM8K (-0.60 to +0.69 pp on 2026-08-28) is the same size as the
deltas this script reports.

    python tools/analyze_adapted_benchmarks.py --runs results/adapted_bench [--check]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_static_benchmarks import (  # noqa: E402
    BENCHMARKS, DISPLAY, MODEL_IDS, ROOT, STATIC_PATH, ranks,
)
from _artifact_check import _diff  # noqa: E402

DEFAULT_RUNS = ROOT / "results" / "adapted_bench"
OUTPUT_JSON = ROOT / "results" / "adapted_benchmark_analysis.json"
CORPORA = ["news", "math"]


def _metric_at(pattern: str, benchmark: str):
    """The metric the lm-eval harness itself recorded at ``pattern``, or None if absent."""
    spec = BENCHMARKS[benchmark]
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    payload = json.loads(Path(matches[-1]).read_text())
    task = payload.get("results", {}).get(spec["raw_task"], {})
    value = task.get(spec["raw_metric"])
    return None if value is None else float(value)


def harness_score(run_root: Path, corpus: str, benchmark: str, label: str):
    """Accuracy of the <corpus>-adapted model, or None if the cell is absent."""
    return _metric_at(str(run_root / corpus / benchmark / f"{label}__{corpus}-adapted" /
                          "*" / "results_*.json"), benchmark)


def base_score(run_root: Path, benchmark: str, label: str):
    """Accuracy of the RELEASED weights re-scored on the adapted run's own host (E2 phase 2).

    Preferred over the committed July numbers because a delta between the two would span a
    hardware and an engine change. Measured on 2026-08-28, that drift is +/-0.05 pp on HellaSwag
    (loglikelihood) but -0.60 to +0.69 pp on GSM8K (greedy generation) -- the same magnitude as
    several of the deltas E2 exists to measure. Returns None when phase 2 has no cell for the
    model, in which case the caller falls back and records the fallback.
    """
    return _metric_at(str(run_root / "base" / benchmark / label /
                          "*" / "results_*.json"), benchmark)


def rank_preservation(cells: dict) -> dict:
    """How strong is "the adapted ordering matched the base ordering" as a statistic?

    Reported conservatively on purpose. Under a null of random reordering, one cell of ``n`` models
    coming back in exactly the base order has probability ``1/n!``. The four cells here are NOT
    independent -- they share the same models, and GSM8K and MMLU-Pro are known in this study to
    agree on 51 of 55 pairs -- so multiplying the four probabilities would be wrong. We therefore
    quote the single-cell probability as the significance of the whole result: that is the value
    obtained by treating all four cells as one observation, which is the most conservative reading
    available and still resolves.
    """
    per_cell = {}
    for name, cell in cells.items():
        n = cell["n_aggregated"]
        per_cell[name] = {
            "n": n,
            "ranks_preserved": cell["ranks_preserved"],
            "fully_preserved": cell["ranks_preserved"] == n,
            "p_if_random": 1.0 / math.factorial(n) if n else float("nan"),
        }
    ns = {c["n"] for c in per_cell.values()}
    all_full = all(c["fully_preserved"] for c in per_cell.values())
    return {
        "cells_fully_preserved": sum(1 for c in per_cell.values() if c["fully_preserved"]),
        "n_cells": len(per_cell),
        "orderings_preserved": sum(c["ranks_preserved"] for c in per_cell.values()),
        "orderings_total": sum(c["n"] for c in per_cell.values()),
        "all_cells_fully_preserved": all_full,
        # Deliberately the single-cell value, not the product -- see the docstring.
        "p_conservative": (1.0 / math.factorial(max(ns))) if (all_full and len(ns) == 1)
                          else float("nan"),
        "per_cell": per_cell,
    }


def build(run_root: Path) -> dict:
    july = json.loads(STATIC_PATH.read_text())
    cells: dict[str, dict] = {}
    for corpus in CORPORA:
        for benchmark in BENCHMARKS:
            rows = {}
            for label in MODEL_IDS:
                adapted = harness_score(run_root, corpus, benchmark, label)
                if adapted is None:
                    continue
                before = base_score(run_root, benchmark, label)
                source = "same_host"
                if before is None:
                    before, source = float(july[label][benchmark]), "july_a100"
                rows[label] = {
                    "base_accuracy": before,
                    "base_source": source,
                    "adapted_accuracy": adapted,
                    "delta_pp": (adapted - before) * 100.0,
                }
            if rows:
                cells[f"{corpus}__{benchmark}"] = summarize(corpus, benchmark, rows)
    return {
        "schema_version": 3,
        "rank_preservation": rank_preservation(cells),
        "run_root": str(run_root.relative_to(ROOT)) if run_root.is_relative_to(ROOT)
                    else str(run_root),
        "base_accuracy_source": {
            "preferred": "<runs>/base/<benchmark>/<label>/ -- released weights re-scored on the "
                         "adapted run's own host and engine (E2 phase 2)",
            "fallback": "results/combined_bpb_vs_static.json -- the July A100 run; a delta "
                        "against it spans a hardware and engine change, so any model marked "
                        "july_a100 carries an unmatched delta",
        },
        "cohort_size": len(MODEL_IDS),
        "cells": cells,
        "verdict": verdict(cells),
    }


def summarize(corpus: str, benchmark: str, rows: dict) -> dict:
    """Aggregate the cell over its MATCHED rows only.

    A row whose base accuracy came from the July A100 run carries a delta that spans a hardware
    and engine change of the same magnitude as the effect, so folding it into the mean, the
    improved count or the rank comparison would reintroduce exactly the confound phase 2 removed.
    Unmatched rows stay in ``per_model`` (flagged by ``base_source``) but are excluded from every
    aggregate; ``aggregate_basis`` says which population the aggregates describe. If nothing is
    matched at all there is no matched population to report, so the cell falls back to all rows
    and says so rather than reporting nothing.
    """
    matched = {m: r for m, r in rows.items() if r["base_source"] == "same_host"}
    basis = "same_host" if matched else "july_a100_fallback"
    agg = matched or rows

    deltas = [r["delta_pp"] for r in agg.values()]
    base_rank = ranks({m: r["base_accuracy"] for m, r in agg.items()}, lower_is_better=False)
    adapted_rank = ranks({m: r["adapted_accuracy"] for m, r in agg.items()},
                         lower_is_better=False)
    preserved = sum(1 for m in agg if base_rank[m] == adapted_rank[m])
    return {
        "corpus": corpus,
        "benchmark": benchmark,
        "n_models": len(rows),
        "n_matched": len(matched),
        "n_aggregated": len(agg),
        "aggregate_basis": basis,
        "complete": len(rows) == len(MODEL_IDS),
        "mean_delta_pp": sum(deltas) / len(deltas),
        "min_delta_pp": min(deltas),
        "max_delta_pp": max(deltas),
        "n_improved": sum(1 for d in deltas if d > 0),
        "n_degraded": sum(1 for d in deltas if d < 0),
        "ranks_preserved": preserved,
        "per_model": rows,
    }


def verdict(cells: dict) -> dict:
    """State the probe/stress-test call only where the matched cell is actually complete."""
    key = "math__gsm8k"
    cell = cells.get(key)
    if cell is None or not cell["complete"]:
        have = 0 if cell is None else cell["n_models"]
        return {
            "resolved": False,
            "reason": f"math/GSM8K has {have}/{len(MODEL_IDS)} models; "
                      "the probe-vs-stress-test call needs the full cohort",
        }
    unmatched = cell["n_models"] - cell["n_matched"]
    if unmatched:
        return {
            "resolved": False,
            "reason": f"math/GSM8K has {unmatched} model(s) whose base accuracy still comes from "
                      "the July A100 run; those deltas span a hardware and engine change that is "
                      "the same size as the effect. Re-score them with "
                      "tools/gcp_base_benchmark_run.sh before calling probe vs stress test",
        }
    mean = cell["mean_delta_pp"]
    if mean > 0.5:
        call = ("PROBE: math-domain adaptation raises absolute GSM8K accuracy. Unsupervised "
                "domain adaptation improving task performance is a stronger claim than the "
                "paper currently makes, and Section 5.5 should be rewritten upward.")
    elif mean < -0.5:
        call = ("STRESS TEST: math-domain adaptation lowers absolute GSM8K accuracy. The corpus "
                "does not reveal a latent capability; rank agreement improves while accuracy "
                "falls. Section 5.5's 'trades calibration for calibration' sentence must be "
                "restated as degradation that is unevenly distributed across families.")
    else:
        call = ("NEUTRAL: adaptation leaves absolute GSM8K accuracy essentially unchanged while "
                "rank agreement improves. That is consistent with BPB becoming a better "
                "predictor without the adaptation itself changing task ability.")
    return {"resolved": True, "mean_delta_pp": mean, "call": call}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(DEFAULT_RUNS),
                    help="directory holding <corpus>/<benchmark>/<label>__<corpus>-adapted/")
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed analysis differs from a fresh computation")
    args = ap.parse_args()

    run_root = Path(args.runs)
    if not run_root.is_dir():
        raise SystemExit(
            f"{run_root} does not exist. Run tools/gcp_adapted_benchmark_run.sh on a GPU host "
            "and sync gs://gpu-llm-training-gceval/token_gain/adapted_bench/ into it first.")

    report = build(run_root)
    if not report["cells"]:
        raise SystemExit(f"no adapted benchmark results found under {run_root}")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not OUTPUT_JSON.exists():
            raise SystemExit(f"{OUTPUT_JSON.relative_to(ROOT)} is missing")
        drift = _diff(json.loads(OUTPUT_JSON.read_text()), report)
        if drift:
            raise SystemExit(f"{OUTPUT_JSON.relative_to(ROOT)} is stale: " + "; ".join(drift[:5]))
        print("adapted benchmark analysis is up to date")
        return

    OUTPUT_JSON.write_text(payload)
    for name, cell in sorted(report["cells"].items()):
        status = "" if cell["complete"] else f"  [PARTIAL {cell['n_models']}/{len(MODEL_IDS)}]"
        unmatched = cell["n_models"] - cell["n_matched"]
        if unmatched:
            status += f"  [{unmatched} UNMATCHED BASE]"
        print(f"\n=== {name}{status} ===")
        print(f"{'model':<20}{'base':>9}{'adapted':>10}{'delta pp':>11}")
        for label, row in sorted(cell["per_model"].items(),
                                 key=lambda kv: -kv[1]["delta_pp"]):
            # A trailing * means the base number is the July A100 one, so the delta is not
            # within-host and must not be quoted at the 1 pp scale the effect lives on.
            mark = "" if row["base_source"] == "same_host" else " *"
            print(f"{DISPLAY.get(label, label):<20}{row['base_accuracy']*100:>8.1f}%"
                  f"{row['adapted_accuracy']*100:>9.1f}%{row['delta_pp']:>+11.2f}{mark}")
        n = cell["n_aggregated"]
        basis = "" if cell["aggregate_basis"] == "same_host" else "  [UNMATCHED BASIS]"
        print(f"{'mean':<20}{'':>9}{'':>10}{cell['mean_delta_pp']:>+11.2f}"
              f"   over {n}/{cell['n_models']} matched,"
              f" improved {cell['n_improved']}/{n},"
              f" ranks preserved {cell['ranks_preserved']}/{n}{basis}")

    v = report["verdict"]
    print("\n=== VERDICT ===")
    print(v["call"] if v["resolved"] else f"UNRESOLVED: {v['reason']}")
    print(f"\nwrote {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E12: does adapted BPB predict a downstream task that is NOT prose overlap?

Written before the job returned. The lede fine-tune (E11) answered "the yardstick never leaves the
corpus", but both readers of the final draft raised the sharper version: its metric is n-gram overlap
on in-domain generation, which is close kin to what BPB measures, so agreement could be definitional.

This scores the same fifteen models on an in-domain multiple-choice task -- pick which paragraph
continues an article -- fine-tuned identically and scored by ACCURACY on a one-letter answer. No
overlap metric is involved and the output is a label, so a model cannot win by writing fluent prose.

**Preregistered outcomes.** Against the fine-tuned systems' accuracy, over in-band pairs:

* adapted BPB top and its margin over the benchmarks positive -> the selector survives a
  non-generative outcome;
* a benchmark top -> the paper's claim is specific to prose tasks and must say so;
* the task degenerate (every model at chance, or at ceiling, or the spread inside the noise the
  items themselves carry) -> UNINFORMATIVE, and nothing is claimed either way.

The degeneracy guard is not optional bookkeeping: a task everyone passes or everyone fails orders
models by noise, and would otherwise be read as a result. Same error class as a collapsed bootstrap
interval.

    python tools/analyze_mcq.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_downstream import BAND, EXCLUDED_MULTI_GPU, _selectors_17  # noqa: E402

CELLS = Path(os.environ.get("E12_DIR", str(Path(os.environ.get("SCRATCH", "/tmp")) / "e12_mcq_hard")))
OUTPUT_JSON = ROOT / "results" / "mcq.json"
SCHEMA_VERSION = 1
SEED = 20260912
PAIR_BOOT = 2_000
LETTERS = "ABCD"
CHANCE = 0.25
# Declared before the data. A task nobody can do, or everybody aces, orders models by noise.
CEILING = 0.95
MIN_SPREAD = 0.02


def parse_letter(text: str) -> str | None:
    for ch in text.strip():
        if ch.upper() in LETTERS:
            return ch.upper()
        if not ch.isspace() and ch not in ".:)-":
            return None
    return None


def score_cell(cell: dict) -> dict[str, int]:
    """Per-item correctness, keyed by item id, so pairs can be bootstrapped item-wise."""
    out = {}
    for g in cell["generations"]:
        out[g["id"]] = int(parse_letter(g["generated"]) == g["reference"].strip())
    return out


def build() -> dict:
    cohort = sorted(set(json.loads((ROOT / "results" / "criterion_uncertainty.json").read_text())
                        ["cohort_17"]) - set(EXCLUDED_MULTI_GPU))
    missing = [m for m in cohort if not (CELLS / f"mcq__{m}.json").exists()]
    result: dict = {"schema_version": SCHEMA_VERSION, "cohort": cohort,
                    "excluded_multi_gpu": list(EXCLUDED_MULTI_GPU), "models_missing": missing,
                    "chance": CHANCE,
                    "preregistered": {"metric": "accuracy on a one-letter answer",
                                      "degenerate_if": f"all at chance, any >{CEILING}, or spread "
                                                       f"<{MIN_SPREAD}"}}
    if missing:
        # The sweep was never run: pilot 2 tripped the ceiling guard. Report the pilot rather than
        # leaving a permanently "INCOMPLETE" artifact that explains nothing.
        have = [m for m in cohort if m not in missing]
        if have:
            acc = {m: (lambda v: sum(v.values()) / len(v))(
                score_cell(json.loads((CELLS / f"mcq__{m}.json").read_text()))) for m in have}
            result["pilot_accuracy"] = acc
            result["models_scored"] = have
            top = max(acc.values())
            result["verdict"] = (
                "UNINFORMATIVE (pilot): %s reaches %.3f, above the %.2f ceiling declared before the "
                "data; four-way discrimination saturates, so the sweep was not run"
                % (max(acc, key=acc.get), top, CEILING)) if top > CEILING else "INCOMPLETE"
        else:
            result["verdict"] = "INCOMPLETE"
        return result

    per_item = {m: score_cell(json.loads((CELLS / f"mcq__{m}.json").read_text())) for m in cohort}
    acc = {m: sum(v.values()) / len(v) for m, v in per_item.items()}
    result["accuracy"] = acc
    result["n_eval"] = {m: len(v) for m, v in per_item.items()}
    spread = max(acc.values()) - min(acc.values())
    result["spread"] = spread
    result["at_ceiling"] = [m for m in acc if acc[m] > CEILING]
    if result["at_ceiling"] or spread < MIN_SPREAD or max(acc.values()) <= CHANCE:
        result["verdict"] = ("UNINFORMATIVE: the task does not separate these models "
                             f"(spread {spread:.3f}, max {max(acc.values()):.3f})")
        return result

    from analyze_cloze_validity import _bootstrap_tables
    selectors, params = _selectors_17()
    result["selector_table"] = _bootstrap_tables(selectors, acc, cohort, params, BAND,
                                                 compared="matched_adapted_bpb")

    rng = random.Random(SEED)
    pairs = []
    for a, b in combinations(cohort, 2):
        if max(params[a], params[b]) / min(params[a], params[b]) > BAND:
            continue
        ids = sorted(set(per_item[a]) & set(per_item[b]))
        diffs = [per_item[a][i] - per_item[b][i] for i in ids]
        n = len(diffs)
        means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(PAIR_BOOT))
        lo, hi = means[int(0.025 * PAIR_BOOT)], means[int(0.975 * PAIR_BOOT) - 1]
        winner = a if acc[a] > acc[b] else b
        row = {"pair": f"{a}|{b}", "winner": winner, "diff_a_minus_b": sum(diffs) / n,
               "ci_lo": lo, "ci_hi": hi, "resolved": lo > 0 or hi < 0}
        for name, (vals, lower) in selectors.items():
            if a in vals and b in vals and vals[a] != vals[b]:
                pick = (a if vals[a] < vals[b] else b) if lower else (a if vals[a] > vals[b] else b)
                row[f"{name}_right"] = pick == winner
        pairs.append(row)
    resolved = [p for p in pairs if p["resolved"]]
    result["pairs"] = pairs
    result["n_in_band_pairs"] = len(pairs)
    result["n_resolved"] = len(resolved)
    result["accuracy_on_resolved"] = {
        name: (sum(p[f"{name}_right"] for p in resolved if f"{name}_right" in p)
               / max(1, sum(1 for p in resolved if f"{name}_right" in p)))
        for name in selectors}
    table = result["selector_table"]["selectors"]
    top = max(table, key=lambda k: table[k]["pairwise_accuracy"])
    result["top_selector"] = top
    result["verdict"] = ("adapted BPB predicts a non-generative outcome too"
                         if top == "matched_adapted_bpb"
                         else f"a benchmark predicts this outcome better: {top}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    built = build()
    if args.check:
        problems = []
        if not OUTPUT_JSON.exists():
            problems.append("results/mcq.json is missing")
        elif json.dumps(json.loads(OUTPUT_JSON.read_text()), sort_keys=True) != json.dumps(
                built, sort_keys=True):
            problems.append("results/mcq.json is stale")
        for p in problems:
            print("  " + p)
        print("mcq: PROBLEMS", len(problems))
        sys.exit(1 if problems else 0)
    OUTPUT_JSON.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n")
    print("wrote", OUTPUT_JSON.relative_to(ROOT))
    print("VERDICT:", built["verdict"])
    if "selector_table" in built:
        t = built["selector_table"]["selectors"]
        for n, r in sorted(t.items(), key=lambda kv: -kv[1]["pairwise_accuracy"]):
            print("  %-20s %.3f [%.3f, %.3f]" % (n, r["pairwise_accuracy"], r["ci_lo"], r["ci_hi"]))


if __name__ == "__main__":
    main()

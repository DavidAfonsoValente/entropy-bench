#!/usr/bin/env python
"""Refresh results/combined_bpb_vs_static.json from the staged benchmark column.

`analyze_static_benchmarks.py` cross-checks every committed accuracy against the raw harness
log and refuses to run if they disagree by more than half a rounding unit. That guard is what
caught the E9 replacement (it flagged gemma-4-31B mmlu_pro 0.578 vs 0.571), and it is doing
its job: the committed table is a second, independently-editable copy of numbers that must
agree with the logs.

After a whole-column replacement the correct move is to regenerate the table from the logs
rather than hand-edit eleven rows times three benchmarks. Non-benchmark fields in each row
(e.g. the superseded adapted-BPB marker) are preserved untouched.

    python tools/refresh_static_table.py [--check]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from analyze_static_benchmarks import BENCHMARKS, MODEL_IDS, load_raw_static  # noqa: E402

PATH = ROOT / "results" / "combined_bpb_vs_static.json"
# Match the rounding already in the file, so the guard's half-unit tolerance holds.
PLACES = {"mmlu_pro": 3, "hellaswag": 4, "gsm8k": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    table = json.loads(PATH.read_text())
    if set(table) != set(MODEL_IDS):
        sys.exit(f"FATAL: table covers {len(table)} models, cohort has {len(MODEL_IDS)}")

    changes, problems = [], []
    for model in sorted(table):
        for task in BENCHMARKS:
            exact, _ = load_raw_static(model, task)
            new = round(exact, PLACES[task])
            old = table[model][task]
            if new != old:
                changes.append(f"  {model:18s} {task:10s} {old} -> {new}")
                if a.check:
                    problems.append(f"{model}/{task}: table {old}, column {new}")
                else:
                    table[model][task] = new

    if a.check:
        if problems:
            for p in problems:
                print(f"FATAL: {p}", file=sys.stderr)
            return 1
        print(f"committed table agrees with the staged column ({len(table)} models)")
        return 0

    if not changes:
        print("no change needed")
        return 0
    print(f"refreshing {len(changes)} cells:")
    print("\n".join(changes))
    PATH.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(f"wrote {PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

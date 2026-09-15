#!/usr/bin/env python
"""Derive the extension cohort's benchmark cells from the staged E9 column.

`analyze_cohort_extension.py` reads the six extension models' benchmark scores from
`results/cohort_ext/bench/<benchmark>__<label>.json`, while the eleven published models come
straight from the raw harness logs via `load_raw_static`. Before E9 those two halves came
from different hosts and the extension half was mostly missing, which is why Q2 sat at
PENDING.

E9 re-scored **all seventeen** models on one host and one build, so both halves now come
from the same column. This writes the extension half from that column rather than leaving
it to be filled in by hand, and records per cell exactly which artifact each score came
from.

    python tools/sync_cohort_ext_bench.py [--check]

`--check` verifies the derived files agree with the staged column and writes nothing.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from analyze_static_benchmarks import BENCHMARKS, MODEL_IDS  # noqa: E402

DEST = ROOT / "results" / "cohort_ext" / "bench"
# analyze_cohort_extension.py keys benchmarks by raw_dir name, not by the BENCHMARKS key
# ("mmlu_pro_1k", not "mmlu_pro"), and the filenames follow that.
BY_DIR = {meta["raw_dir"]: meta for meta in BENCHMARKS.values()}


def staged_score(label: str, raw_dir: str):
    meta = BY_DIR[raw_dir]
    hits = sorted((ROOT / "results" / raw_dir / label).glob("*/results_*.json"))
    if not hits:
        return None, None
    d = json.loads(hits[-1].read_text())
    return d["results"][meta["raw_task"]][meta["raw_metric"]], hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    published = set(MODEL_IDS)
    labels = sorted({p.name for d in BY_DIR for p in (ROOT / "results" / d).glob("*")
                     if p.is_dir()} - published)
    if not labels:
        sys.exit("FATAL: no extension-cohort labels found in the staged column")

    problems, written = [], 0
    for label in labels:
        for raw_dir in BY_DIR:
            score, src = staged_score(label, raw_dir)
            if score is None:
                problems.append(f"{label}:{raw_dir} absent from the staged column")
                continue
            out = DEST / f"{raw_dir}__{label}.json"
            payload = {
                "label": label,
                "benchmark": raw_dir,
                "score": score,
                "metric": BY_DIR[raw_dir]["raw_metric"],
                "n": BY_DIR[raw_dir]["n"],
                "source": str(src.relative_to(ROOT)),
                "provenance": "E9 re-score: all 17 models, one host (A100-SXM-64GB), one build, "
                              "uniform decode protocol (eval/lm_eval_uniform.py)",
            }
            if a.check:
                if not out.exists():
                    problems.append(f"{out.relative_to(ROOT)} missing")
                elif json.loads(out.read_text()).get("score") != score:
                    problems.append(f"{out.relative_to(ROOT)} stale: "
                                    f"{json.loads(out.read_text()).get('score')} != {score}")
            else:
                DEST.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                written += 1

    if problems:
        for p in problems:
            print(f"FATAL: {p}", file=sys.stderr)
        return 1
    print(f"{'checked' if a.check else 'wrote'} {len(labels) * len(BY_DIR)} extension bench cells "
          f"for {len(labels)} models: {', '.join(labels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

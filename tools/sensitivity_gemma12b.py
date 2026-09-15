#!/usr/bin/env python
"""Is the Q2 answer an artifact of gemma-4-12B's two broken generative cells?

gemma-4-12B (`Gemma4UnifiedForConditionalGeneration`) reproduces July on hellaswag
(loglikelihood, -0.27 pp) but not on the two generative benchmarks (gsm8k -9.86 pp,
mmlu_pro_1k -14.30 pp; mechanism unidentified, see docs/RUN_LEDGER.md). The preregistered
agreement gate covers gsm8k and hellaswag only, so the mmlu error passes it untouched --
which is exactly why this runs.

The counterfactual tested is the sharp one: **what if that cell were right?** July's own
vLLM numbers for those two cells are substituted, the whole analysis is re-run, and the two
answers are compared. Substituting beats deleting here, because it asks whether the
conclusion depends on the defect rather than merely whether it survives a smaller cohort.

Writes results/sensitivity_gemma12b.json. Restores the staged column in a finally block --
it must never leave a July number sitting in the new column.

    python tools/sensitivity_gemma12b.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = "gemma-4-12B"
# July's vLLM values for the two cells the hf backend gets wrong, from the superseded column.
JULY = {"gsm8k": ("gsm8k", "exact_match,flexible-extract"),
        "mmlu_pro_1k": ("mmlu_pro_1k", "exact_match,custom-extract")}


def staged(raw_dir):
    hits = sorted((ROOT / "results" / raw_dir / LABEL).glob("*/results_*.json"))
    return hits[-1] if hits else None


def july_value(raw_dir, task, metric):
    hits = sorted((ROOT / "results" / "legacy_july_bench" / raw_dir / LABEL)
                  .glob("*/results_*.json"))
    if not hits:
        sys.exit(f"FATAL: no superseded July cell for {LABEL}/{raw_dir}")
    return json.loads(hits[-1].read_text())["results"][task][metric]


def read_answer():
    d = json.loads((ROOT / "results" / "cohort_extension.json").read_text())
    out = {}
    for conv in ("lenient", "strict"):
        block = d.get(conv) or {}
        sels = block.get("selectors") or block
        out[conv] = {k: (v.get("accuracy") if isinstance(v, dict) else v)
                     for k, v in sels.items() if isinstance(v, (dict, float, int))}
    out["q2"] = d.get("q2_any_benchmark_clears_chance")
    return out


def main():
    backups = {}
    try:
        baseline = read_answer()
        subs = {}
        for raw_dir, (task, metric) in JULY.items():
            path = staged(raw_dir)
            if path is None:
                sys.exit(f"FATAL: {LABEL}/{raw_dir} not in the staged column")
            backups[path] = path.read_text()
            d = json.loads(backups[path])
            before = d["results"][task][metric]
            after = july_value(raw_dir, task, metric)
            d["results"][task][metric] = after
            path.write_text(json.dumps(d))
            subs[raw_dir] = {"hf_backend": before, "july_vllm": after,
                             "delta_pp": 100 * (before - after)}
            print(f"  substituted {raw_dir}: {before:.4f} -> {after:.4f} (July)")

        r = subprocess.run([sys.executable, "tools/sync_cohort_ext_bench.py"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"FATAL: sync failed: {r.stderr}")
        r = subprocess.run([sys.executable, "tools/analyze_cohort_extension.py"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"FATAL: analysis failed: {r.stderr}")
        print(r.stdout)
        counterfactual = read_answer()
    finally:
        for path, text in backups.items():
            path.write_text(text)
        # Restoring the cells is not enough: the counterfactual run OVERWROTE
        # results/cohort_extension.json, and leaving that artifact holding substituted July
        # numbers while the column holds hf ones is precisely the kind of silent mismatch
        # this project keeps getting caught by. Regenerate it from the restored column.
        for tool in ("tools/sync_cohort_ext_bench.py", "tools/analyze_cohort_extension.py"):
            rr = subprocess.run([sys.executable, tool], cwd=ROOT, capture_output=True, text=True)
            if rr.returncode:
                print(f"CRITICAL: could not regenerate via {tool}: {rr.stderr}", file=sys.stderr)
        print(f"restored {len(backups)} staged cells and regenerated cohort_extension.json")

    out = {
        "question": "Does the Q2 answer depend on gemma-4-12B's two broken generative cells?",
        "method": "Substitute July's vLLM values for gemma-4-12B gsm8k and mmlu_pro_1k, "
                  "re-run the full cohort-extension analysis, compare. The staged column is "
                  "restored afterwards.",
        "substitutions": subs,
        "baseline_hf_column": baseline,
        "counterfactual_july_gemma_cells": counterfactual,
    }
    (ROOT / "results" / "sensitivity_gemma12b.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote results/sensitivity_gemma12b.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

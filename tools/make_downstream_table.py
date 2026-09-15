#!/usr/bin/env python3
"""Render the downstream fine-tune (E11) as LaTeX rows.

One row per selector: pairwise accuracy against the fine-tuned systems' ROUGE-L over every in-band
pair of the fifteen fine-tuned models (E11b), with the model-level cluster-bootstrap interval, and
accuracy on the subset of pairs the fine-tune itself resolves.

Reads ``results/downstream.json``. Nothing is recomputed here.

    python tools/make_downstream_table.py            # -> downstream_table.tex
    python tools/make_downstream_table.py --check    # verify the committed .tex is current
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "downstream_table.tex"
SRC = ROOT / "results" / "downstream.json"

LABELS = {
    "matched_adapted_bpb": "Adapted BPB", "hellaswag": "HellaSwag",
    "zero_shot_bpb": "Zero-shot BPB", "parameter_count": "Parameter count",
    "mmlu_pro": "MMLU-Pro", "gsm8k": "GSM8K",
}


def render() -> str:
    """One row per selector: how well it orders the FINE-TUNED systems, over every in-band pair and
    over the pairs the fine-tune itself resolves. Sorted by accuracy so the table reads top-down."""
    data = json.loads(SRC.read_text())
    c = data["cohort"]
    b = data["cohort_recipe_b"]["selector_table"]["selectors"]
    table = c["selector_table"]["selectors"]
    resolved = c["accuracy_on_resolved"]
    top = max(r["pairwise_accuracy"] for r in table.values())
    lines = []
    for name, row in sorted(table.items(), key=lambda kv: -kv[1]["pairwise_accuracy"]):
        acc = "%.3f" % row["pairwise_accuracy"]
        if row["pairwise_accuracy"] == top:
            acc = "\\textbf{%s}" % acc
        lines.append("%s & %s & [%.3f, %.3f] & %.3f & %.3f \\\\" % (
            LABELS[name], acc, row["ci_lo"], row["ci_hi"], resolved[name],
            b[name]["pairwise_accuracy"]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    body = render()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != body:
            print("downstream table: STALE -- regenerate with tools/make_downstream_table.py")
            sys.exit(1)
        print("downstream table: PROBLEMS 0")
    else:
        OUT.write_text(body)
        print("wrote", OUT.name)

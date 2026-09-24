#!/usr/bin/env python3
"""Emit the 17-model extended-cohort roster as LaTeX rows: every model with its parameter count,
adapted news BPB and whether it belongs to the original eleven or the preregistered extension.

Reads ``results/cohort_extension.json`` (which models) and the domain-transfer and extension cells
(adapted BPB and parameter counts). Nothing is recomputed here.

Usage:
    python tools/make_cohort_table.py            # -> cohort_table.tex
    python tools/make_cohort_table.py --check    # verify the committed .tex is current
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_decision_rule import _adapted_bpb_17  # noqa: E402  (one loader, not two)

OUT = ROOT / "cohort_table.tex"

DISPLAY = {
    "gemma-4-31B": "Gemma-4-31B", "gemma-4-12B": "Gemma-4-12B",
    "Qwen3.5-35B-MoE": "Qwen-3.5-35B-MoE$^{*}$", "Ministral-3-14B": "Ministral-3-14B",
    "Qwen2.5-7B": "Qwen-2.5-7B", "Qwen3.5-9B": "Qwen-3.5-9B", "Llama-3.2-1B": "Llama-3.2-1B",
    "Qwen3.5-4B": "Qwen-3.5-4B", "Qwen2.5-1.5B": "Qwen-2.5-1.5B",
    "LFM2.5-1.2B": "LiquidAI-LFM2.5", "Qwen2.5-0.5B": "Qwen-2.5-0.5B",
    "Qwen2.5-3B": "Qwen-2.5-3B", "Qwen2.5-14B": "Qwen-2.5-14B", "Falcon3-7B": "Falcon3-7B",
    "Falcon3-10B": "Falcon3-10B", "Mistral-Nemo-12B": "Mistral-Nemo-12B",
    "SmolLM2-1.7B": "SmolLM2-1.7B",
}


def render() -> str:
    bpb, params = _adapted_bpb_17()
    ext = json.loads((ROOT / "results" / "cohort_extension.json").read_text())
    published = set(ext["published_cohort"])
    cohort = published | set(ext["extension_cohort"])  # the loader also returns excluded cells
    assert len(cohort) == ext["n_models"] == 17, cohort
    lines = []
    for m in sorted(cohort, key=bpb.get):
        origin = "original" if m in published else "\\textbf{added}"
        lines.append("%s & %.1f & %.3f & %s \\\\" % (DISPLAY[m], params[m] / 1e9, bpb[m], origin))
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    body = render()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != body:
            print("cohort table: STALE -- regenerate with tools/make_cohort_table.py")
            sys.exit(1)
        print("cohort table: PROBLEMS 0")
    else:
        OUT.write_text(body)
        print("wrote", OUT.name)

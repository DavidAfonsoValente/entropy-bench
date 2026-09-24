#!/usr/bin/env python3
"""Emit Table 1: the eleven-model cohort on news, ranked every way the paper compares.

One row per model, ordered by adapted news BPB: parameters, unadapted BPB and its rank, adapted BPB
and its rank, how many places adaptation moved it, the loss reduction adaptation bought, and the
model's score and rank on HellaSwag, GSM8K and MMLU-Pro. It is the table behind Section 3.1 (which
models adaptation moves, and why) and Section 3.2 (the adapted order converging on HellaSwag).

Reads results/alignment_matrix.json (both BPB readings), results/combined_bpb_vs_static.json
(benchmark scores) and the cohort loader shared with make_cohort_table.py (parameter counts).

Usage:
    python tools/make_rank_table.py            # -> rank_table.tex
    python tools/make_rank_table.py --check    # verify the committed .tex is current
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_decision_rule import _adapted_bpb_17  # noqa: E402  (one loader, not two)
from make_cohort_table import DISPLAY  # noqa: E402

OUT = ROOT / "rank_table.tex"
BENCHMARKS = ("hellaswag", "gsm8k", "mmlu_pro")


def _rank(vals: dict[str, float], lower: bool) -> dict[str, int]:
    order = sorted(vals, key=lambda m: vals[m] if lower else -vals[m])
    return {m: i + 1 for i, m in enumerate(order)}


def render() -> str:
    cells = json.loads((ROOT / "results" / "alignment_matrix.json").read_text())["cells"]
    zero, adapted = cells["news__zero_shot"]["bpb"], cells["news__adapted"]["bpb"]
    static = json.loads((ROOT / "results" / "combined_bpb_vs_static.json").read_text())
    _, params = _adapted_bpb_17()
    rz, ra = _rank(zero, True), _rank(adapted, True)
    bench = {b: {m: static[m][b] for m in zero} for b in BENCHMARKS}
    rb = {b: _rank(bench[b], False) for b in BENCHMARKS}
    lines = []
    for m in sorted(adapted, key=adapted.get):
        move = rz[m] - ra[m]
        moved = ("$\\uparrow$%d" % move) if move > 0 else ("$\\downarrow$%d" % -move) if move < 0 else "--"
        if move >= 3:
            moved = "\\textbf{%s}" % moved
        gain = 100.0 * (1.0 - adapted[m] / zero[m])
        gain_s = ("\\textbf{%.1f\\%%}" % gain) if gain >= 10 else ("%.1f\\%%" % gain)
        scores = " & ".join("%.1f (%d)" % (100 * bench[b][m], rb[b][m]) for b in BENCHMARKS)
        lines.append("%s & %.1f & %.3f (%d) & %.3f (%d) & %s & %s & %s \\\\" % (
            DISPLAY[m], params[m] / 1e9, zero[m], rz[m], adapted[m], ra[m], moved, gain_s, scores))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    body = render()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != body:
            print("rank table: STALE -- regenerate with tools/make_rank_table.py")
            sys.exit(1)
        print("rank table: PROBLEMS 0")
    else:
        OUT.write_text(body)
        print("wrote", OUT.name)

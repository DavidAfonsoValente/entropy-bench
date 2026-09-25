#!/usr/bin/env python3
"""Emit the cross-corpus agreement table: how closely the eleven models' BPB on one corpus matches
their BPB on another, unadapted and adapted, as rank (Spearman) and value (Pearson) correlations.

The three general-text corpora agree almost perfectly under both readings, so the ordering is a
property of the models rather than of one corpus. The mathematics corpus is the informative row:
adaptation makes it agree *less* with the general corpora, because adapted on mathematics the
ranking follows a different capability (Section 3.2).

Reads results/domain_transfer/<corpus>__*.json. Nothing else.

Usage:
    python tools/make_crosscorpus_table.py            # -> crosscorpus_table.tex
    python tools/make_crosscorpus_table.py --check    # verify the committed .tex is current
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "crosscorpus_table.tex"
CORPORA = [("news", "News"), ("reddit", "Reddit"), ("hackernews", "Hacker News"), ("math", "arXiv maths")]


def load() -> dict[str, dict[str, tuple[float, float]]]:
    cells: dict[str, dict[str, tuple[float, float]]] = {}
    for corpus, _ in CORPORA:
        cells[corpus] = {}
        for path in sorted((ROOT / "results" / "domain_transfer").glob(f"{corpus}__*.json")):
            run = json.loads(path.read_text())
            cells[corpus][run["model_id"]] = (run["zero_shot_bpb"], run["adapted_bpb"])
    return cells


def render() -> str:
    cells = load()
    label = dict(CORPORA)
    lines = []
    for a, b in itertools.combinations([c for c, _ in CORPORA], 2):
        models = sorted(set(cells[a]) & set(cells[b]))
        assert len(models) == 11, (a, b, len(models))
        stats = []
        for reading in (0, 1):
            x = [cells[a][m][reading] for m in models]
            y = [cells[b][m][reading] for m in models]
            stats.append((spearmanr(x, y)[0], pearsonr(x, y)[0]))
        (rs_z, rp_z), (rs_a, rp_a) = stats
        lines.append("%s--%s & %.3f & %.3f & %.3f & %.3f \\\\" % (label[a], label[b], rs_z, rs_a, rp_z, rp_a))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    body = render()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != body:
            print("cross-corpus table: STALE -- regenerate with tools/make_crosscorpus_table.py")
            sys.exit(1)
        print("cross-corpus table: PROBLEMS 0")
    else:
        OUT.write_text(body)
        print("wrote", OUT.name)

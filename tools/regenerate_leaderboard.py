#!/usr/bin/env python3
"""Regenerate each track's public leaderboard from the committed fixed-recipe result cells.

Tracks share the eleven-model cohort and the fixed adaptation recipe, and differ in corpus:
news (controlled corpus, maintainer-run) and arXiv mathematics (public corpus, anyone can run it).
Model display names and parameter counts are carried over from the news board, which is the
cohort's registry.

    python tools/regenerate_leaderboard.py           # rewrite results/leaderboard*.json
    python tools/regenerate_leaderboard.py --check   # fail if any board is stale
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "domain_transfer"
FIXED_LORA = {"r": 16, "alpha": 32, "dropout": 0.05, "lr": 0.0001, "effective_batch": 32}
PROTOCOL = {
    "adaptation": "fixed LoRA: r=16, alpha=32, dropout=0.05, learning rate=1e-4, effective batch=32, 250 update steps",
    "evaluation": "held-out corpus-wide BPB over independent 512-token blocks; injected document-start marker targets excluded",
}
TRACKS = [
    {
        "board": ROOT / "results" / "leaderboard.json",
        "prefix": "news",
        "dataset": {
            "benchmark_id": "primary-news-2026-06-08-fixed-lora-v1",
            "manifest": "benchmarks/primary-news-2026-06-08-fixed-lora-v1.json",
            "name": "Google News",
            "snapshot_date": "2026-06-08",
            "articles": 119054,
            "split": "80/10/10",
            "seed": 42,
        },
        "contamination": "risk-screened global-unified held-out set",
    },
    {
        "board": ROOT / "results" / "leaderboard_math.json",
        "prefix": "math",
        "dataset": {
            "benchmark_id": "arxiv-math-2026-08-fixed-lora-v1",
            "manifest": "benchmarks/arxiv-math-2026-08-fixed-lora-v1.json",
            "name": "arXiv math.* titles and abstracts",
            "snapshot_date": "2026-06-09 to 2026-08-20",
            "articles": 3207,
            "split": "80/10/10",
            "seed": 42,
        },
        "contamination": "post-release window plus a GSM8K-test shingle screen",
    },
]


def build(track: dict, metadata: dict[str, tuple[str, str]]) -> dict:
    runs = []
    for path in sorted(RUNS.glob(f"{track['prefix']}__*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if not run.get("mask_injected_special_tokens"):
            raise ValueError(f"{path} did not mask injected special-token targets")
        if run.get("train_steps") != 250 or run.get("lora") != FIXED_LORA:
            raise ValueError(f"{path} does not use the fixed headline protocol")
        runs.append((path, run))
    if set(metadata) != {run["model_id"] for _, run in runs}:
        raise ValueError(f"{track['prefix']} cells do not cover the public 11-model cohort")
    entries = []
    for rank, (path, run) in enumerate(sorted(runs, key=lambda item: item[1]["adapted_bpb"]), start=1):
        model_name, parameters = metadata[run["model_id"]]
        entries.append({
            "rank": rank,
            "model_id": run["model_id"],
            "model_name": model_name,
            "parameters": parameters,
            "zero_shot_bpb": run["zero_shot_bpb"],
            "adapted_bpb": run["adapted_bpb"],
            "reduction_pct": run["reduction_pct"],
            "evidence": str(path.relative_to(ROOT)),
        })
    return {
        "schema_version": 1,
        "metric": "adapted_bits_per_byte",
        "direction": "lower_is_better",
        "dataset": track["dataset"],
        "protocol": {**PROTOCOL, "contamination": track["contamination"]},
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail instead of rewriting stale output")
    args = parser.parse_args()
    registry = json.loads(TRACKS[0]["board"].read_text(encoding="utf-8"))
    metadata = {e["model_id"]: (e["model_name"], e["parameters"]) for e in registry["entries"]}
    for track in TRACKS:
        rendered = json.dumps(build(track, metadata), indent=2) + "\n"
        board = track["board"]
        if args.check:
            if not board.exists() or board.read_text(encoding="utf-8") != rendered:
                raise SystemExit(f"{board.relative_to(ROOT)} is stale; run tools/regenerate_leaderboard.py")
            print(f"leaderboard artifact: current ({board.name})")
        else:
            board.write_text(rendered, encoding="utf-8")
            print(f"wrote {board.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

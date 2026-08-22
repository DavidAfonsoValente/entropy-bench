#!/usr/bin/env python3
"""Regenerate the public leaderboard from the corrected news-domain rerun."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "results" / "leaderboard.json"
RUNS = ROOT / "results" / "domain_transfer"
MANIFEST = "benchmarks/primary-news-2026-06-08-fixed-lora-v1.json"
BENCHMARK_ID = "primary-news-2026-06-08-fixed-lora-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail instead of rewriting stale output")
    args = parser.parse_args()
    previous = json.loads(BOARD.read_text(encoding="utf-8"))
    metadata = {
        entry["model_id"]: (entry["model_name"], entry["parameters"])
        for entry in previous["entries"]
    }
    runs = []
    for path in sorted(RUNS.glob("news__*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if not run.get("mask_injected_special_tokens"):
            raise ValueError(f"{path} did not mask injected special-token targets")
        if run.get("train_steps") != 250 or run.get("lora") != {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "lr": 0.0001,
            "effective_batch": 32,
        }:
            raise ValueError(f"{path} does not use the fixed headline protocol")
        runs.append((path, run))
    if set(metadata) != {run["model_id"] for _, run in runs}:
        raise ValueError("corrected rerun does not cover the public 11-model cohort")

    entries = []
    for rank, (path, run) in enumerate(
        sorted(runs, key=lambda item: item[1]["adapted_bpb"]), start=1
    ):
        model_id = run["model_id"]
        model_name, parameters = metadata[model_id]
        entries.append(
            {
                "rank": rank,
                "model_id": model_id,
                "model_name": model_name,
                "parameters": parameters,
                "zero_shot_bpb": run["zero_shot_bpb"],
                "adapted_bpb": run["adapted_bpb"],
                "reduction_pct": run["reduction_pct"],
                "evidence": str(path.relative_to(ROOT)),
            }
        )

    board = {
        "schema_version": 1,
        "metric": "adapted_bits_per_byte",
        "direction": "lower_is_better",
        "dataset": {
            "benchmark_id": BENCHMARK_ID,
            "manifest": MANIFEST,
            "name": "Google News",
            "snapshot_date": "2026-06-08",
            "articles": 119054,
            "split": "80/10/10",
            "seed": 42,
        },
        "protocol": {
            "adaptation": "fixed LoRA: r=16, alpha=32, dropout=0.05, learning rate=1e-4, effective batch=32, 250 update steps",
            "evaluation": "held-out corpus-wide BPB over independent 512-token blocks; injected document-start marker targets excluded",
            "contamination": "risk-screened global-unified held-out set",
        },
        "entries": entries,
    }
    rendered = json.dumps(board, indent=2) + "\n"
    if args.check:
        if BOARD.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results/leaderboard.json is stale; run tools/regenerate_leaderboard.py")
        print("leaderboard artifact: current")
    else:
        BOARD.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

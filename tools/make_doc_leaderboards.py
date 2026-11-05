#!/usr/bin/env python3
"""Build the two document-reading leaderboards (news and arXiv maths) from the committed cells.

Every model with a ``{corpus}_doc__*.json`` cell in results/doc_protocol_cells is ranked by its
unadapted whole-document BPB. Display names and parameter
strings come from the fixed-LoRA boards where a model is on them, else from NAMES below.

    python tools/make_doc_leaderboards.py           # writes results/leaderboard_{news,math}_documents.json
    python tools/make_doc_leaderboards.py --check   # fail if either board is stale
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "results" / "doc_protocol_cells"
TRACKS = {  # corpus: (benchmark id, board, fixed-LoRA board the dataset block is copied from)
    "news": ("primary-news-2026-06-08-documents-v1", "leaderboard_news_documents.json", "leaderboard.json"),
    "math": ("arxiv-math-2026-08-documents-v1", "leaderboard_math_documents.json", "leaderboard_math.json"),
}
# The six models that are not on the fixed-LoRA boards.
NAMES = {"Qwen/Qwen2.5-14B": ("Qwen-2.5-14B", "14.8B"), "Qwen/Qwen2.5-3B": ("Qwen-2.5-3B", "3.1B"),
         "tiiuae/Falcon3-10B-Base": ("Falcon3-10B", "10.4B"), "tiiuae/Falcon3-7B-Base": ("Falcon3-7B", "7.5B"),
         "mistralai/Mistral-Nemo-Base-2407": ("Mistral-Nemo-12B", "12.2B"),
         "HuggingFaceTB/SmolLM2-1.7B": ("SmolLM2-1.7B", "1.7B")}


def build(corpus: str) -> dict:
    benchmark_id, _, fixed = TRACKS[corpus]
    old = json.loads((ROOT / "results" / fixed).read_text())
    names = {e["model_id"]: (e["model_name"], e["parameters"]) for e in old["entries"]} | NAMES
    entries = []
    for path in sorted(CELLS.glob(f"{corpus}_doc__*.json")):
        cell = json.loads(path.read_text())
        name, params = names[cell["model_id"]]
        entries.append({"rank": 0, "model_id": cell["model_id"], "model_name": name, "parameters": params,
                        "doc_bpb": cell["doc_protocol"]["zero_shot"]["bpb"],
                        "evidence": str(path.relative_to(ROOT))})
    entries.sort(key=lambda e: e["doc_bpb"])
    for rank, e in enumerate(entries, start=1):
        e["rank"] = rank
    manifest = json.loads((ROOT / "benchmarks" / f"{benchmark_id}.json").read_text())
    return {"schema_version": 1, "metric": "document_bits_per_byte", "direction": "lower_is_better",
            "dataset": {**old["dataset"], "benchmark_id": benchmark_id,
                        "manifest": f"benchmarks/{benchmark_id}.json"},
            "protocol": {"adaptation": "none",
                         "evaluation": manifest["evaluation"]["test_aggregation"],
                         "contamination": old["protocol"]["contamination"]},
            "entries": entries}


def main() -> None:
    stale = []
    for corpus, (_, board, _) in TRACKS.items():
        path = ROOT / "results" / board
        body = json.dumps(build(corpus), indent=2) + "\n"
        if "--check" in sys.argv:
            if not path.exists() or path.read_text() != body:
                stale.append(board)
        else:
            path.write_text(body)
            print("wrote", path.relative_to(ROOT))
    if stale:
        sys.exit(f"stale: {stale} -- regenerate with tools/make_doc_leaderboards.py")
    if "--check" in sys.argv:
        print("document leaderboards: PROBLEMS 0")


if __name__ == "__main__":
    main()

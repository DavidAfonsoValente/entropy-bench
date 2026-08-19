"""Emit the domain-transfer results table body as LaTeX rows, ordered by news adapted BPB."""
import glob
import json
import sys

CORP = ["news", "reddit", "hackernews"]
SHORT = {"google/gemma-4-31B": "Gemma-4-31B", "google/gemma-4-12B": "Gemma-4-12B",
         "mistralai/Ministral-3-14B-Base-2512": "Ministral-3-14B",
         "Qwen/Qwen3.5-35B-A3B-Base": "Qwen-3.5-35B-MoE", "Qwen/Qwen3.5-9B-Base": "Qwen-3.5-9B",
         "Qwen/Qwen2.5-7B": "Qwen-2.5-7B", "Qwen/Qwen3.5-4B-Base": "Qwen-3.5-4B",
         "meta-llama/Llama-3.2-1B": "Llama-3.2-1B", "Qwen/Qwen2.5-1.5B": "Qwen-2.5-1.5B",
         "LiquidAI/LFM2.5-1.2B-Base": "LiquidAI-LFM2.5", "Qwen/Qwen2.5-0.5B": "Qwen-2.5-0.5B"}

src = sys.argv[1] if len(sys.argv) > 1 else "results/domain_transfer"
out = sys.argv[2] if len(sys.argv) > 2 else "dt_table.tex"

cells = {}
for f in glob.glob(src + "/*.json"):
    d = json.load(open(f))
    corpus = f.split("/")[-1].split("__")[0]
    cells.setdefault(SHORT[d["model_id"]], {})[corpus] = d

order = sorted(cells, key=lambda m: cells[m].get("news", {}).get("adapted_bpb", 9.0))
rows = []
for m in order:
    parts = []
    for c in CORP:
        d = cells[m].get(c)
        parts.append("%.3f & %.3f" % (d["zero_shot_bpb"], d["adapted_bpb"])
                     if d and "adapted_bpb" in d else "--- & ---")
    rows.append("%s & %s \\\\" % (m, " & ".join(parts)))

open(out, "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
print("\nwrote", out, "(%d models)" % len(rows))

"""Which models are fully adapted on which domain-transfer corpora?"""
import glob
import json
import sys

CORP = ["news", "reddit", "hackernews"]
ALL = ["google/gemma-4-31B", "Qwen/Qwen3.5-35B-A3B-Base",
       "mistralai/Ministral-3-14B-Base-2512", "google/gemma-4-12B",
       "Qwen/Qwen3.5-9B-Base", "Qwen/Qwen2.5-7B", "Qwen/Qwen3.5-4B-Base",
       "Qwen/Qwen2.5-1.5B", "meta-llama/Llama-3.2-1B",
       "LiquidAI/LFM2.5-1.2B-Base", "Qwen/Qwen2.5-0.5B"]

src = sys.argv[1] if len(sys.argv) > 1 else "results/domain_transfer"
cells = {}
for f in glob.glob(src + "/*.json"):
    d = json.load(open(f))
    cells.setdefault(d["model_id"], {})[f.split("/")[-1].split("__")[0]] = d

full, partial = [], []
print("%-26s %s" % ("model", " ".join(c[:6] for c in CORP)))
print("-" * 52)
for m in ALL:
    have = [c for c in CORP if c in cells.get(m, {}) and "adapted_bpb" in cells[m][c]]
    print("%-26s %s" % (m.split("/")[-1],
                        "   ".join("yes " if c in have else " -  " for c in CORP)))
    (full if len(have) == 3 else partial).append(m)

print()
print("fully adapted on all three domains: %d of %d" % (len(full), len(ALL)))
if partial:
    print("incomplete:")
    for m in partial:
        got = sorted(cells.get(m, {}))
        print("   %-26s has: %s" % (m.split("/")[-1], ", ".join(got) or "nothing"))

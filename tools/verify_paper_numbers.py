"""Check that every quantitative claim in the paper still matches the run artefacts.

Run before publishing. Exits nonzero if anything drifted.
"""
import glob
import json
import math
import os
import sys

DT = os.environ.get("LMAB_DOMAIN_TRANSFER_DIR", "results/domain_transfer")
TG = os.environ.get("LMAB_TOKEN_GAIN_DIR", "results/token_gain")
TG_BPB = os.environ.get("LMAB_TOKEN_GAIN_BPB_DIR", "results/token_gain_bpb")
CTX = "data/context_length"
CORPORA = ("news", "reddit", "hackernews")
MODELS = (
    "google/gemma-4-31B", "google/gemma-4-12B",
    "mistralai/Ministral-3-14B-Base-2512", "Qwen/Qwen3.5-35B-A3B-Base",
    "Qwen/Qwen3.5-9B-Base", "Qwen/Qwen2.5-7B", "Qwen/Qwen3.5-4B-Base",
    "meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-1.5B",
    "LiquidAI/LFM2.5-1.2B-Base", "Qwen/Qwen2.5-0.5B",
)
SHORT = {
    "google/gemma-4-31B": "Gemma-4-31B", "google/gemma-4-12B": "Gemma-4-12B",
    "mistralai/Ministral-3-14B-Base-2512": "Ministral-3-14B",
    "Qwen/Qwen3.5-35B-A3B-Base": "Qwen-3.5-35B-MoE",
    "Qwen/Qwen3.5-9B-Base": "Qwen-3.5-9B", "Qwen/Qwen2.5-7B": "Qwen-2.5-7B",
    "Qwen/Qwen3.5-4B-Base": "Qwen-3.5-4B", "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B": "Qwen-2.5-1.5B", "LiquidAI/LFM2.5-1.2B-Base": "LiquidAI-LFM2.5",
    "Qwen/Qwen2.5-0.5B": "Qwen-2.5-0.5B",
}

cells = {}
cell_sources = {}
duplicates = []
for f in sorted(glob.glob(DT + "/*.json")):
    d = json.load(open(f))
    corpus = os.path.basename(f).split("__")[0]
    key = (d["model_id"], corpus)
    if key in cell_sources:
        duplicates.append((key, cell_sources[key], f))
    cell_sources[key] = f
    cells.setdefault(d["model_id"], {})[corpus] = d

tg = {}
for f in glob.glob(TG + "/*.json"):
    d = json.load(open(f))
    if "model_id" in d:
        tg[d["model_id"]] = d

tex = open("paper_sota.tex").read()
try:
    tex += open("dt_table.tex").read()
except FileNotFoundError:
    pass

bad = 0


def require(label, ok, detail=""):
    global bad
    print("  %-42s %s%s" % (label, "OK" if ok else "MISMATCH",
                             "  " + detail if detail else ""))
    if not ok:
        bad += 1


def check(label, claimed, actual, tol=6e-4):
    global bad
    ok = abs(claimed - actual) <= tol
    print("  %-42s claimed %-9.4f actual %-9.4f %s"
          % (label, claimed, actual, "OK" if ok else "MISMATCH"))
    if not ok:
        bad += 1


print("Coverage and shared protocol:")
expected_cells = {(m, c) for m in MODELS for c in CORPORA}
actual_cells = set(cell_sources)
missing = sorted(expected_cells - actual_cells)
extra = sorted(actual_cells - expected_cells)
require("domain-transfer coverage", not missing and not extra and not duplicates,
        "%d/%d cells%s%s%s" % (
            len(actual_cells & expected_cells), len(expected_cells),
            " missing=" + repr(missing) if missing else "",
            " extra=" + repr(extra) if extra else "",
            " duplicates=" + repr(duplicates) if duplicates else ""))

protocol = (("lora.r", 16), ("lora.alpha", 32), ("lora.dropout", 0.05),
            ("lora.lr", 1e-4), ("lora.effective_batch", 32),
            ("train_steps", 250), ("max_seq_len", 512),
            ("mask_injected_special_tokens", True))
protocol_problems = []
for (mid, corpus), f in sorted(cell_sources.items()):
    d = cells[mid][corpus]
    for key in ("zero_shot_bpb", "adapted_bpb"):
        if not isinstance(d.get(key), (int, float)):
            protocol_problems.append((mid, corpus, key, d.get(key), "number"))
    for path, want in protocol:
        obj = d
        for part in path.split("."):
            obj = obj.get(part) if isinstance(obj, dict) else None
        if obj != want:
            protocol_problems.append((mid, corpus, path, obj, want))
require("shared per-cell protocol", not protocol_problems,
        "%d checked%s" % (len(actual_cells),
                           " problems=" + repr(protocol_problems) if protocol_problems else ""))

expected_rows = []
if not missing:
    for mid in sorted(MODELS, key=lambda m: cells[m]["news"]["adapted_bpb"]):
        values = []
        for corpus in CORPORA:
            d = cells[mid][corpus]
            values.append("%.3f & %.3f" % (d["zero_shot_bpb"], d["adapted_bpb"]))
        expected_rows.append("%s & %s \\\\" % (SHORT[mid], " & ".join(values)))
actual_table = open("dt_table.tex").read().strip().splitlines()
require("generated LaTeX table", not missing and actual_table == expected_rows,
        "%d rows" % len(actual_table))

print("Domain-transfer claims:")
g = cells.get("google/gemma-4-31B", {})
if g:
    for c, v in (("news", 0.624), ("reddit", 0.913), ("hackernews", 0.805)):
        if c in g:
            check("Gemma-4-31B adapted %s" % c, v, g[c]["adapted_bpb"], 1e-3)
lfm = cells.get("LiquidAI/LFM2.5-1.2B-Base", {})
if lfm:
    r = [lfm[c]["reduction_pct"] for c in ("news", "reddit", "hackernews") if c in lfm]
    ok = r and 34.5 <= min(r) and max(r) <= 38.5
    print("  %-42s %s  %s" % ("LFM reductions vs paper's 35--38%",
                              [round(x, 1) for x in r], "OK" if ok else "MISMATCH"))
    bad += 0 if ok else 1

print("Ordering stability (recomputed from the cells, not read off the .tex):")


def _ranks(d):
    return {k: i + 1 for i, k in enumerate(sorted(d, key=lambda k: d[k]))}


adapted = {}
zero_shot = {}
for mid, per in cells.items():
    for corp, d in per.items():
        if "adapted_bpb" in d:
            adapted.setdefault(corp, {})[mid] = d["adapted_bpb"]
        if "zero_shot_bpb" in d:
            zero_shot.setdefault(corp, {})[mid] = d["zero_shot_bpb"]

print("Adaptation-induced rank shifts:")
for corp, claimed_rho, claimed_shift, claimed_gemma_zero in (
        ("news", 0.655, 7, 4), ("reddit", 0.609, 7, 5),
        ("hackernews", 0.682, 5, 6)):
    before, after = zero_shot.get(corp, {}), adapted.get(corp, {})
    common = sorted(set(before) & set(after))
    if len(common) != 11:
        continue
    r0 = _ranks({m: before[m] for m in common})
    r1 = _ranks({m: after[m] for m in common})
    n = len(common)
    rho = 1 - 6 * sum((r0[m] - r1[m]) ** 2 for m in common) / (n * (n * n - 1))
    check("zero/adapt spearman on %s" % corp, claimed_rho, rho, 5e-4)
    shift = max(abs(r0[m] - r1[m]) for m in common)
    require("maximum adaptation rank move on %s" % corp, shift == claimed_shift,
            str(shift))
    gemma = "google/gemma-4-31B"
    require("Gemma-4-31B rank shift on %s" % corp,
            r0[gemma] == claimed_gemma_zero and r1[gemma] == 1,
            "%d->%d" % (r0[gemma], r1[gemma]))

for c1, c2, claimed in (("news", "reddit", 0.991), ("news", "hackernews", 0.955),
                        ("reddit", "hackernews", 0.964)):
    a, b = adapted.get(c1, {}), adapted.get(c2, {})
    common = [m for m in a if m in b]
    if len(common) < 3:
        continue
    r1, r2 = _ranks({m: a[m] for m in common}), _ranks({m: b[m] for m in common})
    n = len(common)
    rho = 1 - 6 * sum((r1[m] - r2[m]) ** 2 for m in common) / (n * (n * n - 1))
    check("spearman %s vs %s (n=%d)" % (c1, c2, n), claimed, rho, 5e-4)
    shift = max(abs(r1[m] - r2[m]) for m in common)
    if shift > 2:
        print("  MISMATCH: max rank shift %d exceeds the paper's 'two places'" % shift)
        bad += 1


def _pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) /
            math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)))


for c1, c2, claimed in (("news", "reddit", 0.992), ("news", "hackernews", 0.977),
                        ("reddit", "hackernews", 0.990)):
    a, b = adapted.get(c1, {}), adapted.get(c2, {})
    common = sorted(set(a) & set(b))
    if len(common) == 11:
        check("pearson %s vs %s (n=11)" % (c1, c2), claimed,
              _pearson([a[m] for m in common], [b[m] for m in common]), 5e-4)

# the cluster-compression argument: paper quotes these spreads to 4 dp
CLUSTERS = {"mid": (["google/gemma-4-12B", "mistralai/Ministral-3-14B-Base-2512",
                     "Qwen/Qwen3.5-35B-A3B-Base"], {"news": 0.0259, "hackernews": 0.0089}),
            "small": (["meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-1.5B"],
                      {"news": 0.0258, "hackernews": 0.0088})}
for name, (members, want) in CLUSTERS.items():
    for corp, claimed in want.items():
        vals = [adapted.get(corp, {}).get(m) for m in members]
        if any(v is None for v in vals):
            continue
        check("%s-cluster spread on %s" % (name, corp), claimed,
              max(vals) - min(vals), 5e-5)

print("Token-level:")
for mid, base, adapt, red in (
        ("google/gemma-4-12B", 2.838, 2.015, 29.0),
        ("LiquidAI/LFM2.5-1.2B-Base", 4.298, 2.728, 36.5),
        ("Qwen/Qwen2.5-1.5B", 2.737, 2.647, 3.3)):
    d = tg.get(mid)
    if not d:
        continue
    check(mid.split("/")[-1] + " base nats", base, d["mean_nats_per_token_base"], 1e-3)
    check(mid.split("/")[-1] + " adapted nats", adapt, d["mean_nats_per_token_adapted"], 1e-3)
    check(mid.split("/")[-1] + " reduction", red, d["relative_nats_reduction_pct"], 0.06)

print("Byte-normalized block positions:")
tg_bpb = {}
for f in glob.glob(TG_BPB + "/*.json"):
    d = json.load(open(f))
    tg_bpb[d["model_id"]] = d


def _bucket_bpb(d, keys, metric):
    rows = d["gain_by_block_position"]
    total_nats = sum(rows[k][metric] * rows[k]["positions"] for k in keys)
    total_bytes = sum(rows[k]["mean_bytes_per_token"] * rows[k]["positions"] for k in keys)
    return total_nats / math.log(2) / total_bytes


for mid, first8, late in (("google/gemma-4-12B", 2.566, 0.717),
                          ("Qwen/Qwen2.5-1.5B", 1.490, 0.773)):
    d = tg_bpb.get(mid)
    if d:
        check(mid.split("/")[-1] + " first-8 base BPB", first8,
              _bucket_bpb(d, ["0-0", "1-1", "2-3", "4-7"], "mean_base_loss"), 5e-4)
        check(mid.split("/")[-1] + " positions 256+ base BPB", late,
              d["gain_by_block_position"]["256-510"]["mean_base_bits_per_byte"], 5e-4)

print("Longer-context transfer:")
context_results = {}
for f in glob.glob(CTX + "/*.json"):
    d = json.load(open(f))
    context_results[d["model_id"]] = d
require("context-length model coverage",
        set(context_results) == {"google/gemma-4-12B", "Qwen/Qwen2.5-1.5B"},
        "%d/2 models" % len(context_results))
for mid, expected in {
    "google/gemma-4-12B": {"512": (0.6465, 0.5473), "2048": (0.6054, 0.5474)},
    "Qwen/Qwen2.5-1.5B": {"512": (0.7501, 0.7316), "2048": (0.7400, 0.7271)},
}.items():
    d = context_results.get(mid, {})
    require(mid.split("/")[-1] + " context protocol",
            d.get("training_context_tokens") == 512 and d.get("n_windows") == 64 and
            d.get("target_tokens_per_window") == 64)
    for context, (base, adapted_value) in expected.items():
        row = d.get("by_context_length", {}).get(context)
        if row:
            check(mid.split("/")[-1] + " base BPB @" + context, base, row["base_bpb"], 5e-5)
            check(mid.split("/")[-1] + " adapted BPB @" + context, adapted_value,
                  row["adapted_bpb"], 5e-5)

print("Literals present in the .tex:")
for lit in ["+0.991", "+0.955", "+0.964", "0.0259", "0.0089",
            "40{,}000", "250-step", "119{,}054", "141{,}527", "6{,}086",
            "0.992", "0.977", "0.990", "2.566", "1.490", "0.717", "0.773",
            "0.655", "0.609", "0.682", "8.6\\%", "0.844", "0.6465", "0.6054",
            "0.5473", "0.5474", "9.6\\%", "gemma2026gemma4", "liu2026ministral3",
            "qwen2026qwen35"]:
    if lit not in tex:
        print("  MISSING:", lit)
        bad += 1
    else:
        print("  ok:", lit)

for figure in ("figures/fig_block_position.pdf", "figures/fig_cross_corpus.pdf",
               "figures/fig_context_length.pdf"):
    require("figure " + os.path.basename(figure), os.path.isfile(figure) and os.path.getsize(figure) > 1000)

print("\nPROBLEMS:", bad)
sys.exit(1 if bad else 0)

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
STATIC_ANALYSIS = "results/static_benchmark_analysis.json"
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
try:
    tex += open("static_benchmark_table.tex").read()
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
# "math" is a deliberately separate, targeted experiment (docs/PLAN.md), not a fourth
# domain in the 33-cell design -- checked on its own below, not folded into "extra" here.
non_math_actual = {c for c in actual_cells if c[1] != "math"}
missing = sorted(expected_cells - non_math_actual)
extra = sorted(non_math_actual - expected_cells)
require("domain-transfer coverage", not missing and not extra and not duplicates,
        "%d/%d cells%s%s%s" % (
            len(non_math_actual & expected_cells), len(expected_cells),
            " missing=" + repr(missing) if missing else "",
            " extra=" + repr(extra) if extra else "",
            " duplicates=" + repr(duplicates) if duplicates else ""))

expected_math = {(m, "math") for m in MODELS}
actual_math = {c for c in actual_cells if c[1] == "math"}
require("math-domain coverage", actual_math == expected_math,
        "%d/%d cells" % (len(actual_math), len(expected_math)))

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

print("Static-benchmark raw-log recomputation (results/{gsm8k,hellaswag,mmlu_pro_1k}/):")
RAW_LABELS = ("LFM2.5-1.2B", "Llama-3.2-1B", "Ministral-3-14B", "Qwen2.5-0.5B",
              "Qwen2.5-1.5B", "Qwen2.5-7B", "Qwen3.5-4B", "Qwen3.5-9B", "gemma-4-12B",
              "gemma-4-31B", "Qwen3.5-35B-MoE")
_static_raw = json.load(open("results/combined_bpb_vs_static.json"))
_raw_bad = []
for label in RAW_LABELS:
    if label not in _static_raw:
        _raw_bad.append((label, "missing from combined_bpb_vs_static.json"))
        continue
    committed = _static_raw[label]
    for bench, task, metric in (
            ("gsm8k", "gsm8k", "exact_match,flexible-extract"),
            ("hellaswag", "hellaswag", "acc_norm,none"),
            ("mmlu_pro_1k", "mmlu_pro_1k", "exact_match,custom-extract")):
        matches = sorted(glob.glob("results/%s/%s/*/results_*.json" % (bench, label)))
        if not matches:
            _raw_bad.append((label, bench, "no results_*.json found"))
            continue
        raw = json.load(open(matches[-1]))
        got = raw.get("results", {}).get(task, {}).get(metric)
        committed_key = {"gsm8k": "gsm8k", "hellaswag": "hellaswag",
                          "mmlu_pro_1k": "mmlu_pro"}[bench]
        want = committed.get(committed_key)
        if got is None or want is None or abs(got - want) > 6e-4:
            _raw_bad.append((label, bench, "got=%r want=%r" % (got, want)))
require("all 33 cells recompute from raw logs", not _raw_bad,
        "%d/33 label-benchmark pairs%s" % (
            len(RAW_LABELS) * 3 - len(_raw_bad),
            "  problems=" + repr(_raw_bad) if _raw_bad else ""))

print("Static-benchmark alignment:")
static = json.load(open(STATIC_ANALYSIS))
require("static benchmark cohort", len(static.get("rows", [])) == 11,
        "%d/11 models" % len(static.get("rows", [])))
require("static BPB source is corrected rerun",
        static.get("bpb_protocol", {}).get("injected_special_token_targets_masked") is True and
        static.get("bpb_source") == "results/domain_transfer/news__*.json")
for task, claimed in {
    "mmlu_pro": (-0.826, 0.764, 0.564, 3, 1.64, 5),
    "hellaswag": (-0.975, 0.982, 0.927, 7, 0.36, 1),
    "gsm8k": (-0.614, 0.727, 0.564, 0, 2.00, 5),
}.items():
    row = static["alignment"][task]
    check(task + " Pearson", claimed[0], row["pearson_bpb_vs_accuracy"], 5e-4)
    check(task + " Spearman", claimed[1], row["spearman_rank_alignment"], 5e-4)
    check(task + " Kendall", claimed[2], row["kendall_tau_bpb_vs_error"], 5e-4)
    require(task + " exact ranks", row["identical_ranks"] == claimed[3], str(row["identical_ranks"]))
    check(task + " mean rank shift", claimed[4], row["mean_absolute_rank_shift"], 0.006)
    require(task + " max rank shift", row["max_absolute_rank_shift"] == claimed[5],
            str(row["max_absolute_rank_shift"]))

print("Math-domain claims (Table tab:math_rank_alignment, Section sec:math):")


def _ranks_local(d):
    return {k: i + 1 for i, k in enumerate(sorted(d, key=lambda k: d[k]))}


math_bpb = {mid: cells[mid]["math"]["adapted_bpb"] for mid in MODELS if "math" in cells.get(mid, {})}
if len(math_bpb) == 11:
    static_acc = json.load(open("results/combined_bpb_vs_static.json"))
    id_to_short = {v: k for k, v in {
        "gemma-4-31B": "google/gemma-4-31B", "gemma-4-12B": "google/gemma-4-12B",
        "Ministral-3-14B": "mistralai/Ministral-3-14B-Base-2512",
        "Qwen3.5-35B-MoE": "Qwen/Qwen3.5-35B-A3B-Base", "Qwen3.5-9B": "Qwen/Qwen3.5-9B-Base",
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B", "Qwen3.5-4B": "Qwen/Qwen3.5-4B-Base",
        "Llama-3.2-1B": "meta-llama/Llama-3.2-1B", "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
        "LFM2.5-1.2B": "LiquidAI/LFM2.5-1.2B-Base", "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    }.items()}
    math_rank = _ranks_local(math_bpb)
    news_bpb_math_cohort = {mid: cells[mid]["news"]["adapted_bpb"] for mid in MODELS}
    news_rank_math_cohort = _ranks_local(news_bpb_math_cohort)

    for task, claimed_news_rho, claimed_math_rho, claimed_delta, claimed_exact in (
            ("gsm8k", 0.727, 0.936, 0.209, 6),
            ("mmlu_pro", 0.764, 0.936, 0.172, 3),
            ("hellaswag", 0.982, 0.818, -0.164, 2)):
        acc = {mid: static_acc[id_to_short[mid]][task] for mid in MODELS}
        acc_rank = _ranks_local({m: -acc[m] for m in acc})  # higher accuracy -> better (lower) rank
        n = len(MODELS)
        d2 = sum((math_rank[m] - acc_rank[m]) ** 2 for m in MODELS)
        rho = 1 - 6 * d2 / (n * (n * n - 1))
        exact = sum(math_rank[m] == acc_rank[m] for m in MODELS)
        check(task + " math-BPB Spearman", claimed_math_rho, rho, 5e-4)
        require(task + " math-BPB exact ranks", exact == claimed_exact, str(exact))
        check(task + " Spearman delta vs news", claimed_delta, rho - claimed_news_rho, 6e-4)

    d2_nm = sum((math_rank[m] - news_rank_math_cohort[m]) ** 2 for m in MODELS)
    rho_news_math = 1 - 6 * d2_nm / (len(MODELS) * (len(MODELS) ** 2 - 1))
    check("news-vs-math BPB rank correlation", 0.755, rho_news_math, 5e-4)

    loo_values = []
    for drop in MODELS:
        subset = [m for m in MODELS if m != drop]
        sub_math_rank = _ranks_local({m: math_bpb[m] for m in subset})
        sub_acc = {mid: static_acc[id_to_short[mid]]["gsm8k"] for mid in subset}
        sub_acc_rank = _ranks_local({m: -sub_acc[m] for m in sub_acc})
        n = len(subset)
        d2 = sum((sub_math_rank[m] - sub_acc_rank[m]) ** 2 for m in subset)
        loo_values.append(1 - 6 * d2 / (n * (n * n - 1)))
    require("GSM8K leave-one-out range", round(min(loo_values), 3) == 0.915 and
            round(max(loo_values), 3) == 0.952,
            "[%.3f, %.3f]" % (min(loo_values), max(loo_values)))
else:
    require("math-domain BPB present for all 11 models", False, "%d/11" % len(math_bpb))

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

print("Alignment matrix (tools/analyze_alignment_matrix.py, all 4 corpora x 2 tiers):")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import analyze_alignment_matrix as _am  # noqa: E402

_matrix = _am.build()
# Every rho the paper states inline, checked against a fresh recomputation rather than
# against the committed JSON, so a stale artifact cannot hide a changed number.
for _cell, _task, _claimed in [
        ("news__zero_shot", "gsm8k", 0.836), ("news__zero_shot", "mmlu_pro", 0.855),
        ("news__zero_shot", "hellaswag", 0.718),
        ("news__adapted", "gsm8k", 0.727), ("news__adapted", "mmlu_pro", 0.764),
        ("news__adapted", "hellaswag", 0.982),
        ("reddit__zero_shot", "gsm8k", 0.791), ("reddit__zero_shot", "hellaswag", 0.655),
        ("reddit__adapted", "gsm8k", 0.691), ("reddit__adapted", "mmlu_pro", 0.736),
        ("reddit__adapted", "hellaswag", 0.964),
        ("hackernews__zero_shot", "gsm8k", 0.827), ("hackernews__zero_shot", "mmlu_pro", 0.818),
        ("hackernews__zero_shot", "hellaswag", 0.627),
        ("hackernews__adapted", "gsm8k", 0.791), ("hackernews__adapted", "mmlu_pro", 0.845),
        ("hackernews__adapted", "hellaswag", 0.973),
        ("math__zero_shot", "gsm8k", 0.782),
        ("math__adapted", "gsm8k", 0.936), ("math__adapted", "mmlu_pro", 0.936),
        ("math__adapted", "hellaswag", 0.818)]:
    check("rho %s/%s" % (_cell, _task), _claimed,
          _matrix["cells"][_cell]["alignment"][_task]["spearman"], 5e-4)

# Selection regret, which the abstract and Section 5.4 quote in accuracy points.
for _cell, _task, _claimed in [
        ("news__zero_shot", "gsm8k", 0.0), ("news__zero_shot", "mmlu_pro", 0.0),
        ("news__adapted", "gsm8k", 2.3), ("news__adapted", "mmlu_pro", 4.2),
        ("news__adapted", "hellaswag", 0.0),
        ("math__adapted", "gsm8k", 0.0), ("math__adapted", "mmlu_pro", 0.0)]:
    check("regret pp %s/%s" % (_cell, _task), _claimed,
          _matrix["cells"][_cell]["alignment"][_task]["top1_regret_pp"], 6e-2)

# Pairwise counts quoted in the paper.
for _cell, _claimed in [("news__zero_shot", 46), ("news__adapted", 43),
                        ("reddit__adapted", 42), ("hackernews__adapted", 45),
                        ("math__adapted", 50)]:
    require("GSM8K pairs %s = %d/55" % (_cell, _claimed),
            _matrix["cells"][_cell]["alignment"]["gsm8k"]["pairs_correct"] == _claimed)

_red = _matrix["benchmark_redundancy"]["gsm8k__vs__mmlu_pro"]
require("GSM8K/MMLU-Pro redundancy 51/55", _red["pairs_agree"] == 51 and _red["pairs_total"] == 55)
check("GSM8K/MMLU-Pro Kendall tau", 0.855, _red["kendall_tau"], 5e-4)

_boot = _matrix["contrasts"]["math_adapted_vs_news_adapted"]["gsm8k"]
check("bootstrap delta math vs news", 0.209, _boot["delta_spearman"], 5e-4)
check("bootstrap CI low", -0.096, _boot["ci_low"], 5e-3)
check("bootstrap CI high", 0.704, _boot["ci_high"], 5e-3)
require("bootstrap CI spans zero (paper says so)", not _boot["excludes_zero"])
check("bootstrap delta math vs zero-shot", 0.100,
      _matrix["contrasts"]["math_adapted_vs_news_zero_shot"]["gsm8k"]["delta_spearman"], 5e-4)

_bias = _matrix["tokenizer_bias_bound"]
require("token-blocking bias overturns no rank", not _bias["overturns_any_rank"])
require("token-blocking bias <= 0.08x gap (paper's bound)", _bias["max_bias_over_gap"] <= 0.085)

_mb = _bias["scored_text_mb"].values()
require("scored text spread 5.92-6.24 MB",
        abs(min(_mb) - 5.92) < 0.01 and abs(max(_mb) - 6.24) < 0.01)

# The generated tables the paper \input must match a fresh computation.
require("alignment_matrix_table.tex current",
        open("alignment_matrix_table.tex").read() == _am.render_matrix_tex(_matrix))
require("selection_regret_table.tex current",
        open("selection_regret_table.tex").read() == _am.render_regret_tex(_matrix))

print("Literals present in the .tex:")
for lit in ["0.991", "0.955", "0.964", "0.0259", "0.0089",
            "119{,}054", "141{,}527", "6{,}086",
            "0.992", "0.977", "0.990", "2.566", "1.490", "0.717", "0.773",
            "0.655", "0.609", "0.682", "8.6\\%", "0.844", "0.6465", "0.6054",
            "0.5473", "0.5474", "9.6\\%", "gemma2026gemma4", "liu2026ministral3",
            "qwen2026qwen35", "0.982", "0.764", "0.727", "7/11", "3/11", "0/11",
            "2.41", "5.03", "1.92", "0.40", "zhang2025trainbeforetest",
            "heineman2025signal",
            "0.936", "0.755", "0.915, 0.952", "+0.209", "3{,}207", "sec:math",
            "0.836", "0.855", "0.718", "0.791", "0.736", "0.827", "0.845", "0.973",
            "0.782", "0.691", "0.627", "0.964", "0.818",
            "46/55", "50/55", "51 of 55", "0.855",
            "-0.096, +0.704", "+0.000, +0.393", "$+0.100$",
            "0.08", "5.92--6.24", "sec:convergence",
            "tab:alignment_matrix", "tab:regret", "legacy\\_sweep",
            "sec:adaptedacc", "$-1.47$", "$t(4)=-1.63$", "$r=-0.935$", "$-0.036$",
            "$36$--$42\\%$", "$3.6$--$7.9\\%$", "$-0.60$ to $+0.69$",
            "$\\rho=1.00$", "$1/5!=0.0083$", "$p<0.01$", "on the order",
            "sec:pairwise", "$23$ of the $24$", "$0.909$", "$0.964$",
            "$[0.478,0.980]$", "$+0.127$", "$-0.127$", "cluster bootstrap",
            "$0.545$", "$0.818$", "$0.857$", "$0.964$", "$0.636$",
            "thrush2024perplexity", "arXiv:2409.05816",
            "$0.933$", "$0.863$", "$0.0003$"]:
    if lit not in tex:
        print("  MISSING:", lit)
        bad += 1
    else:
        print("  ok:", lit)

# ------------------------------------------------- gemma-4 exclusion, quoted in sec:adaptedacc
# The paper says gemma-4's per-layer attention width genuinely varies, which is why the one-line
# config override was refused rather than used. That is a factual claim about a released model, so
# check it against the model's own committed config instead of trusting the prose.
print("\ngemma-4 exclusion grounds")
GEMMA_CFG = "results/model_configs/gemma-4-12B.config.json"
try:
    gcfg = json.load(open(GEMMA_CFG))["text_config"]
except (FileNotFoundError, KeyError) as exc:
    print("  MISSING or malformed:", GEMMA_CFG, exc)
    bad += 1
else:
    require("gemma head_dim is 256", gcfg.get("head_dim") == 256, str(gcfg.get("head_dim")))
    require("gemma global_head_dim is 512", gcfg.get("global_head_dim") == 512,
            str(gcfg.get("global_head_dim")))
    require("the two genuinely differ", gcfg.get("head_dim") != gcfg.get("global_head_dim"))
    require("no per_layer_config to read instead", "per_layer_config" not in gcfg)
    n_full = sum(1 for x in gcfg.get("layer_types", []) if x == "full_attention")
    n_slide = sum(1 for x in gcfg.get("layer_types", []) if x == "sliding_attention")
    require("48 layers, 40 sliding + 8 full", (n_slide, n_full) == (40, 8),
            "%d sliding / %d full" % (n_slide, n_full))
    require("paper quotes head_dim=256", "\\mathtt{head\\_dim}=256" in tex)
    require("paper quotes global_head_dim=512", "\\mathtt{global\\_head\\_dim}=512" in tex)

# ------------------------------------------------- selection accuracy, quoted in sec:pairwise
print("\nselection accuracy (model-level cluster bootstrap)")
try:
    ps = json.load(open("results/pairwise_significance.json"))
except FileNotFoundError:
    print("  MISSING: results/pairwise_significance.json"); bad += 1
else:
    sel = ps["selection_accuracy"]
    require("24 cells measured", len(sel) == 24, str(len(sel)))
    require("23 of 24 beat chance", ps["cells_beating_chance"] == 23,
            "%d/%d" % (ps["cells_beating_chance"], ps["cells_total"]))
    fails = [k for k, v in sel.items() if not v["beats_chance"]]
    require("the exception is math zero-shot vs HellaSwag",
            fails == ["math__zero_shot__hellaswag"], str(fails))
    check("that cell's accuracy", 0.727, sel["math__zero_shot__hellaswag"]["accuracy"], tol=1e-3)
    check("news-adapted HellaSwag accuracy", 0.964, sel["news__adapted__hellaswag"]["accuracy"],
          tol=1e-3)
    check("math-adapted GSM8K accuracy", 0.909, sel["math__adapted__gsm8k"]["accuracy"], tol=1e-3)

    # The paper's honesty rests on these NOT being significant. Gate that they stay that way.
    for bench, r in ps["corpus_steering_contrast"].items():
        require("steering contrast unresolved (%s)" % bench, not r["excludes_zero"],
                "[%.3f, %.3f]" % (r["ci_lo"], r["ci_hi"]))
    for name, r in ps["adaptation_vs_doing_nothing"].items():
        require("adapt-vs-nothing unresolved (%s)" % name, not r["excludes_zero"],
                "[%.3f, %.3f]" % (r["ci_lo"], r["ci_hi"]))
    # Reproduction claim in sec:pairwise: the re-run cells must match the published ones.
    for corpus, zs, ad in (("news", 0.7181, 0.6791), ("math", 0.6718, 0.6221)):
        pub = json.load(open("results/domain_transfer/%s__mistralai_Ministral-3-14B-Base-2512.json"
                             % corpus))
        rerun = json.load(open("results/adapted_bench/bpb/%s__mistralai_Ministral-3-14B-Base-2512"
                               ".adapted.json" % corpus))
        check("reproduction %s zero-shot" % corpus, pub["zero_shot_bpb"], rerun["zero_shot_bpb"],
              tol=5e-5)
        require("reproduction %s adapted within 0.0003" % corpus,
                abs(pub["adapted_bpb"] - rerun["adapted_bpb"]) <= 3.5e-4,
                "%.4f vs %.4f" % (pub["adapted_bpb"], rerun["adapted_bpb"]))

    base = ps["size_baseline"]
    check("size baseline, all pairs, gsm8k", 0.855, base["all_pairs"]["gsm8k"]["bigger_model"],
          tol=1e-3)
    check("bpb, all pairs, gsm8k", 0.909, base["all_pairs"]["gsm8k"]["bpb"], tol=1e-3)
    check("size baseline, all pairs, hellaswag", 0.964,
          base["all_pairs"]["hellaswag"]["bigger_model"], tol=1e-3)
    require("size beats BPB on HellaSwag over all pairs",
            base["all_pairs"]["hellaswag"]["bigger_model"] > base["all_pairs"]["hellaswag"]["bpb"])
    check("within 2x, size on gsm8k", 0.545, base["within_2x"]["gsm8k"]["bigger_model"], tol=1e-3)
    check("within 2x, bpb on gsm8k", 0.818, base["within_2x"]["gsm8k"]["bpb"], tol=1e-3)
    check("within 1.5x, bpb on gsm8k", 0.857, base["within_1.5x"]["gsm8k"]["bpb"], tol=1e-3)
    check("within 2x, zero-shot on gsm8k", 0.636, base["within_2x"]["gsm8k"]["zero_shot"],
          tol=1e-3)
    require("adapted beats zero-shot beats size in the hard band",
            base["within_2x"]["gsm8k"]["bpb"] > base["within_2x"]["gsm8k"]["zero_shot"]
            > base["within_2x"]["gsm8k"]["bigger_model"])
    require("paper reports the size baseline",
            "pick the\nbigger model" in tex and "$70\\times$" in tex)
    # The steering DiD used to be hand-carried into the LaTeX and only string-matched here, which
    # is exactly the failure mode this script exists to prevent. It now regenerates from
    # results/pairwise_significance.json, and the FAMILY-level resampling is checked too: the
    # paper's own narrative says GSM8K rank movement is a family property, so a model-level
    # bootstrap alone is not the conservative reading and must not be the only one reported.
    did = ps["steering_difference_in_differences"]
    check("steering DiD, pooled", 0.242, did["pooled"], tol=5e-4)
    check("steering DiD vs news", 0.255, did["per_corpus"]["news"], tol=5e-4)
    check("steering DiD vs hackernews", 0.218, did["per_corpus"]["hackernews"], tol=5e-4)
    model_level = did["resampling"]["model"]
    family_level = did["resampling"]["publisher_family"]
    require("paper reports the DiD sign split it computed",
            ("%.1f" % (model_level["frac_positive"] * 100)) + "\\%" in tex,
            "%.1f%%" % (model_level["frac_positive"] * 100))
    require("family-level resampling is weaker than model-level, as the paper says",
            family_level["frac_positive"] < model_level["frac_positive"])
    require("family-level direction never reverses",
            family_level["frac_negative"] == 0.0,
            "%.3f" % family_level["frac_negative"])
    require("paper reports the family-level clustering at all",
            "family-level" in tex or "publisher" in tex)
    require("paper names the family the effect is concentrated in",
            did["most_load_bearing_family"] in ("Gemma", "gemma"))
    require("leave-one-family-out is reported in the paper",
            "$+0.056$" in tex)

    # Was a hard-coded string match on 93.5%/1.4%. Those were hand-carried numbers; the split is
    # now regenerated from the artifact, so the gate checks the paper against what the tool
    # actually computed -- including the family-clustered reading, which is the primary one.
    require("paper leads the steering result with the resample split",
            ("%.1f" % (model_level["frac_positive"] * 100)) + "\\%" in tex
            and ("%.1f" % (family_level["frac_positive"] * 100)) + "\\%" in tex)

    require("paper does not quote the invalid p-values",
            "$p=0.033$" not in tex and "$p=0.031$" not in tex and "$p=0.012$" not in tex)

# ---------------------------------------------------------------- experiment E2
# Section sec:adaptedacc reports rank preservation, the paired contrast and the leverage check.
# Recompute all three from the committed artefacts rather than trusting the prose.
print("\nE2 adapted-benchmark claims")
try:
    e2 = json.load(open("results/adapted_benchmark_analysis.json"))
    audit = json.load(open("results/e2_confound_audit.json"))
except FileNotFoundError as exc:
    print("  MISSING artefact:", exc)
    bad += 1
else:
    rp = e2["rank_preservation"]
    require("E2 every ordering preserved",
            rp["orderings_preserved"] == rp["orderings_total"] == 20,
            "%d/%d" % (rp["orderings_preserved"], rp["orderings_total"]))
    require("E2 all four cells fully preserved", rp["all_cells_fully_preserved"] is True)
    check("E2 conservative p", 1.0 / 120.0, rp["p_conservative"], tol=1e-9)
    require("E2 every cell rho = 1.00",
            all(c["ranks_preserved"] == c["n"] for c in rp["per_cell"].values()))
    total = sum(c["n_aggregated"] for c in e2["cells"].values())
    n_rankings = len(e2["cells"])
    per = {c["n_aggregated"] for c in e2["cells"].values()}
    require("E2 unit is 4 rankings of 5 models",
            n_rankings == 4 and per == {5} and "four rankings of five models" in tex,
            "%d rankings, sizes %s" % (n_rankings, sorted(per)))
    unmatched = sum(c["n_models"] - c["n_matched"] for c in e2["cells"].values())
    require("E2 every base is same-host", unmatched == 0, "%d unmatched" % unmatched)

    pc = audit["paired_contrast"]
    check("E2 paired contrast (pp)", -1.47, pc["mean_pp"], tol=5e-3)
    check("E2 paired t statistic", -1.63, pc["t"], tol=5e-3)
    require("E2 paired df matches the paper", pc["df"] == 4, "df=%d" % pc["df"])
    require("E2 negative-side count", pc["n_negative"] == 3, "%d/%d" % (pc["n_negative"], pc["n"]))
    require("E2 power target is ~15 models",
            14 <= pc["n_models_for_80pct_power"] <= 16,
            "%.1f" % pc["n_models_for_80pct_power"])

    lev = audit["bpb_leverage"]
    check("E2 leverage r, all cells", -0.935, lev["all"]["pearson"], tol=5e-4)
    check("E2 leverage r, dropping one", -0.036, lev["excluding"]["pearson"], tol=5e-4)
    require("E2 leverage model is LFM2.5", lev["excluding"]["model"] == "LFM2.5-1.2B",
            lev["excluding"]["model"])

for figure in ("figures/fig_block_position.pdf", "figures/fig_cross_corpus.pdf",
               "figures/fig_context_length.pdf", "figures/fig_benchmark_alignment.tex"):
    require("figure " + os.path.basename(figure), os.path.isfile(figure) and os.path.getsize(figure) > 1000)

print("\nPROBLEMS:", bad)
sys.exit(1 if bad else 0)

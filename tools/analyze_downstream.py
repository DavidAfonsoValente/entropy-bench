#!/usr/bin/env python3
"""Does the selector's pick actually build the better fine-tuned system?

E11, the verdict half. Written and committed BEFORE the GPU job returns, so the prediction, the
metric and the equivalence margin are all fixed in advance rather than chosen once the numbers are
visible.

Two size-matched pairs are run, chosen because the selectors disagree on them and for opposite
reasons -- one at the top of the cohort and one at the bottom:

* **Gemma-4-12B vs Ministral-3-14B** (1.17x apart). Adapted BPB says Gemma; MMLU-Pro (44.8 vs 52.8)
  and GSM8K (64.8 vs 80.8) say Ministral. The in-domain cloze criterion splits by convention here --
  lenient resolves for Gemma, strict resolves for Ministral -- which is exactly why a third,
  independent outcome is worth having on this pair.
* **Llama-3.2-1B vs Qwen-2.5-1.5B** (1.25x apart). Adapted BPB says Llama and the cloze criterion
  agrees under both conventions; all three static benchmarks say Qwen, GSM8K by 6.4 against 61.6.

**The preregistered outcomes.** For each pair, the mean ROUGE-L F1 difference between the two
fine-tuned systems on 500 held-out articles, with a paired bootstrap over documents:

* CI excludes zero, sign matches the BPB pick   -> ``BPB PREDICTS``
* CI excludes zero, sign matches the benchmarks -> ``BENCHMARKS PREDICT``
* CI lies entirely inside +/- MARGIN            -> ``EQUIVALENT``
* otherwise                                     -> ``INCONCLUSIVE``

**What this cannot establish**, stated here so the paper does not overshoot it when the number
lands: one task, one domain, one seed per model, two pairs. A result is "on this task the selector's
pick produced the better fine-tuned system", never "adapted BPB predicts downstream utility".

Lede generation rewards copying: a news body restates its own opening. Two reference points are
reported beside every score -- the article's first body sentence scored as if it were the answer,
and the unadapted base model's own generations -- so a reader can see how much of any score is
retrieval and how much the fine-tune added.

    python tools/analyze_downstream.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = Path(os.environ.get("E11_DIR", str(Path(os.environ.get("SCRATCH", "/tmp")) /
                                           "e11_downstream")))
OUTPUT_JSON = ROOT / "results" / "downstream.json"
SCHEMA_VERSION = 1
SEED = 20260911
N_BOOT = 20_000
# Practical-equivalence margin on ROUGE-L F1, fixed before the data. One point of ROUGE-L is below
# what a reader would act on when choosing between two candidate base models.
MARGIN = 0.01

# (label_a, label_b, who adapted BPB picks, who the static benchmarks pick).
PAIRS = [
    ("gemma-4-12B", "Ministral-3-14B", "gemma-4-12B", "Ministral-3-14B"),
    ("Llama-3.2-1B", "Qwen2.5-1.5B", "Llama-3.2-1B", "Qwen2.5-1.5B"),
]

WORD = re.compile(r"[a-z0-9]+")
# Base checkpoints have no stop convention and will write on past the lede. The same cutoff is
# applied to every model: the first blank line, which is the paragraph boundary the task is defined
# by in the first place.
PARA_BREAK = re.compile(r"\n\s*\n")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length, two rows of DP (O(len(a)) memory)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1. Hand-rolled because rouge_score is not installed on this cluster; the unit test
    in tools/test_downstream_metric.py pins it against worked examples."""
    c, r = tokenize(candidate), tokenize(reference)
    if not c or not r:
        return 0.0
    l = lcs_length(c, r)
    if l == 0:
        return 0.0
    p, rec = l / len(c), l / len(r)
    return 2 * p * rec / (p + rec)


def first_paragraph(text: str) -> str:
    return PARA_BREAK.split(text.strip(), 1)[0].strip()


def load_cell(label: str, prefix: str = "sft__") -> dict:
    path = CELLS / f"{prefix}{label}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run slurm/downstream_sft.sbatch first")
    return json.loads(path.read_text())


def score_cell(cell: dict, key: str) -> dict[str, float]:
    return {g["id"]: rouge_l(first_paragraph(g["generated"]), g["reference"])
            for g in cell[key]}


def paired_bootstrap(a: dict[str, float], b: dict[str, float]) -> dict:
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit("the two cells share no evaluation ids")
    diffs = [a[i] - b[i] for i in ids]
    n = len(diffs)
    rng = random.Random(SEED)
    means = []
    for _ in range(N_BOOT):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {"n_items": n, "mean_diff": sum(diffs) / n,
            "ci_lo": means[int(0.025 * N_BOOT)], "ci_hi": means[int(0.975 * N_BOOT) - 1]}


def verdict_for(ci: dict, bpb_pick: str, bench_pick: str, a: str) -> str:
    """The preregistered decision rule, applied mechanically."""
    lo, hi = ci["ci_lo"], ci["ci_hi"]
    if bpb_pick == bench_pick:
        raise SystemExit(f"{a}: the two selectors agree, so this pair decides nothing")
    if lo > 0 or hi < 0:
        # The interval is on a minus b, and the two selectors disagree by construction, so the
        # sign names the winner without any further reasoning.
        winner = a if ci["mean_diff"] > 0 else (bench_pick if bpb_pick == a else bpb_pick)
        if winner == bpb_pick:
            return "BPB PREDICTS"
        if winner == bench_pick:
            return "BENCHMARKS PREDICT"
        return "INCONCLUSIVE"
    if -MARGIN <= lo and hi <= MARGIN:
        return "EQUIVALENT"
    return "INCONCLUSIVE"


def build() -> dict:
    bodies = {}
    eval_path = CELLS / "eval.jsonl"
    if eval_path.exists():
        for line in eval_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                bodies[row["id"]] = row

    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "question": "On a real in-domain fine-tune, does the model adapted BPB picks beat the one "
                    "the static benchmarks pick?",
        "preregistered": {"metric": "ROUGE-L F1 against the held-out human lede",
                          "uncertainty": f"paired bootstrap over evaluation documents, {N_BOOT} "
                                         "draws, 95% percentile interval",
                          "equivalence_margin_rouge_l": MARGIN,
                          "outcomes": ["BPB PREDICTS", "BENCHMARKS PREDICT", "EQUIVALENT",
                                       "INCONCLUSIVE"]},
        "scope_limit": "one task, one domain, one seed per model, two pairs; this is evidence "
                       "about this fine-tune, not a general claim about downstream utility",
        "pairs": [],
    }

    for a, b, bpb_pick, bench_pick in PAIRS:
        cells = {m: load_cell(m) for m in (a, b)}
        sft = {m: score_cell(cells[m], "generations") for m in (a, b)}
        base = {m: (score_cell(cells[m], "base_generations")
                    if cells[m].get("base_generations") else {}) for m in (a, b)}

        ci = paired_bootstrap(sft[a], sft[b])
        row = {
            "pair": f"{a} vs {b}", "adapted_bpb_picks": bpb_pick,
            "static_benchmarks_pick": bench_pick,
            "rouge_l": {m: sum(sft[m].values()) / len(sft[m]) for m in (a, b)},
            "rouge_l_base_weights": {m: (sum(base[m].values()) / len(base[m]) if base[m] else None)
                                     for m in (a, b)},
            "paired_bootstrap_a_minus_b": ci,
            "verdict": verdict_for(ci, bpb_pick, bench_pick, a),
        }
        # A fine-tune that did not move a model off its base score is not a fine-tune, and ordering
        # two such systems would be ordering their base behaviour under another name.
        gains = {m: (row["rouge_l"][m] - row["rouge_l_base_weights"][m])
                 if row["rouge_l_base_weights"][m] is not None else None for m in (a, b)}
        row["rouge_l_gain_from_finetuning"] = gains
        if any(g is not None and g <= 0 for g in gains.values()):
            row["verdict"] = "VOID: fine-tuning did not improve at least one model over its base " \
                             "weights, so the comparison is not between two adapted systems"

        if bodies:
            ids = sorted(sft[a])
            # How much of any score is available by copying the article back out.
            first_sent = [rouge_l(SENTENCE.split(bodies[i]["body"].strip())[0],
                                  bodies[i]["lede"]) for i in ids if i in bodies]
            rng = random.Random(SEED)
            rand_sent = []
            for i in ids:
                if i not in bodies:
                    continue
                sents = [s for s in SENTENCE.split(bodies[i]["body"].strip()) if len(s) > 40]
                if sents:
                    rand_sent.append(rouge_l(rng.choice(sents), bodies[i]["lede"]))
            row["copy_baselines"] = {
                "first_body_sentence": sum(first_sent) / len(first_sent) if first_sent else None,
                "random_body_sentence": sum(rand_sent) / len(rand_sent) if rand_sent else None,
                "note": "a news body restates its own lede; these bound what retrieval alone buys",
            }
        row["effective_sample"] = {
            "n_scored": ci["n_items"],
            # rouge_l tokenizes on [a-z0-9]+, so an item written in a non-Latin script scores 0 for
            # BOTH models. That is not a bias between them, but it dilutes n, so it is counted.
            "n_references_with_fewer_than_10_ascii_words": sum(
                1 for i in sorted(sft[a]) if i in bodies and len(tokenize(bodies[i]["lede"])) < 10),
        }
        result["pairs"].append(row)

    result["cohort"] = build_cohort()
    # E11c: the same fifteen models fine-tuned with recipe B (rank 64, lr 3e-4, batch 16, 400
    # steps, new seed), so the downstream outcome no longer shares the selector's recipe.
    result["cohort_recipe_b"] = build_cohort("sftB__")
    # Do the two recipes build the same ORDERING of systems? If they did not, agreement with either
    # would be a fact about that recipe. Spearman over the fifteen fine-tuned ROUGE-L scores.
    ra, rb = result["cohort"].get("rouge_l"), result["cohort_recipe_b"].get("rouge_l")
    if ra and rb and set(ra) == set(rb):
        ms = sorted(ra)
        rank = lambda v: {m: i for i, m in enumerate(sorted(ms, key=lambda k: -v[k]))}
        ka, kb = rank(ra), rank(rb)
        n = len(ms)
        result["recipe_agreement"] = {
            "spearman_rouge_a_vs_b": 1 - 6 * sum((ka[m] - kb[m]) ** 2 for m in ms) / (n * (n * n - 1)),
            "recipe_b_below_a_for_every_model": all(rb[m] < ra[m] for m in ms),
            "n_models": n}
    verdicts = [p["verdict"] for p in result["pairs"]]
    result["summary"] = ("both pairs: " + "; ".join(verdicts)) if len(set(verdicts)) > 1 else \
        f"both pairs agree: {verdicts[0]}"
    return result


# ---------------------------------------------------------------------------- E11b: every pair
# Written before any E11b cell returned. A model's fine-tuned score does not depend on which model
# it is later compared with, so fine-tuning every cohort model that fits one A100 lets the
# downstream outcome score EVERY in-band pair -- the same statistic, band and cluster bootstrap as
# the paper's headline table, with the fine-tuned system's ROUGE-L as the criterion. No pair is
# chosen by anyone. The two 30B-class checkpoints need more than one GPU and are excluded by that
# rule, fixed in slurm/downstream_sft_all.sbatch before the run; it removes one in-band pair.
EXCLUDED_MULTI_GPU = ("gemma-4-31B", "Qwen3.5-35B-MoE")
PAIR_BOOT = 2_000   # per-pair resolution only; the headline interval uses the full 20,000
BAND = 2.0


def _selectors_17():
    """Every selector for the 17-model cohort, loaded the way analyze_cohort_extension does."""
    sys.path.insert(0, str(ROOT / "tools"))
    from analyze_alignment_matrix import load_bpb_matrix
    from analyze_cohort_extension import CONTROL_LABEL, _load_ext
    from analyze_static_benchmarks import MODEL_IDS, load_raw_static
    from pairwise_significance import _load_params

    dt, bench, _ = _load_ext()
    pub, _ = load_bpb_matrix()
    zero = dict(pub["news"]["zero_shot_bpb"])
    adapt = dict(pub["news"]["adapted_bpb"])
    params = dict(_load_params())
    accs = {b: {m: load_raw_static(m, b)[0] for m in MODEL_IDS}
            for b in ("gsm8k", "mmlu_pro", "hellaswag")}
    raw = {"gsm8k": "gsm8k", "mmlu_pro": "mmlu_pro_1k", "hellaswag": "hellaswag"}
    for label, d in dt.items():
        if label == CONTROL_LABEL or label.startswith("OLMo"):
            continue
        zero[label], adapt[label], params[label] = (d["zero_shot_bpb"], d["adapted_bpb"],
                                                    d["total_params"])
        for key, r in raw.items():
            if r in bench.get(label, {}):
                accs[key][label] = bench[label][r]
    sel = {"matched_adapted_bpb": (adapt, True), "zero_shot_bpb": (zero, True),
           "parameter_count": (params, False)}
    sel.update({b: (accs[b], False) for b in ("gsm8k", "mmlu_pro", "hellaswag")})
    return sel, params


def _off_domain(target: dict[str, float], params: dict[str, float]) -> dict:
    """EXPLORATORY, added after E11b returned, at a reader's request: does adapted BPB on a
    DIFFERENT corpus predict the news fine-tune as well as adapted news BPB does? If it does, the
    claim is "adapted BPB on fresh text", not "on your own domain". Only the eleven-model cohort has
    all four corpora, so this runs on the nine of them that were fine-tuned -- ten in-band pairs,
    far too few for an interval, and reported as point estimates only."""
    from analyze_alignment_matrix import load_bpb_matrix
    bpb, _ = load_bpb_matrix()
    out = {"note": "exploratory, post hoc, point estimates over the in-band pairs of the "
                   "fine-tuned models that have all four corpora"}
    for corpus, tiers in bpb.items():
        for tier in ("adapted_bpb", "zero_shot_bpb"):
            vals = tiers.get(tier, {})
            ms = sorted(m for m in vals if m in target)
            right = total = 0
            for i, a in enumerate(ms):
                for b in ms[i + 1:]:
                    if max(params[a], params[b]) / min(params[a], params[b]) > BAND:
                        continue
                    if vals[a] == vals[b] or target[a] == target[b]:
                        continue
                    total += 1
                    right += (a if vals[a] < vals[b] else b) == (a if target[a] > target[b] else b)
            out[f"{corpus}__{tier}"] = {"n_models": len(ms), "n_pairs": total,
                                       "accuracy": right / total if total else None}
    return out


def build_cohort(prefix: str = "sft__") -> dict:
    """``prefix`` selects the recipe: ``sft__`` is the paper's own adaptation recipe (E11/E11b),
    ``sftB__`` the deliberately different recipe B (E11c), scored with the identical statistic."""
    from analyze_cloze_validity import _bootstrap_tables
    selectors, params = _selectors_17()
    cohort = sorted(set(json.loads((ROOT / "results" / "criterion_uncertainty.json").read_text())
                        ["cohort_17"]) - set(EXCLUDED_MULTI_GPU))
    missing = [m for m in cohort if not (CELLS / f"{prefix}{m}.json").exists()]
    out = {"cohort": cohort, "excluded_multi_gpu": list(EXCLUDED_MULTI_GPU),
           "models_missing": missing}
    if missing:
        out["verdict"] = "INCOMPLETE"
        return out
    scored = {m: score_cell(load_cell(m, prefix), "generations") for m in cohort}
    target = {m: sum(v.values()) / len(v) for m, v in scored.items()}
    out["rouge_l"] = target
    out["selector_table"] = _bootstrap_tables(selectors, target, cohort, params, BAND,
                                              compared="matched_adapted_bpb")

    # Which in-band pairs the fine-tune itself can order, and how each selector does on those.
    rng = random.Random(SEED)
    pairs = []
    for i, a in enumerate(cohort):
        for b in cohort[i + 1:]:
            if max(params[a], params[b]) / min(params[a], params[b]) > BAND:
                continue
            ids = sorted(set(scored[a]) & set(scored[b]))
            diffs = [scored[a][k] - scored[b][k] for k in ids]
            n = len(diffs)
            means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                           for _ in range(PAIR_BOOT))
            lo, hi = means[int(0.025 * PAIR_BOOT)], means[int(0.975 * PAIR_BOOT) - 1]
            winner = a if target[a] > target[b] else b
            row = {"pair": f"{a}|{b}", "winner": winner, "diff_a_minus_b": sum(diffs) / n,
                   "ci_lo": lo, "ci_hi": hi, "resolved": lo > 0 or hi < 0}
            for name, (vals, lower) in selectors.items():
                if a in vals and b in vals and vals[a] != vals[b]:
                    pick = (a if vals[a] < vals[b] else b) if lower else (
                        a if vals[a] > vals[b] else b)
                    row[f"{name}_right"] = pick == winner
            pairs.append(row)
    resolved = [p for p in pairs if p["resolved"]]
    out["pairs"] = pairs
    out["off_domain"] = _off_domain(target, params)
    out["n_in_band_pairs"] = len(pairs)
    out["n_resolved"] = len(resolved)
    out["accuracy_on_resolved"] = {
        name: (sum(p[f"{name}_right"] for p in resolved if f"{name}_right" in p)
               / max(1, sum(1 for p in resolved if f"{name}_right" in p)))
        for name in selectors}
    return out

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="fail if the artifact is stale")
    args = ap.parse_args()
    built = build()
    if args.check:
        problems = []
        if not OUTPUT_JSON.exists():
            problems.append("results/downstream.json is missing")
        elif json.dumps(json.loads(OUTPUT_JSON.read_text()), sort_keys=True) != json.dumps(
                built, sort_keys=True):
            problems.append("results/downstream.json is stale")
        for p in problems:
            print("  " + p)
        print("downstream: PROBLEMS", len(problems))
        sys.exit(1 if problems else 0)
    OUTPUT_JSON.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n")
    print("wrote", OUTPUT_JSON.relative_to(ROOT))
    for p in built["pairs"]:
        print("  %-34s %s" % (p["pair"], p["verdict"]))
        print("      ROUGE-L %s | diff %+.4f [%+.4f, %+.4f]" % (
            {k: round(v, 4) for k, v in p["rouge_l"].items()},
            p["paired_bootstrap_a_minus_b"]["mean_diff"],
            p["paired_bootstrap_a_minus_b"]["ci_lo"],
            p["paired_bootstrap_a_minus_b"]["ci_hi"]))


if __name__ == "__main__":
    main()

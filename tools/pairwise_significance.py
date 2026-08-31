#!/usr/bin/env python3
"""Selection accuracy with uncertainty computed at the MODEL level, where the dependence lives.

Each ranking induces 55 model pairs, and "does lower BPB pick the better model?" is binary per
pair, which makes pairwise accuracy a more informative summary than a rank correlation over 11
points. The trap is testing it as if those 55 pairs were 55 independent trials. They are not: they
are 55 dependent comparisons among 11 models, and an exact binomial or McNemar over them treats one
model's idiosyncrasy as up to ten separate pieces of evidence. Doing that here turned a null result
into p = 0.033; the numbers were arithmetically correct and inferentially worthless.

Everything below therefore resamples MODELS -- a cluster bootstrap over the 11 models, 20,000
draws, seeded -- and reports percentile intervals on statistics computed from whatever pairs the
resampled cohort induces. That is the same unit of analysis as the paper's existing bootstrap, and
it is the unit the design actually randomises over.

What survives this treatment and what does not is the point of the tool:

  * Selection accuracy beats chance in 23 of the 24 (corpus, tier, benchmark) cells. That is the
    claim the leaderboard makes, and it holds.
  * The CONTRAST between adaptation corpora, and between adapting and not adapting, does not
    separate from zero at this cohort size. The four-corpus pattern remains the paper's evidence
    for steering, as it was before.

    python tools/pairwise_significance.py [--check] [--reps N]
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_static_benchmarks import MODEL_IDS, ROOT, STATIC_PATH  # noqa: E402
from _artifact_check import _diff  # noqa: E402

OUTPUT_JSON = ROOT / "results" / "pairwise_significance.json"
DT = ROOT / "results" / "domain_transfer"
CORPORA = ("news", "reddit", "hackernews", "math")
TIERS = ("zero_shot", "adapted")
BENCHES = ("gsm8k", "mmlu_pro", "hellaswag")
MATCHED = {"math": "gsm8k", "news": "hellaswag"}
# Size bands for the parameter-count baseline. A cohort spanning 0.5B-35B makes "pick the bigger
# model" look strong for a trivial reason, so the comparison that matters restricts to candidates
# of comparable size -- which is the choice a practitioner is actually stuck on.
SIZE_BANDS = (3.0, 2.0, 1.5)
SEED = 20260829
REPS = 20000

# The difference-in-differences the steering claim rests on compares math adaptation against each
# general-text corpus in turn, then pools. Named here so the paper cannot quote a comparator set
# the artifact does not use.
GENERAL_CORPORA = ("news", "reddit", "hackernews")

# Resampling units for the steering DiD.
#
# The published figure resampled MODELS. That is inconsistent with this project's own narrative:
# the paper argues that GSM8K rank movement is a FAMILY property ("every Qwen holds or improves,
# both Gemmas fall"), and a bootstrap that treats family members as exchangeable draws contradicts
# the mechanism it is being used to support. Eleven models drawn from five publishers is an
# effective n far below eleven. Both units are therefore computed and both are reported; the
# family-level one is the conservative reading and the one a reviewer will ask for.
PUBLISHER = {
    "Qwen2.5-0.5B": "Qwen", "Qwen2.5-1.5B": "Qwen", "Qwen2.5-7B": "Qwen",
    "Qwen3.5-4B": "Qwen", "Qwen3.5-9B": "Qwen", "Qwen3.5-35B-MoE": "Qwen",
    "gemma-4-12B": "Gemma", "gemma-4-31B": "Gemma",
    "Llama-3.2-1B": "Llama", "LFM2.5-1.2B": "LiquidAI", "Ministral-3-14B": "Mistral",
}
# Publisher-and-generation is the finer unit: Qwen-2.5 and Qwen-3.5 are different data mixes from
# the same publisher, so collapsing them is conservative in one direction and wrong in the other.
GENERATION = dict(PUBLISHER, **{
    "Qwen2.5-0.5B": "Qwen2.5", "Qwen2.5-1.5B": "Qwen2.5", "Qwen2.5-7B": "Qwen2.5",
    "Qwen3.5-4B": "Qwen3.5", "Qwen3.5-9B": "Qwen3.5", "Qwen3.5-35B-MoE": "Qwen3.5",
})


def _load_params() -> dict:
    out = {}
    for label, model_id in MODEL_IDS.items():
        slug = model_id.replace("/", "_").replace(".", "_")
        out[label] = json.loads((DT / f"news__{slug}.json").read_text())["total_params"]
    return out


def size_baseline(bpb, params, acc, labels, bench, band=None):
    """Selection accuracy of adapted BPB against 'just pick the bigger model', on the same pairs."""
    by_bpb = by_zero = by_size = comparable = 0
    for a, b in itertools.combinations(labels, 2):
        if a == b:
            continue
        if band is not None and max(params[a], params[b]) / min(params[a], params[b]) > band:
            continue
        aa, ab = acc[a][bench], acc[b][bench]
        if aa == ab:
            continue
        better_a = aa > ab
        ba, bb = bpb[("math", "adapted", a)], bpb[("math", "adapted", b)]
        if ba != bb:
            by_bpb += (ba < bb) == better_a
        za, zb = bpb[("news", "zero_shot", a)], bpb[("news", "zero_shot", b)]
        if za != zb:
            by_zero += (za < zb) == better_a
        if params[a] != params[b]:
            by_size += (params[a] > params[b]) == better_a
        comparable += 1
    if not comparable:
        return None
    return {"pairs": comparable, "bpb": by_bpb / comparable, "zero_shot": by_zero / comparable,
            "bigger_model": by_size / comparable}


def _load_bpb() -> dict:
    out = {}
    for corpus in CORPORA:
        for label, model_id in MODEL_IDS.items():
            slug = model_id.replace("/", "_").replace(".", "_")
            d = json.loads((DT / f"{corpus}__{slug}.json").read_text())
            out[(corpus, "zero_shot", label)] = d["zero_shot_bpb"]
            out[(corpus, "adapted", label)] = d["adapted_bpb"]
    return out


def pairwise_accuracy(bpb, acc, models, corpus, tier, bench):
    """Share of comparable model pairs whose BPB ordering matches the benchmark ordering."""
    correct = comparable = 0
    for a, b in itertools.combinations(models, 2):
        if a == b:                       # a bootstrap draw can repeat a model
            continue
        ba, bb = bpb[(corpus, tier, a)], bpb[(corpus, tier, b)]
        aa, ab = acc[a][bench], acc[b][bench]
        if ba == bb or aa == ab:         # ties carry no ordering information
            continue
        comparable += 1
        correct += (ba < bb) == (aa > ab)
    return correct / comparable if comparable else None


def cluster_ci(stat, labels, reps, rng):
    """Percentile interval from resampling MODELS with replacement."""
    draws = []
    for _ in range(reps):
        sample = [rng.choice(labels) for _ in labels]
        value = stat(sample)
        if value is not None:
            draws.append(value)
    draws.sort()
    return {"lo": draws[int(0.025 * len(draws))], "hi": draws[int(0.975 * len(draws))],
            "n_draws": len(draws)}


def steering_did(bpb, acc, models, general: str):
    """Joint movement: (GSM8K minus HellaSwag) under math adaptation, minus the same under `general`.

    The steering hypothesis predicts a *pair* of movements -- toward the matched benchmark and away
    from the other -- so a difference in differences is better powered than either single contrast,
    and it is the statistic the paper leads with.
    """
    def pa(corpus, bench):
        return pairwise_accuracy(bpb, acc, models, corpus, "adapted", bench)
    parts = [pa("math", "gsm8k"), pa("math", "hellaswag"),
             pa(general, "gsm8k"), pa(general, "hellaswag")]
    if any(p is None for p in parts):
        return None
    return (parts[0] - parts[1]) - (parts[2] - parts[3])


def pooled_did(bpb, acc, models):
    values = [steering_did(bpb, acc, models, g) for g in GENERAL_CORPORA]
    return None if any(v is None for v in values) else sum(values) / len(values)


def sign_split(stat, units: dict, reps: int, rng) -> dict:
    """Resample `units` (a {label: cluster} map) with replacement; report the sign of the statistic.

    A percentile interval alone hides what matters here: at eleven models the statistic is discrete
    and a large share of draws land exactly on zero. Reporting the three-way split makes the
    one-sided p depend on a stated tie convention rather than on an unstated one.
    """
    clusters: dict[str, list[str]] = {}
    for label, cluster in units.items():
        clusters.setdefault(cluster, []).append(label)
    keys = sorted(clusters)
    draws = []
    for _ in range(reps):
        cohort = [m for k in (rng.choice(keys) for _ in keys) for m in clusters[k]]
        value = stat(cohort)
        if value is not None:
            draws.append(value)
    draws.sort()
    n = len(draws)
    positive = sum(1 for d in draws if d > 0)
    zero = sum(1 for d in draws if d == 0)
    negative = n - positive - zero
    return {
        "n_clusters": len(keys),
        "clusters": {k: sorted(v) for k, v in sorted(clusters.items())},
        "n_draws": n,
        "frac_positive": positive / n,
        "frac_zero": zero / n,
        "frac_negative": negative / n,
        "ci_lo": draws[int(0.025 * n)],
        "ci_hi": draws[int(0.975 * n)],
        # Three tie conventions, all reported, none chosen for us by the arithmetic.
        "p_one_sided_ties_against": (zero + negative) / n,
        "p_one_sided_mid_p": (zero / 2 + negative) / n,
        "p_one_sided_ties_ignored": negative / n,
    }


def build_steering_did(bpb, acc, reps: int) -> dict:
    """The DiD, its per-corpus values, and its stability under both resampling units.

    Takes its own seeded generator rather than sharing the caller's. Sharing one stream makes every
    statistic's draws depend on how many draws the statistics before it happened to consume, so
    adding an unrelated cell upstream silently moves this one -- which is how the published 93.5%
    became 93.8% the first time this was wired in.
    """
    rng = random.Random(SEED)
    labels = list(MODEL_IDS)
    per_corpus = {g: steering_did(bpb, acc, labels, g) for g in GENERAL_CORPORA}
    stat = lambda m: pooled_did(bpb, acc, m)  # noqa: E731

    identity = {m: m for m in labels}
    resampling = {
        "model": sign_split(stat, identity, reps, rng),
        "publisher_family": sign_split(stat, PUBLISHER, reps, rng),
        "publisher_and_generation": sign_split(stat, GENERATION, reps, rng),
    }

    # Leave-one-family-out. The bootstrap says whether the sign is robust; this says which family
    # carries the magnitude, which is the question a reviewer asks next.
    jackknife = {}
    for family in sorted(set(PUBLISHER.values())):
        kept = [m for m in labels if PUBLISHER[m] != family]
        jackknife[family] = {"n_models_kept": len(kept), "pooled_did": pooled_did(bpb, acc, kept)}
    carried = min(jackknife, key=lambda f: jackknife[f]["pooled_did"])

    return {
        "statistic": "(gsm8k - hellaswag selection accuracy | math-adapted) minus the same "
                     "| general-text-adapted, pooled over news, reddit, hackernews",
        "per_corpus": per_corpus,
        "pooled": pooled_did(bpb, acc, labels),
        "resampling": resampling,
        "leave_one_family_out": jackknife,
        "most_load_bearing_family": carried,
        "note": (
            "The model-level split is the published figure. The family-level splits are the "
            "conservative reading and the one this project's own family narrative implies: models "
            "within a publisher are not exchangeable draws. Direction survives both -- the "
            "publisher-level resampling produces no negative draws at all -- while the magnitude "
            "does not, and the leave-one-family-out column shows it is concentrated in one family."
        ),
    }


def build(reps: int = REPS) -> dict:
    acc = json.loads(STATIC_PATH.read_text())
    bpb = _load_bpb()
    labels = list(MODEL_IDS)
    rng = random.Random(SEED)

    selection = {}
    for corpus in CORPORA:
        for tier in TIERS:
            for bench in BENCHES:
                obs = pairwise_accuracy(bpb, acc, labels, corpus, tier, bench)
                ci = cluster_ci(lambda m, c=corpus, t=tier, b=bench:
                                pairwise_accuracy(bpb, acc, m, c, t, b), labels, reps, rng)
                selection[f"{corpus}__{tier}__{bench}"] = {
                    "accuracy": obs, "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                    "beats_chance": ci["lo"] > 0.5,
                }

    def contrast(f):
        obs = f(labels)
        ci = cluster_ci(f, labels, reps, rng)
        return {"estimate": obs, "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                "excludes_zero": ci["lo"] > 0 or ci["hi"] < 0}

    steering = {}
    for bench in BENCHES:
        def diff(m, b=bench):
            x = pairwise_accuracy(bpb, acc, m, "math", "adapted", b)
            y = pairwise_accuracy(bpb, acc, m, "news", "adapted", b)
            return None if x is None or y is None else x - y
        steering[bench] = contrast(diff)

    beats_nothing = {}
    for corpus, bench in MATCHED.items():
        def diff(m, c=corpus, b=bench):
            x = pairwise_accuracy(bpb, acc, m, c, "adapted", b)
            y = pairwise_accuracy(bpb, acc, m, c, "zero_shot", b)
            return None if x is None or y is None else x - y
        beats_nothing[f"{corpus}__{bench}"] = contrast(diff)

    params = _load_params()
    baseline = {}
    for band in (None,) + SIZE_BANDS:
        key = "all_pairs" if band is None else f"within_{band:g}x"
        baseline[key] = {b: size_baseline(bpb, params, acc, labels, b, band) for b in BENCHES}

    return {
        "schema_version": 4,
        "size_baseline": baseline,
        "parameter_counts": params,
        "method": "cluster bootstrap over the 11 models, %d draws, seed %d; percentile intervals. "
                  "Pairs are NOT resampled -- they are dependent within a model." % (reps, SEED),
        "selection_accuracy": selection,
        "cells_beating_chance": sum(1 for v in selection.values() if v["beats_chance"]),
        "cells_total": len(selection),
        "corpus_steering_contrast": steering,
        "steering_difference_in_differences": build_steering_did(bpb, acc, reps),
        "adaptation_vs_doing_nothing": beats_nothing,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()
    report = build(args.reps)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT_JSON.exists():
            raise SystemExit(f"{OUTPUT_JSON.relative_to(ROOT)} is missing")
        drift = _diff(json.loads(OUTPUT_JSON.read_text()), report)
        if drift:
            raise SystemExit(f"{OUTPUT_JSON.relative_to(ROOT)} is stale: " + "; ".join(drift[:5]))
        print("pairwise significance is up to date")
        return
    OUTPUT_JSON.write_text(payload)

    print("Selection accuracy, 95%% CI from resampling the 11 models "
          "(> 0.5 means the metric picks the better model)")
    for name, r in report["selection_accuracy"].items():
        flag = "beats chance" if r["beats_chance"] else "includes chance"
        print(f"   {name:<32}{r['accuracy']:.3f}   [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]   {flag}")
    print(f"\n   {report['cells_beating_chance']}/{report['cells_total']} cells beat chance.")

    print("\nContrasts, same bootstrap. These are the comparisons this cohort cannot settle:")
    for bench, r in report["corpus_steering_contrast"].items():
        print(f"   math- minus news-adapted, {bench:<10}{r['estimate']:+.3f}   "
              f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]   "
              f"{'excludes 0' if r['excludes_zero'] else 'includes 0'}")
    for name, r in report["adaptation_vs_doing_nothing"].items():
        print(f"   adapted minus zero-shot, {name:<11}{r['estimate']:+.3f}   "
              f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]   "
              f"{'excludes 0' if r['excludes_zero'] else 'includes 0'}")
    print(f"\nwrote {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

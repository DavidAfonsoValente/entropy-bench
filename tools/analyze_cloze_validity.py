#!/usr/bin/env python3
"""E4: does prediction loss on your own text predict in-domain performance better than benchmarks?

This is the experiment that answers the circularity objection. Everywhere else the paper validates
BPB against MMLU-Pro / HellaSwag / GSM8K, which is the very thing a reviewer objects to: those
suites are the ones whose preparability motivates the work, so agreement with them cannot establish
that BPB is a *better* instrument -- only that it recovers the same ordering.

The in-domain cloze set (tools/build_domain_cloze.py) is a criterion none of those suites supply:
in-domain, from the pipeline's own held-out split, postdating every release, and scored by greedy
generation with string match rather than by log-likelihood. Against it, each candidate selector can
be scored on equal terms:

    zero-shot news BPB   (the free tier we recommend)
    news-adapted BPB     (the paid tier)
    GSM8K / MMLU-Pro / HellaSwag  (what a practitioner would read instead)
    parameter count      (the baseline that beats us over a wide size range)

The statistic is pairwise selection accuracy -- of all model pairs, how often does the selector put
the one that is actually better on the target domain first -- because that is the decision a
practitioner makes. Intervals come from the same model-level cluster bootstrap used elsewhere;
pairs are NOT resampled, since 55 pairs from 11 models are not 55 independent trials.

    python tools/analyze_cloze_validity.py --results <dir> [--check]

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _artifact_check import _diff  # noqa: E402
from analyze_alignment_matrix import load_bpb_matrix  # noqa: E402
from analyze_static_benchmarks import (  # noqa: E402
    BENCHMARKS, MODEL_IDS, ROOT, load_raw_static,
)
from pairwise_significance import SIZE_BANDS, _load_params  # noqa: E402

OUTPUT_JSON = ROOT / "results" / "cloze_validity.json"
SCHEMA_VERSION = 5
# A resampled cohort can contain no in-band pair at all, and such draws carry no statistic. Skipping
# them conditions the bootstrap distribution on non-degeneracy, which is a real change to what the
# interval covers rather than a neutral repair -- in the limit (one in-band pair, always ordered the
# same way) it would report [1.000, 1.000] off a single pair. Above this fraction the tool refuses to
# publish an interval instead of publishing a conditioned one. Declared here, before the data: the
# observed rate is ~0.3%, so nothing in this study is near the threshold.
MAX_DEGENERATE_FRAC = 0.05
SEED = 20260831
REPS = 20000
# The headline metric. Base models have no stop convention and will run on past the answer, so a
# strict prefix match would partly measure stopping behaviour rather than knowledge; lenient is the
# convention-independent read. Both are recorded and the artifact carries the strict table too.
HEADLINE_METRIC = "lenient_accuracy"
LENIENT_NOTE = "short window"
# The size band the recommendation is scoped to. Over the full 0.5B-35B range this study already
# concedes that "take the bigger model" is as good a rule as any selector, so a full-range
# comparison answers a question no practitioner asks. The decision they are actually stuck on is
# between candidates of comparable size. 2x is NOT chosen here to suit this criterion: it is the
# band pairwise_significance.SIZE_BANDS already used for the benchmark targets, and the one the
# operating rule in docs/POSITIONING.md was written around, both fixed before the cloze set
# existed. The other bands ship in the artifact so the choice is auditable rather than trusted.
PRIMARY_BAND = 2.0
# The selector every comparison is stated about, fixed BY DESIGN rather than by picking the winner.
# The criterion is a news cloze set, so the domain-matched paid tier is news-adapted BPB; choosing
# it with max() over the observed accuracies would make every reported lead a winner's-curse
# artifact, since the same numbers would have selected the comparand. `best_bpb_selector` remains in
# the artifact as a descriptive field and is not what any claim rests on.
COMPARED_SELECTOR = "news_adapted_bpb"


def load_cloze(results_dir: Path, metric: str, pattern: str = "cloze__*.json") -> dict[str, float]:
    """Per-model scores for one metric. ``pattern`` selects the run: a second criterion is written
    by the runner under its own prefix (``cloze_math__*.json``) so the two cannot be mixed."""
    out = {}
    for path in sorted(results_dir.glob(pattern)):
        row = json.loads(path.read_text())
        label = row["label"]
        if label not in MODEL_IDS:
            raise SystemExit(f"{path.name}: unknown model label {label!r}")
        out[label] = row[metric]
    if not out:
        raise SystemExit(f"no {pattern} in {results_dir}")
    return out


def _score(selector: dict[str, float], target: dict[str, float], models: list[str],
           lower_is_better: bool, params: dict[str, float] | None = None,
           band: float | None = None) -> tuple[float | None, int]:
    """Pairwise selection accuracy and the number of pairs it was computed over.

    ``band`` restricts to pairs whose parameter counts are within that factor of each other, which
    is the only regime where a selector has to beat "take the bigger model" to be worth running.
    """
    if band is not None:
        if params is None:
            raise ValueError("band restriction needs parameter counts")
        if band <= 0:
            raise ValueError(f"band must be positive, got {band}")
        missing = [m for m in models if params.get(m, 0) <= 0]
        if missing:
            raise ValueError(f"models with missing or non-positive parameter counts: {missing}")
    correct = comparable = 0
    for a, b in combinations(models, 2):
        if a == b:
            continue
        if a not in selector or b not in selector or a not in target or b not in target:
            continue
        if band is not None and max(params[a], params[b]) / min(params[a], params[b]) > band:
            continue
        if selector[a] == selector[b] or target[a] == target[b]:
            continue          # ties carry no ordering information
        picks_a = selector[a] < selector[b] if lower_is_better else selector[a] > selector[b]
        truth_a = target[a] > target[b]
        comparable += 1
        correct += picks_a == truth_a
    return (correct / comparable if comparable else None), comparable


def pairwise_accuracy(selector: dict[str, float], target: dict[str, float],
                      models: list[str], lower_is_better: bool) -> float | None:
    """How often the selector orders a pair the way the target actually does."""
    return _score(selector, target, models, lower_is_better)[0]


def _quantile(sorted_draws: list[float], q: float) -> float:
    """Percentile interval endpoint, matching pairwise_significance.cluster_ci exactly.

    This is the lower order statistic rather than an interpolated quantile, which biases both
    endpoints marginally upward. It is kept because every interval in this project uses it and a
    silent change of convention would make cross-table comparisons wrong in a way nobody would see.
    """
    return sorted_draws[int(q * len(sorted_draws))]


def _bootstrap_tables(selectors: dict, cloze: dict[str, float], models: list[str],
                      params: dict[str, float] | None = None,
                      band: float | None = None,
                      compared: str | None = None) -> dict:
    """Marginal intervals and PAIRED selector differences from one shared resample stream.

    Two things here are deliberate and neither is cosmetic.

    **One stream, shared by every selector.** Drawing a separate cohort per selector gives valid
    marginal intervals and nothing else: the difference between two selectors cannot be read off
    two independently resampled intervals, and overlapping intervals do not imply an unresolved
    difference. Every selector is scored on the *same* resampled cohort, so the distribution of
    their difference is available directly -- which is what the claims in the paper are about.

    **One eligibility mask, shared by every selector.** A draw counts only if every selector
    produced a statistic on it. Otherwise selectors would be compared over different pair sets, and
    a verdict could come out true because one selector got an easier subset rather than a better
    ordering.
    """
    rng = random.Random(SEED)
    names = list(selectors)
    obs = {n: _score(v, cloze, models, lb, params, band) for n, (v, lb) in selectors.items()}
    best_bpb = compared or COMPARED_SELECTOR

    have_static = all(b in selectors for b in SELECTOR_STATIC)
    draws: dict[str, list[float]] = {n: [] for n in names}
    diffs: dict[str, list[float]] = {"bpb_minus_parameter_count": []}
    if have_static:
        diffs["bpb_minus_best_benchmark"] = []
    # The adapted-vs-zero-shot contrast: does adapting change the answer, or is the free tier
    # already as good? POST-HOC -- the preregistration named the size and benchmark contrasts and
    # not this one, so it is reported as exploratory. It rides the same resample stream and draws
    # no extra randomness, so adding it leaves every other number bit-identical.
    zero_key = next((k for k in ("zero_shot_bpb", "zero_shot_news_bpb") if k in selectors), None)
    if zero_key is not None and zero_key != best_bpb:
        diffs["bpb_minus_zero_shot"] = []
    degenerate = 0
    for _ in range(REPS):
        sample = [rng.choice(models) for _ in models]
        vals = {n: _score(v, cloze, sample, lb, params, band)[0]
                for n, (v, lb) in selectors.items()}
        if any(v is None for v in vals.values()):
            degenerate += 1
            continue
        for n in names:
            draws[n].append(vals[n])
        if obs[best_bpb][0] is not None:
            # The benchmark contrast only exists when the benchmark column does. A cohort scored on
            # BPB and the criterion alone is a legitimate table, not a broken one.
            if have_static:
                diffs["bpb_minus_best_benchmark"].append(
                    vals[best_bpb] - max(vals[b] for b in SELECTOR_STATIC))
            diffs["bpb_minus_parameter_count"].append(vals[best_bpb] - vals["parameter_count"])
            if "bpb_minus_zero_shot" in diffs:
                diffs["bpb_minus_zero_shot"].append(vals[best_bpb] - vals[zero_key])

    degenerate_frac = degenerate / REPS
    usable = degenerate_frac <= MAX_DEGENERATE_FRAC

    table = {}
    for name in names:
        d = sorted(draws[name])
        acc, n_pairs = obs[name]
        lo = hi = None
        if usable and d:
            lo, hi = _quantile(d, 0.025), _quantile(d, 0.975)
        # A selector that orders EVERY in-band pair correctly makes the percentile bootstrap
        # collapse: every resample scores 1.000, so the interval is [1.000, 1.000] and the naive
        # "lower bound above 0.5" test passes on what is really a point estimate at the boundary
        # over ten pairs. That is false precision, not evidence, and it is the exact failure an
        # earlier review predicted in the abstract. Such intervals are marked and excluded from
        # every clears-chance claim.
        degenerate = lo is not None and hi is not None and lo == hi
        table[name] = {
            "pairwise_accuracy": acc,
            "ci_lo": lo, "ci_hi": hi,
            "interval_is_degenerate": degenerate,
            "beats_chance": lo is not None and lo > 0.5 and not degenerate,
            "beats_chance_including_degenerate": lo is not None and lo > 0.5,
            "n_pairs": n_pairs,
            "n_draws": len(d),
        }

    paired = {}
    for key, vals in diffs.items():
        d = sorted(vals)
        lo = hi = None
        if usable and d:
            lo, hi = _quantile(d, 0.025), _quantile(d, 0.975)
        paired[key] = {
            "observed": (table[best_bpb]["pairwise_accuracy"] - max(
                table[b]["pairwise_accuracy"] for b in SELECTOR_STATIC))
            if key == "bpb_minus_best_benchmark"
            else (table[best_bpb]["pairwise_accuracy"] - table[zero_key]["pairwise_accuracy"])
            if key == "bpb_minus_zero_shot"
            else (table[best_bpb]["pairwise_accuracy"]
                  - table["parameter_count"]["pairwise_accuracy"]),
            "preregistered": key != "bpb_minus_zero_shot",
            "ci_lo": lo, "ci_hi": hi,
            # The interval, not the point estimate, is what may be asserted. A point estimate that
            # is positive while the interval spans zero is a direction, not a result. The four
            # fields below keep apart what a single `separates_from_zero` flag conflated: an
            # interval containing zero, one lying entirely BELOW zero, and no interval at all --
            # "does not separate from zero" would otherwise be satisfied by a strongly negative
            # result. Same vocabulary as analyze_cloze_steering, so the two agree.
            "separates_from_zero": lo is not None and lo > 0,
            "inference_available": lo is not None and hi is not None,
            "ci_excludes_zero": lo is not None and hi is not None and (lo > 0 or hi < 0),
            "supports_positive_effect": lo is not None and lo > 0,
            "supports_negative_effect": hi is not None and hi < 0,
            "ci_contains_zero": lo is not None and hi is not None and lo <= 0 <= hi,
            "frac_draws_positive": (sum(1 for v in d if v > 0) / len(d)) if d else None,
            "n_draws": len(d),
        }

    return {
        "selectors": table,
        "paired_differences": paired,
        "compared_selector": best_bpb,
        # Every selector scored the same pair set, so a verdict cannot come from an easier subset.
        "common_pair_mask": len({row["n_pairs"] for row in table.values()}) == 1,
        "degenerate_draw_frac": degenerate_frac,
        "inference_available": usable,
    }


SELECTOR_BPB = ("zero_shot_news_bpb", "news_adapted_bpb", "math_adapted_bpb")
SELECTOR_STATIC = ("gsm8k", "mmlu_pro", "hellaswag")


def _verdicts(table: dict, compared: str | None = None,
              bpb_keys: tuple[str, ...] | None = None) -> dict:
    if any(row["pairwise_accuracy"] is None for row in table.values()):
        # A selector with no comparable pair cannot be ranked, and guessing a default here is how
        # a band with too little data would quietly acquire a verdict.
        return {"best_bpb_selector": None, "best_static_benchmark": None,
                "bpb_beats_every_static_benchmark": None, "bpb_beats_parameter_count": None}
    # Descriptive only -- which tier happened to do best. The verdicts below are about
    # COMPARED_SELECTOR, fixed in advance, for the reason given at its definition.
    bpb_keys = bpb_keys or SELECTOR_BPB
    best_bpb = max(bpb_keys, key=lambda k: table[k]["pairwise_accuracy"])
    best_benchmark = max(SELECTOR_STATIC, key=lambda k: table[k]["pairwise_accuracy"])
    claimed = compared or COMPARED_SELECTOR
    # max() silently returns the first key on a tie, which would let an arbitrary label be reported
    # as "best". Ties are recorded so a reader can see when the label is not meaningful.
    def _tied(keys, winner):
        return sorted(k for k in keys
                      if table[k]["pairwise_accuracy"] == table[winner]["pairwise_accuracy"])
    return {
        # EVERY field below compares POINT ESTIMATES and nothing else. A `true` here means one
        # noisy number exceeded another; whether the lead is resolved is a different question,
        # answered only by `paired_differences.*.separates_from_zero`, which is computed from
        # shared resampled cohorts. Do not quote one without the other.
        "comparison_basis": "point estimates only; see paired_differences for whether a lead is "
                            "distinguishable from zero",
        "best_bpb_selector": best_bpb,
        "best_static_benchmark": best_benchmark,
        "best_bpb_tied_with": _tied(bpb_keys, best_bpb),
        "best_static_benchmark_tied_with": _tied(SELECTOR_STATIC, best_benchmark),
        "claimed_selector": claimed,
        "bpb_beats_every_static_benchmark": all(
            table[claimed]["pairwise_accuracy"] > table[b]["pairwise_accuracy"]
            for b in SELECTOR_STATIC),
        "bpb_beats_parameter_count":
            table[claimed]["pairwise_accuracy"] > table["parameter_count"]["pairwise_accuracy"],
    }


def _band_pair_count(models: list[str], params: dict[str, float], band: float) -> int:
    """Model pairs within `band` of each other in parameters. A property of the cohort alone."""
    return sum(1 for a, b in combinations(models, 2)
               if max(params[a], params[b]) / min(params[a], params[b]) <= band)


def _band_analysis(selectors: dict, cloze: dict[str, float], cloze_strict: dict[str, float],
                   models: list[str], params: dict[str, float]) -> dict:
    """The same comparison restricted to pairs of comparable size.

    This is the regime the recommendation is about. Unrestricted, the cohort spans 0.5B to 35B and
    "take the bigger model" is already a good rule -- a selector that wins there has not shown it is
    worth running. Every band in SIZE_BANDS is reported so that fixing on one is auditable.
    """
    out = {}
    for band in SIZE_BANDS:
        boot = _bootstrap_tables(selectors, cloze, models, params, band)
        boot_strict = _bootstrap_tables(selectors, cloze_strict, models, params, band)
        table, table_strict = boot["selectors"], boot_strict["selectors"]
        verdict, verdict_strict = _verdicts(table), _verdicts(table_strict)
        out[f"within_{band:g}x"] = {
            "band": band,
            "is_primary": band == PRIMARY_BAND,
            # Counted from the cohort and the parameter counts, NOT from a selector's surviving
            # denominator: a selector drops pairs it ties on, so its count answers a different
            # question and would be wrong for the strict table whenever the two differ.
            "n_pairs": _band_pair_count(models, params, band),
            "selectors": table,
            "selectors_strict": table_strict,
            "paired_differences": boot["paired_differences"],
            "paired_differences_strict": boot_strict["paired_differences"],
            "common_pair_mask": boot["common_pair_mask"] and boot_strict["common_pair_mask"],
            "degenerate_draw_frac": max(boot["degenerate_draw_frac"],
                                        boot_strict["degenerate_draw_frac"]),
            "inference_available": boot["inference_available"] and boot_strict["inference_available"],
            **{k: verdict[k] for k in ("best_bpb_selector", "best_static_benchmark",
                                       "bpb_beats_every_static_benchmark",
                                       "bpb_beats_parameter_count")},
            "strict": verdict_strict,
            "claim_holds_under_both_conventions": {
                k: (None if verdict[k] is None or verdict_strict[k] is None
                    else bool(verdict[k] and verdict_strict[k]))
                for k in ("bpb_beats_every_static_benchmark", "bpb_beats_parameter_count")},
        }
    return out


def build_analysis(results_dir: Path) -> dict:
    cloze = load_cloze(results_dir, HEADLINE_METRIC)
    cloze_strict = load_cloze(results_dir, "strict_accuracy")
    bpb, _ = load_bpb_matrix()
    params = _load_params()
    acc = {b: {m: load_raw_static(m, b)[0] for m in MODEL_IDS} for b in BENCHMARKS}

    models = sorted(cloze)          # only models that actually produced a cloze score
    selectors = {
        "zero_shot_news_bpb": (bpb["news"]["zero_shot_bpb"], True),
        "news_adapted_bpb": (bpb["news"]["adapted_bpb"], True),
        "math_adapted_bpb": (bpb["math"]["adapted_bpb"], True),
        "gsm8k": (acc["gsm8k"], False),
        "mmlu_pro": (acc["mmlu_pro"], False),
        "hellaswag": (acc["hellaswag"], False),
        "parameter_count": (params, False),
    }

    boot = _bootstrap_tables(selectors, cloze, models)
    # The SAME analysis against the strict target. This is not a footnote: strict and lenient
    # disagree on the verdict, so reporting only the headline metric would be reporting the
    # convention that happens to flatter the instrument. Both tables ship.
    boot_strict = _bootstrap_tables(selectors, cloze_strict, models)
    table, table_strict = boot["selectors"], boot_strict["selectors"]
    verdict = _verdicts(table)
    verdict_strict = _verdicts(table_strict)
    best_bpb = verdict["best_bpb_selector"]
    best_benchmark = verdict["best_static_benchmark"]

    return {
        "schema_version": SCHEMA_VERSION,
        "question": "Which selector best predicts in-domain cloze performance?",
        "headline_metric": HEADLINE_METRIC,
        "criterion": "in-domain cloze, greedy generation + string match, held-out split, "
                     "postdates every model release; NOT a log-likelihood metric",
        "method": f"pairwise selection accuracy; cluster bootstrap over models, {REPS} draws, "
                  f"seed {SEED}; pairs are NOT resampled. One shared resample stream and one shared "
                  f"eligibility mask across selectors, so paired differences are readable directly. "
                  f"The interval covers variability in WHICH MODELS are in the cohort only: the "
                  f"cloze accuracies and benchmark scores are treated as exact, so this is not a "
                  f"general uncertainty interval for selector validity.",
        "n_models_scored": len(models),
        "models_scored": models,
        "models_missing": sorted(set(MODEL_IDS) - set(models)),
        "cloze_accuracy": {m: cloze[m] for m in models},
        "cloze_accuracy_strict": {m: cloze_strict[m] for m in models},
        "selectors": table,
        "selectors_strict": table_strict,
        # Differences between selectors, from the SAME resampled cohorts. Marginal intervals cannot
        # be differenced by eye; these can be read directly.
        "paired_differences": boot["paired_differences"],
        "paired_differences_strict": boot_strict["paired_differences"],
        "common_pair_mask": boot["common_pair_mask"] and boot_strict["common_pair_mask"],
        "degenerate_draw_frac": max(boot["degenerate_draw_frac"],
                                    boot_strict["degenerate_draw_frac"]),
        "inference_available": boot["inference_available"] and boot_strict["inference_available"],
        "comparison_basis": verdict["comparison_basis"],
        "best_bpb_selector": best_bpb,
        "best_static_benchmark": best_benchmark,
        "best_bpb_tied_with": verdict["best_bpb_tied_with"],
        "best_static_benchmark_tied_with": verdict["best_static_benchmark_tied_with"],
        "bpb_beats_every_static_benchmark": verdict["bpb_beats_every_static_benchmark"],
        "bpb_beats_parameter_count": verdict["bpb_beats_parameter_count"],
        "strict": verdict_strict,
        # Two fields, because agreement is NOT survival: the verdicts agree just as well when both
        # conventions REJECT a claim. Only the second field says whether a claim may be asserted.
        # Neither looks at which selector won, at effect size, or at how far the intervals overlap;
        # they are thresholded comparisons and nothing more.
        "verdicts_agree_across_conventions": (
            verdict["bpb_beats_every_static_benchmark"]
            == verdict_strict["bpb_beats_every_static_benchmark"]
            and verdict["bpb_beats_parameter_count"]
            == verdict_strict["bpb_beats_parameter_count"]),
        "claim_holds_under_both_conventions": {
            k: bool(verdict[k] and verdict_strict[k])
            for k in ("bpb_beats_every_static_benchmark", "bpb_beats_parameter_count")},
        "primary_band": PRIMARY_BAND,
        "bands": _band_analysis(selectors, cloze, cloze_strict, models, params),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "cloze")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    report = build_analysis(a.results)

    if a.check:
        if not OUTPUT_JSON.exists():
            raise SystemExit(f"missing {OUTPUT_JSON.relative_to(ROOT)}; run without --check")
        diffs = _diff(json.loads(OUTPUT_JSON.read_text()), report)
        if diffs:
            for d in diffs[:20]:
                print("  " + d)
            raise SystemExit(f"{OUTPUT_JSON.relative_to(ROOT)} is stale ({len(diffs)} diffs)")
        print("cloze validity artifact is up to date")
        return

    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    print(f"In-domain cloze, {report['n_models_scored']} models "
          f"({', '.join(report['models_missing']) or 'none'} missing)\n")
    print(f"{'selector':<22}{'pairwise acc':>13}{'95% CI':>20}")
    for name, row in sorted(report["selectors"].items(),
                            key=lambda kv: -kv[1]["pairwise_accuracy"]):
        print(f"{name:<22}{row['pairwise_accuracy']:>13.3f}"
              f"   [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]")
    print(f"\nbest BPB tier: {report['best_bpb_selector']}   "
          f"best static benchmark: {report['best_static_benchmark']}")
    print(f"BPB beats every static benchmark: {report['bpb_beats_every_static_benchmark']}")
    print(f"BPB beats parameter count:       {report['bpb_beats_parameter_count']}")

    st = report["strict"]
    print(f"\nsame analysis, STRICT scoring (prefix match instead of a {LENIENT_NOTE}):")
    for name, row in sorted(report["selectors_strict"].items(),
                            key=lambda kv: -kv[1]["pairwise_accuracy"]):
        print(f"{name:<22}{row['pairwise_accuracy']:>13.3f}"
              f"   [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]")
    print(f"best BPB tier: {st['best_bpb_selector']}   "
          f"best static benchmark: {st['best_static_benchmark']}")
    print(f"BPB beats every static benchmark: {st['bpb_beats_every_static_benchmark']}")
    print(f"BPB beats parameter count:       {st['bpb_beats_parameter_count']}")
    print("\npaired differences from SHARED resampled cohorts "
          "(the only inferential comparison here):")
    for conv, key in (("lenient", "paired_differences"), ("strict", "paired_differences_strict")):
        for name, d in report[key].items():
            ci = (f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]" if d["ci_lo"] is not None else "n/a")
            print(f"  {conv:<8}{name:<28}{d['observed']:+.3f}  {ci}"
                  f"  separates={d['separates_from_zero']}"
                  f"  positive in {d['frac_draws_positive']:.1%} of draws")

    print(f"\nverdicts agree across conventions: {report['verdicts_agree_across_conventions']}")
    print("claims that hold under BOTH conventions (the only ones assertable):")
    for k, v in report["claim_holds_under_both_conventions"].items():
        print(f"  {k:<34} {v}")
    print(f"\nRestricted to pairs within a size factor -- the regime the recommendation is about.")
    for key, blk in report["bands"].items():
        mark = "  <- primary" if blk["is_primary"] else ""
        print(f"\n{key} ({blk['n_pairs']} pairs){mark}")
        for label, tkey in (("lenient", "selectors"), ("strict", "selectors_strict")):
            ranked = sorted(blk[tkey].items(),
                            key=lambda kv: -(kv[1]["pairwise_accuracy"] or 0.0))
            row = "  ".join(f"{n}={r['pairwise_accuracy']:.3f}"
                            for n, r in ranked if r["pairwise_accuracy"] is not None)
            print(f"  {label:<8}{row}")
        print(f"  holds under both: {blk['claim_holds_under_both_conventions']}")

    print(f"\nwrote {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

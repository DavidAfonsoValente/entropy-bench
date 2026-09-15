#!/usr/bin/env python3
"""The steering claim, tested entirely off-benchmark: a 2x2 of adaptation corpus x criterion.

Section 5.5 of the paper shows that adapting on mathematical prose rotates the ranking toward
GSM8K. That is measured against a public benchmark, which is the circularity the cloze criterion
exists to escape. This tool asks the same question with no benchmark anywhere in it:

                        | news cloze      | math cloze
    news-adapted BPB    | a               | b
    math-adapted BPB    | c               | d

If adaptation steers the ruler toward the corpus it was run on, then news-adapted BPB should
predict news cloze better than math-adapted BPB does (a > c), and math-adapted BPB should predict
math cloze better than news-adapted BPB does (d > b). The single number for that is the
difference-in-differences:

    DiD = (a - c) - (b - d)

which is positive when each selector does relatively better on its own corpus's criterion. Reading
only one arm would confound steering with "news-adapted BPB is simply the better selector", which
is why the interaction rather than either main effect is the statistic reported.

**Uncertainty is bootstrapped over MODELS, on one shared resample stream**, exactly as in
analyze_cloze_validity.py and for the same reason: pairs sharing a model are not independent, and
two selectors compared from separately resampled cohorts cannot be differenced. Every quantity
below -- all four cells and the interaction -- comes from the same draws.

    python tools/analyze_cloze_steering.py [--check]

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _artifact_check import _diff  # noqa: E402
from itertools import combinations  # noqa: E402

from analyze_cloze_validity import (  # noqa: E402
    MAX_DEGENERATE_FRAC, PRIMARY_BAND, REPS, ROOT, SEED, SIZE_BANDS,
    _band_pair_count, _load_params, _quantile, load_bpb_matrix, load_cloze,
)

OUTPUT_JSON = ROOT / "results" / "cloze_steering.json"
SCHEMA_VERSION = 2
# The estimand declared as primary, so that reporting four size scopes x two conventions is a
# sensitivity analysis rather than a search. Bands are nested and the two conventions score the
# same generations, so the sixteen arm contrasts are ONE observation under sixteen specifications;
# they cannot support a sign test and no combined significance is computed anywhere in this file.
PRIMARY_ESTIMAND = "difference_in_differences over all pairs, reported under both conventions"
CRITERIA = ("news", "math")
# The two selectors under test. Each is the domain-matched tier for one of the two criteria, fixed
# by design; nothing here is chosen by looking at which scored higher.
SELECTORS = ("news_adapted_bpb", "math_adapted_bpb")


def _eligible_pairs(bpb, cloze, models, params, metric, band):
    """Model pairs usable by ALL FOUR cells, which is the only set a $2\\times2$ may be built on.

    Scoring each cell on whatever pairs it happens to survive is a composition confound: a tie in
    one criterion drops a pair from two cells and not the others, and the interaction then
    subtracts averages taken over different dyads. That is not repaired by sharing bootstrap
    draws. In the shipped data it was real -- the mathematics criterion ties one pair, so the four
    cells ran on 55, 54, 55 and 54 pairs before this was fixed.
    """
    out = []
    for a, b in combinations(models, 2):
        if band is not None and max(params[a], params[b]) / min(params[a], params[b]) > band:
            continue
        if any(cloze[c][metric][a] == cloze[c][metric][b] for c in CRITERIA):
            continue
        if any(bpb[c]["adapted_bpb"][a] == bpb[c]["adapted_bpb"][b] for c in CRITERIA):
            continue
        out.append((a, b))
    return out


def _acc(selector, target, pairs):
    """Selection accuracy over an explicitly supplied pair list. Lower selector value = better."""
    if not pairs:
        return None
    correct = sum((selector[a] < selector[b]) == (target[a] > target[b]) for a, b in pairs)
    return correct / len(pairs)


def _cells(bpb, cloze, metric, pairs):
    """Observed accuracy for each (selector, criterion) cell, all on the same pair list."""
    out = {}
    for corpus, sel in zip(CRITERIA, SELECTORS):
        for crit in CRITERIA:
            out[f"{sel}__vs__{crit}_cloze"] = {
                "pairwise_accuracy": _acc(bpb[corpus]["adapted_bpb"], cloze[crit][metric], pairs),
                "n_pairs": len(pairs)}
    return out


def _did(vals):
    """(news-adapted minus math-adapted on news cloze) minus (the same on math cloze)."""
    a = vals["news_adapted_bpb__vs__news_cloze"]
    c = vals["math_adapted_bpb__vs__news_cloze"]
    b = vals["news_adapted_bpb__vs__math_cloze"]
    d = vals["math_adapted_bpb__vs__math_cloze"]
    if None in (a, b, c, d):
        return None
    return (a - c) - (b - d)


def analyse(bpb, cloze, models, params, metric, band=None):
    pairs = _eligible_pairs(bpb, cloze, models, params, metric, band)
    obs = _cells(bpb, cloze, metric, pairs)
    obs_flat = {k: v["pairwise_accuracy"] for k, v in obs.items()}

    pair_set = set(pairs)
    rng = random.Random(SEED)
    draws_did: list[float] = []
    draws_arm: dict[str, list[float]] = {"news_arm": [], "math_arm": []}
    degenerate = 0
    for _ in range(REPS):
        sample = [rng.choice(models) for _ in models]
        # Pairs induced by the resampled cohort, restricted to the same eligibility rule. A model
        # drawn twice contributes no self-pair; drawing (a,b) k times weights it k times, which is
        # what an induced node bootstrap is supposed to do.
        drawn = [(a, b) for a, b in combinations(sample, 2)
                 if a != b and ((a, b) in pair_set or (b, a) in pair_set)]
        vals = {}
        for corpus, sel in zip(CRITERIA, SELECTORS):
            for crit in CRITERIA:
                vals[f"{sel}__vs__{crit}_cloze"] = _acc(
                    bpb[corpus]["adapted_bpb"], cloze[crit][metric], drawn)
        did = _did(vals)
        if did is None:
            degenerate += 1
            continue
        draws_did.append(did)
        # Each arm on its own: the matched selector minus the mismatched one, per criterion.
        draws_arm["news_arm"].append(vals["news_adapted_bpb__vs__news_cloze"]
                                     - vals["math_adapted_bpb__vs__news_cloze"])
        draws_arm["math_arm"].append(vals["math_adapted_bpb__vs__math_cloze"]
                                     - vals["news_adapted_bpb__vs__math_cloze"])

    frac = degenerate / REPS
    usable = frac <= MAX_DEGENERATE_FRAC

    def interval(draws, observed):
        d = sorted(draws)
        lo = hi = None
        if usable and d:
            lo, hi = _quantile(d, 0.025), _quantile(d, 0.975)
        have = lo is not None and hi is not None
        return {
            "observed": observed,
            "ci_lo": lo, "ci_hi": hi,
            "inference_available": have,
            # THREE distinct outcomes that a single "separates_from_zero" flag silently merged:
            # an interval containing zero, an interval entirely BELOW zero, and no interval at all.
            # A strongly negative effect would have satisfied "does not separate from zero", which
            # is the opposite of what that sentence is used to mean.
            "ci_excludes_zero": have and (lo > 0 or hi < 0),
            "supports_positive_effect": have and lo > 0,
            "supports_negative_effect": have and hi < 0,
            "ci_contains_zero": have and lo <= 0 <= hi,
            "frac_draws_positive": (sum(1 for v in d if v > 0) / len(d)) if d else None,
            "n_draws": len(d),
        }

    arms = {
        "news_arm": interval(draws_arm["news_arm"],
                             obs_flat["news_adapted_bpb__vs__news_cloze"]
                             - obs_flat["math_adapted_bpb__vs__news_cloze"]),
        "math_arm": interval(draws_arm["math_arm"],
                             obs_flat["math_adapted_bpb__vs__math_cloze"]
                             - obs_flat["news_adapted_bpb__vs__math_cloze"]),
    }
    return {
        "cells": obs,
        "difference_in_differences": interval(draws_did, _did(obs_flat)),
        "arms": arms,
        # POINT ESTIMATES ONLY, and named so. Both arms positive is the pattern steering predicts,
        # and a positive interaction can arise from a single non-zero arm, so the arms are reported
        # separately -- but this field says nothing about whether either arm is resolved. That is
        # `arms.*.supports_positive_effect`.
        "both_arms_positive_point_estimate": all(
            a["observed"] is not None and a["observed"] > 0 for a in arms.values()),
        "both_arms_resolved": all(a["supports_positive_effect"] for a in arms.values()),
        "degenerate_draw_frac": frac,
        "inference_available": usable,
    }


def build() -> dict:
    bpb, _ = load_bpb_matrix()
    params = _load_params()
    cloze = {}
    for crit in CRITERIA:
        d = ROOT / "results" / ("cloze" if crit == "news" else f"cloze_{crit}")
        pat = "cloze__*.json" if crit == "news" else f"cloze_{crit}__*.json"
        cloze[crit] = {m: load_cloze(d, m, pat) for m in ("lenient_accuracy", "strict_accuracy")}

    models = sorted(set(cloze["news"]["lenient_accuracy"]) & set(cloze["math"]["lenient_accuracy"]))
    report = {
        "schema_version": SCHEMA_VERSION,
        "question": "Does adapting on a corpus make BPB a better selector for THAT corpus's "
                    "in-domain criterion? Asked with no public benchmark anywhere in it.",
        "method": f"pairwise selection accuracy; cluster bootstrap over models, {REPS} draws, "
                  f"seed {SEED}; pairs are NOT resampled; one shared resample stream across all "
                  f"four cells so the interaction is read directly. Interval covers which models "
                  f"are in the cohort only.",
        "n_models": len(models),
        "models": models,
        "models_missing_a_criterion": sorted(
            (set(cloze["news"]["lenient_accuracy"]) | set(cloze["math"]["lenient_accuracy"]))
            - set(models)),
        "primary_band": PRIMARY_BAND,
    }
    for metric, key in (("lenient_accuracy", "lenient"), ("strict_accuracy", "strict")):
        report[key] = {"all_pairs": analyse(bpb, cloze, models, params, metric)}
        for band in SIZE_BANDS:
            blk = analyse(bpb, cloze, models, params, metric, band)
            blk["n_pairs"] = _band_pair_count(models, params, band)
            blk["is_primary"] = band == PRIMARY_BAND
            report[key][f"within_{band:g}x"] = blk
    # Deliberately NOT called "steering holds". It is a statement about the sign of four point
    # estimates on the primary estimand's scope, nothing more; no arm's interval enters it.
    report["primary_estimand"] = PRIMARY_ESTIMAND
    report["sign_pattern_consistent_with_steering"] = bool(
        report["lenient"]["all_pairs"]["both_arms_positive_point_estimate"]
        and report["strict"]["all_pairs"]["both_arms_positive_point_estimate"])
    report["any_arm_resolved"] = any(
        blk["arms"][a]["supports_positive_effect"]
        for k in ("lenient", "strict") for blk in report[k].values() for a in ("news_arm", "math_arm"))
    report["any_interaction_resolved"] = any(
        blk["difference_in_differences"]["ci_excludes_zero"]
        for k in ("lenient", "strict") for blk in report[k].values())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    report = build()

    if a.check:
        if not OUTPUT_JSON.exists():
            raise SystemExit(f"missing {OUTPUT_JSON.relative_to(ROOT)}; run without --check")
        diffs = _diff(json.loads(OUTPUT_JSON.read_text()), report)
        if diffs:
            for d in diffs[:20]:
                print("  " + d)
            raise SystemExit(f"{OUTPUT_JSON.relative_to(ROOT)} is stale ({len(diffs)} diffs)")
        print("cloze steering artifact is up to date")
        return

    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Corpus x criterion, {report['n_models']} models "
          f"(missing: {report['models_missing_a_criterion'] or 'none'})\n")
    for key in ("lenient", "strict"):
        print(f"--- {key} scoring ---")
        for scope in ("all_pairs", f"within_{PRIMARY_BAND:g}x"):
            blk = report[key][scope]
            print(f"  {scope}:")
            for name, cell in blk["cells"].items():
                acc = cell["pairwise_accuracy"]
                print(f"    {name:<44}{acc:.4f}" if acc is not None else f"    {name:<44} n/a")
            for label in ("news_arm", "math_arm"):
                arm = blk["arms"][label]
                ci = (f"[{arm['ci_lo']:+.3f}, {arm['ci_hi']:+.3f}]"
                      if arm["ci_lo"] is not None else "n/a")
                print(f"    {label:<44}{arm['observed']:+.4f}  {ci}"
                      f"  resolved={arm['supports_positive_effect']}")
            did = blk["difference_in_differences"]
            ci = f"[{did['ci_lo']:+.3f}, {did['ci_hi']:+.3f}]" if did["ci_lo"] is not None else "n/a"
            print(f"    {'DiD':<44}{did['observed']:+.4f}  {ci}"
                  f"  excludes0={did['ci_excludes_zero']}"
                  f"  positive in {did['frac_draws_positive']:.1%}")
            print(f"    both arms positive (point est.): {blk['both_arms_positive_point_estimate']}   both resolved: {blk['both_arms_resolved']}")
    print(f"\nsign pattern consistent with steering (POINT ESTIMATES): "
          f"{report['sign_pattern_consistent_with_steering']}")
    print(f"any arm resolved: {report['any_arm_resolved']}   "
          f"any interaction resolved: {report['any_interaction_resolved']}")
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E8: does the resolved comparison survive the training stochasticity nobody measured?

Every published domain-transfer cell ran with the torch RNG unseeded -- `DataConfig.seed`
fixes the split and the 40,000-document subsample, but LoRA A/B initialisation, dropout and
`DataLoader(shuffle=True)` batch order did not. The paper's bootstrap resamples MODELS, so it
carries between-model variation and no within-cell training variance at all. The study's only
resolved comparison, adapted BPB minus parameter count = +0.250 [+0.074, +0.500] under lenient
scoring, therefore rests on one stochastic draw per cell.

This scores three explicit seed replicates per model against the same news cloze criterion and
re-asks the four questions fixed in docs/RUN_LEDGER.md, "E8 -- seed sensitivity: PREREGISTRATION".

**The estimator is the published one, extended in exactly one way.** Q2 re-runs the published
table three times, once per complete seed assignment. Q3 adds a second resampling stage: a draw
picks models with replacement AND, for each drawn model, one of its seed replicates -- so the
interval covers training stochasticity as well as cohort composition. `--check` asserts that with
the seed stage disabled this reproduces the published number, which is what makes the comparison
between them meaningful rather than two different estimators.

The published unseeded cells remain the numbers of record, per the preregistration's
paper-of-record rule; these replicates estimate the variance around them and never replace them.

Standard library only.

    python tools/analyze_seed_sensitivity.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_cloze_validity import (  # noqa: E402
    PRIMARY_BAND, REPS, ROOT, SEED, _load_params, _score, load_bpb_matrix, load_cloze,
)
from analyze_cohort_extension import _load_ext, CONTROL_LABEL  # noqa: E402

CELL_DIR = Path(os.environ.get("E8_DIR",
                               Path(os.environ.get("SCRATCH", "/tmp")) / "e8_seed"))
OUTPUT_JSON = ROOT / "results" / "seed_sensitivity.json"
SCHEMA_VERSION = 1
SEEDS = (1, 2, 3)
# Published lenient value this must reproduce with the seed stage disabled, from
# results/cohort_extension.json. A mismatch means the estimator drifted, not that seeds matter.
PUBLISHED_LENIENT_DIFF = 0.25


def load_seed_cells() -> dict[str, dict[int, float]]:
    """{slug: {seed: adapted_bpb}} from the E8 cells, refusing anything partial."""
    cells: dict[str, dict[int, float]] = {}
    for path in sorted(CELL_DIR.glob("*__seed*__steps250_*.json")):
        d = json.loads(path.read_text())
        slug = path.name.split("__seed")[0]
        seed = d.get("seed")
        if seed is None or d.get("adapted_bpb") is None:
            raise SystemExit(f"FATAL: {path} has no seed or no adapted_bpb")
        if d.get("train_steps") != 250:
            raise SystemExit(f"FATAL: {path} is not the pinned 250-step protocol")
        cells.setdefault(slug, {})[int(seed)] = float(d["adapted_bpb"])
    return cells


def _published_adapted() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Published adapted BPB, parameter counts and the lenient cloze criterion, news, 17 models."""
    bpb_pub, _ = load_bpb_matrix()
    adapt = dict(bpb_pub["news"]["adapted_bpb"])
    params = dict(_load_params())
    lenient = dict(load_cloze(ROOT / "results" / "cloze", "lenient_accuracy"))
    dt, _, cloze_ext = _load_ext()
    for label in dt:
        if label == CONTROL_LABEL:
            continue
        if label in cloze_ext:
            adapt[label] = dt[label]["adapted_bpb"]
            params[label] = dt[label]["total_params"]
            lenient[label] = cloze_ext[label]["lenient_accuracy"]
    return adapt, params, lenient


def _diff_bootstrap(adapted_by_seed: dict[str, dict[int, float]] | None,
                    fixed_adapted: dict[str, float],
                    params: dict[str, float], target: dict[str, float],
                    models: list[str]) -> dict:
    """Paired adapted-BPB minus parameter-count difference, one shared stream.

    ``adapted_by_seed`` None reproduces the published single-value estimator; otherwise each
    drawn model also draws one of its seed replicates, so the interval covers training noise.
    """
    rng = random.Random(SEED)
    obs_bpb, n_pairs = _score(fixed_adapted, target, models, True, params, PRIMARY_BAND)
    obs_size, _ = _score(params, target, models, False, params, PRIMARY_BAND)
    diffs: list[float] = []
    skipped = 0
    for _ in range(REPS):
        sample = [rng.choice(models) for _ in models]
        if adapted_by_seed is None:
            sel = fixed_adapted
        else:
            sel = {m: adapted_by_seed[m][rng.choice(sorted(adapted_by_seed[m]))]
                   for m in set(sample)}
        a, _ = _score(sel, target, sample, True, params, PRIMARY_BAND)
        b, _ = _score(params, target, sample, False, params, PRIMARY_BAND)
        if a is None or b is None:
            skipped += 1
            continue
        diffs.append(a - b)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]
    return {
        "adapted_pairwise_accuracy": obs_bpb,
        "parameter_count_pairwise_accuracy": obs_size,
        "difference": (obs_bpb - obs_size) if None not in (obs_bpb, obs_size) else None,
        "ci_lo": lo, "ci_hi": hi,
        "separates_from_zero": lo > 0 or hi < 0,
        "frac_draws_positive": sum(d > 0 for d in diffs) / len(diffs),
        "n_pairs": n_pairs, "n_draws": len(diffs), "draws_skipped": skipped,
        "resamples_seeds": adapted_by_seed is not None,
    }


def build() -> dict:
    cells = load_seed_cells()
    adapted_pub, params, lenient = _published_adapted()

    complete = sorted(s for s in cells if set(cells[s]) == set(SEEDS))
    partial = sorted(s for s in cells if set(cells[s]) != set(SEEDS))
    models = sorted(set(complete) & set(lenient) & set(params) & set(adapted_pub))
    missing = sorted(set(adapted_pub) & set(lenient) & set(params) - set(complete))

    # Preregistered exclusion rule: more than 2 models short and Q3 is NOT ANSWERED.
    answerable = len(missing) <= 2

    in_band = [(a, b) for a, b in combinations(models, 2)
               if max(params[a], params[b]) / min(params[a], params[b]) <= PRIMARY_BAND]

    # Q1 -- within-cell spread against the between-model spread it has to compete with.
    per_model = {
        m: {"mean": statistics.mean(cells[m].values()),
            "sd": statistics.stdev(cells[m].values()),
            "min": min(cells[m].values()), "max": max(cells[m].values()),
            "range_frac_of_mean": (max(cells[m].values()) - min(cells[m].values()))
                                  / statistics.mean(cells[m].values()),
            "published": adapted_pub.get(m),
            "by_seed": {str(k): v for k, v in sorted(cells[m].items())}}
        for m in models}
    between_sd = statistics.stdev([adapted_pub[m] for m in models]) if len(models) > 1 else None
    median_within = statistics.median([per_model[m]["sd"] for m in models])

    # Q4 -- the only mechanism by which seed noise can move Q2 or Q3.
    reversals = []
    for a, b in in_band:
        orders = {cells[a][s] < cells[b][s] for s in SEEDS}
        if len(orders) > 1:
            reversals.append({"a": a, "b": b,
                              "a_by_seed": {str(s): cells[a][s] for s in SEEDS},
                              "b_by_seed": {str(s): cells[b][s] for s in SEEDS}})

    # Q2 -- the published table, re-run once per complete seed assignment.
    per_seed_tables = {}
    for s in SEEDS:
        sel = {m: cells[m][s] for m in models}
        acc, n = _score(sel, lenient, models, True, params, PRIMARY_BAND)
        per_seed_tables[str(s)] = {"adapted_pairwise_accuracy": acc, "n_pairs": n}
    pub_acc, pub_n = _score(adapted_pub, lenient, models, True, params, PRIMARY_BAND)

    # Q3 -- the primary estimand, with and without the seed resampling stage.
    without = _diff_bootstrap(None, adapted_pub, params, lenient, models)
    with_seeds = _diff_bootstrap({m: cells[m] for m in models}, adapted_pub,
                                 params, lenient, models)

    return {
        "schema_version": SCHEMA_VERSION,
        "question": "Does the resolved comparison survive within-cell training stochasticity?",
        "preregistration": 'docs/RUN_LEDGER.md, "E8 -- seed sensitivity: PREREGISTRATION"',
        "method": f"three explicit seed replicates per model under the pinned E3 protocol; "
                  f"news in-domain cloze criterion, lenient scoring; {PRIMARY_BAND}x band; "
                  f"cluster bootstrap over models, {REPS} draws, seed {SEED}; the seed-resampling "
                  f"variant additionally draws one replicate per drawn model. The published "
                  f"unseeded cells remain the numbers of record and are never replaced.",
        "seeds": list(SEEDS),
        "n_models_complete": len(models),
        "models": models,
        "models_missing_replicates": missing,
        "models_partial": partial,
        "q3_answerable": answerable,
        "n_in_band_pairs": len(in_band),
        "q1_within_cell_spread": {
            "per_model": per_model,
            "median_within_model_sd": median_within,
            "between_model_sd_published": between_sd,
            "within_over_between": (median_within / between_sd) if between_sd else None,
        },
        "q2_pairwise_accuracy": {
            "published_unseeded": {"adapted_pairwise_accuracy": pub_acc, "n_pairs": pub_n},
            "per_seed": per_seed_tables,
        },
        "q3_adapted_minus_parameter_count": {
            "published_estimator_no_seed_stage": without,
            "with_seed_resampling": with_seeds,
        },
        "q4_in_band_order_reversals": {"count": len(reversals), "pairs": reversals},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the committed artifact reproduces and the estimator is intact")
    args = ap.parse_args()

    report = build()

    if args.check:
        if not OUTPUT_JSON.exists():
            print(f"FAIL: {OUTPUT_JSON.relative_to(ROOT)} does not exist", file=sys.stderr)
            return 1
        if json.loads(OUTPUT_JSON.read_text()) != report:
            print(f"FAIL: {OUTPUT_JSON.relative_to(ROOT)} does not reproduce", file=sys.stderr)
            return 1
        # The estimator must be the published one when the seed stage is off, or the
        # with/without comparison is between two different things.
        got = report["q3_adapted_minus_parameter_count"]["published_estimator_no_seed_stage"]
        if got["difference"] is None or abs(got["difference"] - PUBLISHED_LENIENT_DIFF) > 1e-9:
            print(f"FAIL: seed stage disabled gives {got['difference']}, "
                  f"published is {PUBLISHED_LENIENT_DIFF}", file=sys.stderr)
            return 1
        print(f"OK: {OUTPUT_JSON.relative_to(ROOT)} reproduces; estimator matches published "
              f"({got['difference']:+.3f})")
        return 0

    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    q3 = report["q3_adapted_minus_parameter_count"]
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"models with all 3 seeds: {report['n_models_complete']}"
          f"  missing: {report['models_missing_replicates'] or 'none'}"
          f"  Q3 answerable: {report['q3_answerable']}")
    print(f"Q1 median within-model SD {report['q1_within_cell_spread']['median_within_model_sd']:.6f}"
          f" vs between-model SD {report['q1_within_cell_spread']['between_model_sd_published']:.6f}"
          f"  ratio {report['q1_within_cell_spread']['within_over_between']:.4f}")
    print(f"Q2 published {report['q2_pairwise_accuracy']['published_unseeded']} "
          f"per-seed { {k: round(v['adapted_pairwise_accuracy'], 3) for k, v in report['q2_pairwise_accuracy']['per_seed'].items()} }")
    print(f"Q3 no seed stage: {q3['published_estimator_no_seed_stage']['difference']:+.3f} "
          f"[{q3['published_estimator_no_seed_stage']['ci_lo']:+.3f}, "
          f"{q3['published_estimator_no_seed_stage']['ci_hi']:+.3f}] "
          f"separates={q3['published_estimator_no_seed_stage']['separates_from_zero']}")
    print(f"Q3 with seeds:    {q3['with_seed_resampling']['difference']:+.3f} "
          f"[{q3['with_seed_resampling']['ci_lo']:+.3f}, "
          f"{q3['with_seed_resampling']['ci_hi']:+.3f}] "
          f"separates={q3['with_seed_resampling']['separates_from_zero']}")
    print(f"Q4 in-band order reversals across seeds: "
          f"{report['q4_in_band_order_reversals']['count']} of {report['n_in_band_pairs']} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

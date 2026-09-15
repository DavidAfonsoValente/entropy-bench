#!/usr/bin/env python3
"""What a practitioner with two or three candidates should do with a BPB gap.

Everything else in this study is a cohort-level statistic over 40 size-matched pairs. A team
choosing between two models has one pair, and needs a different object: given that candidate A
beats candidate B by some relative margin in adapted BPB, is that margin worth acting on?

Two independent things bound the answer and both are read from committed artifacts rather than
asserted here:

* a **noise floor** -- the gap two identical runs could produce on their own, from seed variation
  (``results/seed_sensitivity.json``) and from token-aligned evaluation blocks scoring slightly
  different amounts of text (``results/tokenizer_bias_ext.json``);
* an **empirical calibration** -- across the cohort's own pairs, how often the model with the lower
  adapted BPB is in fact the better model by the in-domain criterion, as a function of the gap.

The calibration is reported twice: over all pairs at or above each threshold, and over the subset
whose criterion difference is itself resolved (paired item bootstrap,
``results/criterion_uncertainty.json``). The second column is the honest one, because a pair the
yardstick cannot order should not be counted as a selector's mistake. This is the same move
Appendix C already makes for benchmark standard errors.

Writes ``results/decision_rule.json`` and ``decision_rule_table.tex``. ``--check`` fails if either
is stale.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_alignment_matrix import load_bpb_matrix  # noqa: E402
from pairwise_significance import _load_params  # noqa: E402

EXT_DT = ROOT / "results" / "cohort_ext" / "dt"
CRITERION_JSON = ROOT / "results" / "criterion_uncertainty.json"
SEED_JSON = ROOT / "results" / "seed_sensitivity.json"
TOKENIZER_JSON = ROOT / "results" / "tokenizer_bias_ext.json"
OUTPUT_JSON = ROOT / "results" / "decision_rule.json"
OUTPUT_TEX = ROOT / "decision_rule_table.tex"
SCHEMA_VERSION = 1

CORPUS = "news"
BAND = 2.0
CONTROL_PREFIX = "CONTROL-"
# Relative-gap thresholds, in percent of the better model's adapted BPB. The paper already speaks
# in these units ("3.4% apart"), so a reader can apply the rule without a conversion.
# 0.5% is identical to the all-pairs row (no in-band pair is that close) and 3% duplicates 2%;
# four thresholds carry the rule.
THRESHOLDS = (0.0, 1.0, 2.0, 5.0)


def _adapted_bpb_17() -> tuple[dict[str, float], dict[str, float]]:
    """Adapted news BPB and parameter counts for all 17 models.

    ``_load_params`` covers only the published eleven; the extension cells carry their own
    ``total_params``, which is where analyze_cohort_extension.py takes them from too.
    """
    bpb, _ = load_bpb_matrix()
    out = dict(bpb[CORPUS]["adapted_bpb"])
    params = dict(_load_params())
    for path in sorted(EXT_DT.glob(f"{CORPUS}__*.json")):
        row = json.loads(path.read_text())
        label = row["label"]
        if label.startswith(CONTROL_PREFIX):
            continue
        out[label] = row["adapted_bpb"]
        params[label] = row["total_params"]
    return out, params


def _noise_floor(bpb: dict[str, float]) -> dict:
    """The relative gap below which adapted BPB cannot resolve two models at all."""
    seed = json.loads(SEED_JSON.read_text())["q1_within_cell_spread"]
    per_model = seed["per_model"]
    worst_label = max(per_model, key=lambda m: per_model[m]["sd"])
    worst_sd = per_model[worst_label]["sd"]
    median_sd = seed["median_within_model_sd"]
    # Two independently trained cells differ by sd*sqrt(2) in SD terms; 2 SD of that difference is
    # the gap seed noise alone could manufacture between two models.
    typical = sorted(bpb.values())[len(bpb) // 2]
    seed_gap = 2 * (2 ** 0.5) * worst_sd
    tok = json.loads(TOKENIZER_JSON.read_text())["bound"]
    worst_pair = tok["worst_case"]
    return {
        "seed": {
            "worst_model": worst_label, "worst_within_model_sd": worst_sd,
            "median_within_model_sd": median_sd,
            "gap_two_such_models_could_produce": seed_gap,
            "as_frac_of_median_bpb": seed_gap / typical,
            "basis": "2 x sqrt(2) x the largest within-model seed SD over 3 replicates",
        },
        "block_alignment": {
            "worst_case_bias_bound_frac": worst_pair["bias_bound_frac"],
            "worst_pair": f"{worst_pair['better']} over {worst_pair['worse']}",
            "bias_over_gap": worst_pair["bias_over_gap"],
        },
        # The two are independent sources, so they add rather than one dominating.
        "combined_frac": seed_gap / typical + worst_pair["bias_bound_frac"],
    }


def build() -> dict:
    bpb, params = _adapted_bpb_17()
    crit = json.loads(CRITERION_JSON.read_text())
    models = sorted(m for m in bpb if m in crit["cohort_17"])

    rows = {}
    for conv in ("lenient", "strict"):
        acc = {m: crit[conv]["per_model"][m]["accuracy"] for m in models}
        resolved = crit[conv]["pairwise"]
        pairs = []
        for a, b in combinations(models, 2):
            if max(params[a], params[b]) / min(params[a], params[b]) > BAND:
                continue
            if acc[a] == acc[b] or bpb[a] == bpb[b]:
                continue
            better_bpb = a if bpb[a] < bpb[b] else b
            better_crit = a if acc[a] > acc[b] else b
            key = f"{a}|{b}" if f"{a}|{b}" in resolved else f"{b}|{a}"
            gap = abs(bpb[a] - bpb[b]) / min(bpb[a], bpb[b]) * 100
            pairs.append({
                "pair": f"{a}|{b}", "rel_bpb_gap_pct": gap,
                "agrees": better_bpb == better_crit,
                "criterion_resolved": resolved[key]["ci_excludes_zero"],
            })

        table = []
        for t in THRESHOLDS:
            sel = [p for p in pairs if p["rel_bpb_gap_pct"] >= t]
            res = [p for p in sel if p["criterion_resolved"]]
            table.append({
                "threshold_pct": t,
                "n_pairs": len(sel),
                "accuracy": (sum(p["agrees"] for p in sel) / len(sel)) if sel else None,
                "n_pairs_criterion_resolved": len(res),
                "accuracy_criterion_resolved": (sum(p["agrees"] for p in res) / len(res))
                if res else None,
            })
        rows[conv] = {"n_in_band_pairs": len(pairs), "by_threshold": table, "pairs": pairs}

    return {
        "schema_version": SCHEMA_VERSION,
        "question": "Given two candidates whose adapted BPB differs by a relative margin g, how "
                    "often is the lower-BPB model actually the better one?",
        "method": {
            "corpus": CORPUS, "band": BAND,
            "cohort": models,
            "gap_units": "percent of the smaller adapted BPB",
            "criterion_resolved_column": "restricted to pairs whose criterion difference has a "
                                         "paired item-bootstrap CI excluding zero; a pair the "
                                         "yardstick cannot order is not a selector error",
        },
        "noise_floor": _noise_floor(bpb),
        **rows,
    }


def render_tex(data: dict) -> str:
    out = []
    for conv in ("lenient", "strict"):
        for i, row in enumerate(data[conv]["by_threshold"]):
            label = conv if i == 0 else ""
            acc = "---" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
            racc = ("---" if row["accuracy_criterion_resolved"] is None
                    else f"{row['accuracy_criterion_resolved']:.3f}")
            thr = "all" if row["threshold_pct"] == 0 else f"$\\geq{row['threshold_pct']:g}\\%$"
            out.append(f"{label} & {thr} & {row['n_pairs']} & {acc} & "
                       f"{row['n_pairs_criterion_resolved']} & {racc} \\\\")
        if conv == "lenient":
            out.append("\\midrule")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = ap.parse_args()
    built = build()
    tex = render_tex(built)

    if args.check:
        problems = []
        if not OUTPUT_JSON.exists():
            problems.append("results/decision_rule.json is missing")
        elif json.dumps(json.loads(OUTPUT_JSON.read_text()), sort_keys=True) != json.dumps(
                built, sort_keys=True):
            problems.append("results/decision_rule.json is stale")
        if not OUTPUT_TEX.exists():
            problems.append("decision_rule_table.tex is missing")
        elif OUTPUT_TEX.read_text() != tex:
            problems.append("decision_rule_table.tex is stale")
        for p in problems:
            print("  " + p)
        print("decision rule: PROBLEMS", len(problems))
        sys.exit(1 if problems else 0)

    OUTPUT_JSON.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n")
    OUTPUT_TEX.write_text(tex)
    print("wrote", OUTPUT_JSON.relative_to(ROOT), "and", OUTPUT_TEX.name)


if __name__ == "__main__":
    main()

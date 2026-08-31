#!/usr/bin/env python3
"""Three confound checks on experiment E2, all computable from committed artifacts.

E2 (``tools/analyze_adapted_benchmarks.py``) reports that math-domain adaptation costs GSM8K
accuracy roughly twice what news adaptation does, with model ordering preserved. Before that can
be used to rewrite ``paper_sota.tex`` Section 5.5, three alternative explanations have to be ruled
in or out. None of them needs a GPU.

1. ANSWER-EXTRACTION DRIFT. GSM8K is scored by generating and then extracting an answer, while
   HellaSwag is a loglikelihood task. A benchmark pair where only the generative one moves is the
   signature of adaptation shifting output conventions until the extractor misses the answer,
   rather than of any change in ability. lm-eval records both a ``strict-match`` and a lenient
   ``flexible-extract`` score, so the two can be compared directly: a widening gap is format
   drift, and drift that survives the lenient extractor is not an extraction artifact.

2. THE PAIRED CONTRAST. Comparing two cell means discards the pairing. The honest statistic is
   per-model (math delta minus news delta), which cancels model-level noise, plus the cohort size
   that would be needed to resolve the effect actually observed.

3. BPB-MOVEMENT LEVERAGE. "Adaptation damage is proportional to how far the model moved in BPB"
   is an attractive mechanism, and ``paper_sota.tex`` Sec. 7.1 already notes that the models that
   move most begin furthest from the corpus. But one model (LFM2.5-1.2B) reduces BPB by 36-42%
   where every other model reduces it by 3.6-7.9%, so it alone determines any correlation fitted
   over this cohort. Reported with and without it.

    python tools/audit_e2_confounds.py [--runs results/adapted_bench] [--check]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_static_benchmarks import DISPLAY, MODEL_IDS, ROOT  # noqa: E402
from _artifact_check import _diff  # noqa: E402

OUTPUT_JSON = ROOT / "results" / "e2_confound_audit.json"
ANALYSIS = ROOT / "results" / "adapted_benchmark_analysis.json"
CORPORA = ["news", "math"]
STRICT, FLEX = "exact_match,strict-match", "exact_match,flexible-extract"

# Power target for the cohort-size question: two-sided alpha = 0.05 at 80% power.
Z_SUM_SQ = (1.959963985 + 0.841621234) ** 2


def _slug(model_id: str) -> str:
    """Mirror of ``lm_adapt_bench.utils.slugify`` (imported there, but that module needs torch)."""
    return re.sub(r"[^a-zA-Z0-9\-_]", "_", model_id.replace("/", "_")).strip("_")


def _gsm8k(pattern: str) -> dict | None:
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return json.loads(Path(matches[-1]).read_text())["results"]["gsm8k"]


def extraction_drift(run_root: Path, labels: list[str]) -> dict:
    """flexible-extract minus strict-match, per model per tier. A widening gap is format drift."""
    out = {}
    for label in labels:
        tiers = {"base": _gsm8k(str(run_root / "base" / "gsm8k" / label / "*" / "results_*.json"))}
        for corpus in CORPORA:
            tiers[corpus] = _gsm8k(str(run_root / corpus / "gsm8k" /
                                       f"{label}__{corpus}-adapted" / "*" / "results_*.json"))
        if any(v is None for v in tiers.values()):
            continue
        out[label] = {
            tier: {
                "strict": v[STRICT],
                "flexible": v[FLEX],
                "flex_minus_strict_pp": (v[FLEX] - v[STRICT]) * 100.0,
            } for tier, v in tiers.items()
        }
        for corpus in CORPORA:
            # Degradation the lenient extractor cannot rescue is not an extraction artifact.
            out[label][corpus]["strict_delta_pp"] = (
                tiers[corpus][STRICT] - tiers["base"][STRICT]) * 100.0
            out[label][corpus]["flexible_delta_pp"] = (
                tiers[corpus][FLEX] - tiers["base"][FLEX]) * 100.0
            out[label][corpus]["drift_widening_pp"] = (
                out[label][corpus]["flex_minus_strict_pp"] - out[label]["base"]["flex_minus_strict_pp"])
    return out


def paired_contrast(cells: dict, labels: list[str]) -> dict:
    """Per-model (math delta - news delta), and the cohort size that would resolve it."""
    per_model = {label: cells["math__gsm8k"]["per_model"][label]["delta_pp"]
                       - cells["news__gsm8k"]["per_model"][label]["delta_pp"]
                 for label in labels}
    values = list(per_model.values())
    mean, sd = statistics.mean(values), statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    return {
        "per_model_pp": per_model,
        "n": len(values),
        "mean_pp": mean,
        "sd_pp": sd,
        "se_pp": se,
        "t": mean / se if se else float("nan"),
        "df": len(values) - 1,
        "n_negative": sum(1 for v in values if v < 0),
        "n_models_for_80pct_power": Z_SUM_SQ * (sd / abs(mean)) ** 2 if mean else float("inf"),
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


def _spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for pos, i in enumerate(order):
            out[i] = pos + 1
        return out
    return _pearson(rank(xs), rank(ys))


def bpb_leverage(cells: dict, labels: list[str], drop: str) -> dict:
    """Correlate BPB reduction with delta-accuracy over all cells, with and without ``drop``."""
    points = []
    for label in labels:
        for corpus in CORPORA:
            dt = ROOT / "results" / "domain_transfer" / f"{corpus}__{_slug(MODEL_IDS[label])}.json"
            points.append({
                "model": label,
                "corpus": corpus,
                "bpb_reduction_pct": json.loads(dt.read_text())["reduction_pct"],
                "delta_pp": cells[f"{corpus}__gsm8k"]["per_model"][label]["delta_pp"],
            })
    kept = [p for p in points if p["model"] != drop]

    def fit(ps):
        xs = [p["bpb_reduction_pct"] for p in ps]
        ys = [p["delta_pp"] for p in ps]
        return {"n_cells": len(ps), "pearson": _pearson(xs, ys), "spearman": _spearman(xs, ys),
                "bpb_reduction_range_pct": [min(xs), max(xs)]}
    return {"points": points, "all": fit(points), "excluding": {"model": drop, **fit(kept)}}


def build(run_root: Path) -> dict:
    cells = json.loads(ANALYSIS.read_text())["cells"]
    labels = sorted(set(cells["math__gsm8k"]["per_model"]) & set(cells["news__gsm8k"]["per_model"]))
    leverage_model = max(
        labels,
        key=lambda m: json.loads((ROOT / "results" / "domain_transfer" /
                                  f"math__{_slug(MODEL_IDS[m])}.json").read_text())["reduction_pct"])
    return {
        "schema_version": 1,
        "cohort": labels,
        "sources": {
            "deltas": "results/adapted_benchmark_analysis.json",
            "extraction": str(run_root.relative_to(ROOT)) if run_root.is_relative_to(ROOT)
                          else str(run_root),
            "bpb": "results/domain_transfer/<corpus>__<slug>.json",
        },
        "extraction_drift": extraction_drift(run_root, labels),
        "paired_contrast": paired_contrast(cells, labels),
        "bpb_leverage": bpb_leverage(cells, labels, leverage_model),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "results" / "adapted_bench"))
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed audit differs from a fresh computation")
    args = ap.parse_args()

    report = build(Path(args.runs))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT_JSON.exists():
            raise SystemExit(f"{OUTPUT_JSON.relative_to(ROOT)} is missing")
        drift = _diff(json.loads(OUTPUT_JSON.read_text()), report)
        if drift:
            raise SystemExit(f"{OUTPUT_JSON.relative_to(ROOT)} is stale: " + "; ".join(drift[:5]))
        print("E2 confound audit is up to date")
        return
    OUTPUT_JSON.write_text(payload)

    print("=== 1. answer-extraction drift on GSM8K (pp) ===")
    print(f"{'model':<20}{'tier':<7}{'strict':>9}{'flexible':>10}{'flex-strict':>13}{'vs base':>10}")
    for label, tiers in report["extraction_drift"].items():
        for tier in ("base", *CORPORA):
            row = tiers[tier]
            widen = "" if tier == "base" else f"{row['drift_widening_pp']:>+10.2f}"
            print(f"{DISPLAY.get(label, label) if tier == 'base' else '':<20}{tier:<7}"
                  f"{row['strict']*100:>8.2f}%{row['flexible']*100:>9.2f}%"
                  f"{row['flex_minus_strict_pp']:>+13.2f}{widen}")

    pc = report["paired_contrast"]
    print("\n=== 2. paired per-model contrast, math delta - news delta (GSM8K) ===")
    for label, value in sorted(pc["per_model_pp"].items(), key=lambda kv: kv[1]):
        print(f"{DISPLAY.get(label, label):<20}{value:>+9.2f} pp")
    print(f"mean {pc['mean_pp']:+.2f} pp, sd {pc['sd_pp']:.2f}, se {pc['se_pp']:.2f}, "
          f"t({pc['df']}) = {pc['t']:.2f}, {pc['n_negative']}/{pc['n']} negative")
    print(f"cohort needed to resolve this effect at 80% power: "
          f"{pc['n_models_for_80pct_power']:.1f} models")

    lev = report["bpb_leverage"]
    print("\n=== 3. BPB reduction vs delta-accuracy ===")
    print(f"all {lev['all']['n_cells']} cells:      Pearson {lev['all']['pearson']:+.3f}   "
          f"Spearman {lev['all']['spearman']:+.3f}   "
          f"reduction {lev['all']['bpb_reduction_range_pct'][0]:.1f}-"
          f"{lev['all']['bpb_reduction_range_pct'][1]:.1f}%")
    ex = lev["excluding"]
    print(f"without {ex['model']} ({ex['n_cells']} cells): Pearson {ex['pearson']:+.3f}   "
          f"Spearman {ex['spearman']:+.3f}   "
          f"reduction {ex['bpb_reduction_range_pct'][0]:.1f}-{ex['bpb_reduction_range_pct'][1]:.1f}%")
    print(f"\nwrote {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

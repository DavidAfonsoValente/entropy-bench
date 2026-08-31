#!/usr/bin/env python3
"""When the metric says two models are tied, what do the benchmarks say?

The paper's motivation rests on a claim it never measures directly: that a benchmark score
conflates "this model is better" with "this model was trained on more of this". That claim is
already visible in one pair (Section 5, Llama-3.2-1B vs Qwen-2.5-1.5B at adjacent BPB ranks), and
the obvious next move is to turn the anecdote into a statistic -- correlate |BPB gap| against
|benchmark gap| over all 55 model pairs and report that BPB gaps predict HellaSwag gaps but not
GSM8K gaps.

**That statistic does not survive leave-one-out and this script exists to say so.** Over all 11
models the GSM8K correlation is +0.266; dropping Llama-3.2-1B alone takes it to +0.667, and
dropping Qwen-2.5-0.5B takes it to +0.118. A quantity that moves that far when one model of eleven
leaves is not an estimate, and reporting it as one would repeat the leverage-point error already
recorded for the BPB-movement correlation in tools/audit_e2_confounds.py. The HellaSwag
relationship, by contrast, holds between 0.859 and 0.940 under every single-model deletion.

What survives is weaker in form and sufficient for the argument:

* **Existence proofs.** Specific pairs that sit within a few percent on adapted BPB and tens of
  points apart on GSM8K are facts about those models, not estimates, so leverage does not apply.
  Qwen-3.5-4B and Llama-3.2-1B are 2.9% apart on adapted news BPB and 73.9 points apart on GSM8K.
* **A cross-family versus within-family split**, reported as descriptive: among pairs within 5% on
  adapted BPB, cross-family pairs sit far apart on GSM8K and same-family pairs do not.

Both are what the motivation needs. Neither is a claim about the population of language models,
and this script's ``estimable`` fields say so in the artifact rather than in a footnote.

    python tools/analyze_selection_divergence.py [--check]

``--check`` regenerates into memory and fails if the committed artifact differs, which is what
``make check`` and tools/verify_paper_numbers.py use. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _artifact_check import _diff  # noqa: E402
from analyze_alignment_matrix import load_bpb_matrix  # noqa: E402
from analyze_static_benchmarks import (  # noqa: E402
    BENCHMARKS, DISPLAY, DOMAIN_DIR, MODEL_IDS, ROOT, load_raw_static, pearson, ranks,
)

OUTPUT_JSON = ROOT / "results" / "selection_divergence.json"
SCHEMA_VERSION = 1

# Closeness bands on the adapted news BPB gap. Descriptive: there is no principled threshold at
# which two models become "tied", so several are reported and none is privileged.
CLOSENESS_BANDS = (0.02, 0.03, 0.05, 0.10)

# The band the prose quotes. Named once here so the paper, the slides and the artifact cannot
# drift apart.
HEADLINE_BAND = 0.05

# A correlation is reported as estimable only if no single-model deletion moves it by more than
# this. The GSM8K value moves by 0.40 (0.266 -> 0.667 dropping Llama-3.2-1B), so it fails; the
# HellaSwag value moves by at most 0.05 and passes. The threshold is a judgement, stated here
# rather than buried, and the full leave-one-out range is in the artifact either way.
STABILITY_TOLERANCE = 0.15


def family(label: str) -> str:
    """Publisher family for a display label. Used only for the cross/within-family split."""
    for prefix, name in (("Qwen", "Qwen"), ("gemma", "Gemma"), ("Llama", "Llama"),
                         ("LFM", "LiquidAI"), ("Ministral", "Mistral")):
        if label.startswith(prefix):
            return name
    raise ValueError(f"no family mapping for {label!r}")


def load_params() -> dict[str, int]:
    """Parameter counts, read from the committed news domain-transfer cells."""
    by_id = {model_id: label for label, model_id in MODEL_IDS.items()}
    out = {}
    for path in sorted(DOMAIN_DIR.glob("news__*.json")):
        row = json.loads(path.read_text())
        out[by_id[row["model_id"]]] = row["total_params"]
    if set(out) != set(MODEL_IDS):
        raise SystemExit(f"parameter coverage mismatch: {sorted(set(MODEL_IDS) - set(out))}")
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    rx = ranks({i: v for i, v in enumerate(xs)}, lower_is_better=True)
    ry = ranks({i: v for i, v in enumerate(ys)}, lower_is_better=True)
    keys = sorted(rx)
    return pearson([rx[k] for k in keys], [ry[k] for k in keys])


def pair_rows(models: list[str], bpb: dict, acc: dict, params: dict[str, int]) -> list[dict]:
    """One row per unordered model pair: how far apart on BPB, how far apart on each benchmark."""
    adapted = bpb["news"]["adapted_bpb"]
    zero_shot = bpb["news"]["zero_shot_bpb"]
    rows = []
    for a, b in combinations(sorted(models), 2):
        row = {
            "a": a,
            "b": b,
            "adapted_bpb_gap_frac": abs(adapted[a] - adapted[b]) / min(adapted[a], adapted[b]),
            "zero_shot_bpb_gap_frac": abs(zero_shot[a] - zero_shot[b]) / min(zero_shot[a],
                                                                             zero_shot[b]),
            "size_ratio": max(params[a], params[b]) / min(params[a], params[b]),
            "cross_family": family(a) != family(b),
        }
        for bench in BENCHMARKS:
            row[f"{bench}_gap_pp"] = abs(acc[bench][a] - acc[bench][b]) * 100
        rows.append(row)
    return rows


def correlations(rows: list[dict]) -> dict[str, dict[str, float]]:
    """|BPB gap| against |benchmark gap|, both coefficients, for one cohort."""
    xs = [r["adapted_bpb_gap_frac"] for r in rows]
    out = {}
    for bench in BENCHMARKS:
        ys = [r[f"{bench}_gap_pp"] for r in rows]
        out[bench] = {"pearson": pearson(xs, ys), "spearman": spearman(xs, ys), "n_pairs": len(xs)}
    return out


def leave_one_out(bpb: dict, acc: dict, params: dict[str, int]) -> dict:
    """Refutation gate: how far does each correlation move when one model is deleted?

    A magnitude correlation over pairs is not robust the way a selection accuracy is: every model
    appears in ten of the fifty-five pairs, so one unusual model moves the fit a long way. This
    reports the full range rather than a single number, and marks a benchmark ``estimable`` only
    if no deletion moves its coefficient by more than STABILITY_TOLERANCE.
    """
    labels = sorted(MODEL_IDS)
    full = correlations(pair_rows(labels, bpb, acc, params))
    per_benchmark = {}
    for bench in BENCHMARKS:
        baseline = full[bench]["pearson"]
        deletions = {}
        for dropped in labels:
            kept = [m for m in labels if m != dropped]
            deletions[dropped] = correlations(pair_rows(kept, bpb, acc, params))[bench]["pearson"]
        values = list(deletions.values())
        swing = max(abs(v - baseline) for v in values)
        worst = max(deletions, key=lambda m: abs(deletions[m] - baseline))
        per_benchmark[bench] = {
            "full_cohort_pearson": baseline,
            "min_after_deletion": min(values),
            "max_after_deletion": max(values),
            "max_abs_swing": swing,
            "most_influential_model": worst,
            "estimable": swing <= STABILITY_TOLERANCE,
            "per_deletion_pearson": deletions,
        }
    return per_benchmark


def noise_floor_census(rows: list[dict], floors=(0.69, 1.0, 2.0)) -> dict:
    """How many pairwise benchmark gaps in this cohort are smaller than measured reproduction noise.

    Re-scoring identical released weights on a different host moved GSM8K by up to 0.69 pp (E2
    phase 2). Reported here to bound our own claims, and it is a *weak* number by design: this
    cohort spans 0.5B-35B, so most pairs are separated far beyond any noise floor. The number to
    take from it is that undecidable pairs exist and we counted them, not that benchmarks are
    generally undecidable.
    """
    out = {}
    for bench in BENCHMARKS:
        gaps = [r[f"{bench}_gap_pp"] for r in rows]
        out[bench] = {f"n_gaps_within_{floor:g}pp": sum(1 for g in gaps if g <= floor)
                      for floor in floors}
        out[bench]["n_pairs"] = len(gaps)
    return out


def size_disagreement(bpb, acc, params, models) -> dict:
    """When parameter count and BPB disagree about a pair, which one does the benchmark follow?

    The size-band tables answer this by binning; this answers it directly, on the only pairs where
    the two selectors give different advice -- which is the only place a practitioner needs either.
    Pair counts are single digits, so this is descriptive and labelled as such.
    """
    out = {}
    for corpus, tier in (("math", "adapted_bpb"), ("news", "zero_shot_bpb")):
        source = bpb[corpus][tier]
        for bench in BENCHMARKS:
            bpb_right = size_right = 0
            for a, b in combinations(sorted(models), 2):
                if acc[bench][a] == acc[bench][b] or params[a] == params[b]:
                    continue
                if source[a] == source[b]:
                    continue
                better_acc = a if acc[bench][a] > acc[bench][b] else b
                better_bpb = a if source[a] < source[b] else b
                bigger = a if params[a] > params[b] else b
                if better_bpb == bigger:
                    continue                      # selectors agree; the pair is uninformative
                if better_acc == better_bpb:
                    bpb_right += 1
                else:
                    size_right += 1
            total = bpb_right + size_right
            out[f"{corpus}__{tier}__{bench}"] = {
                "n_disagreeing_pairs": total,
                "bpb_correct": bpb_right,
                "size_correct": size_right,
                "bpb_share": bpb_right / total if total else None,
            }
    return out


def build_analysis() -> dict:
    bpb, _ = load_bpb_matrix()
    acc = {bench: {m: load_raw_static(m, bench)[0] for m in MODEL_IDS} for bench in BENCHMARKS}
    params = load_params()
    rows = pair_rows(sorted(MODEL_IDS), bpb, acc, params)

    bands = {}
    for threshold in CLOSENESS_BANDS:
        close = [r for r in rows if r["adapted_bpb_gap_frac"] <= threshold]
        entry = {"n_pairs": len(close)}
        for bench in BENCHMARKS:
            gaps = [r[f"{bench}_gap_pp"] for r in close]
            entry[bench] = {
                "median_gap_pp": statistics.median(gaps) if gaps else None,
                "max_gap_pp": max(gaps) if gaps else None,
                "n_gap_at_least_20pp": sum(1 for g in gaps if g >= 20.0),
            }
        bands[f"within_{int(threshold * 100)}pct"] = entry

    close = [r for r in rows if r["adapted_bpb_gap_frac"] <= HEADLINE_BAND]
    split = {}
    for name, selected in (("cross_family", [r for r in close if r["cross_family"]]),
                           ("within_family", [r for r in close if not r["cross_family"]])):
        gaps = [r["gsm8k_gap_pp"] for r in selected]
        split[name] = {
            "n_pairs": len(selected),
            "median_gsm8k_gap_pp": statistics.median(gaps) if gaps else None,
            "max_gsm8k_gap_pp": max(gaps) if gaps else None,
        }

    # Existence proofs: the widest benchmark divergences among BPB-close pairs. These are facts
    # about named models, so the leverage argument that sinks the correlation does not apply.
    witnesses = sorted(close, key=lambda r: -r["gsm8k_gap_pp"])[:3]

    return {
        "schema_version": SCHEMA_VERSION,
        "bpb_source": "results/domain_transfer/news__*.json (adapted_bpb, zero_shot_bpb)",
        "accuracy_source": "results/{gsm8k,hellaswag,mmlu_pro_1k}/**/results_*.json",
        "note": (
            "Magnitude correlations between |BPB gap| and |benchmark gap| are reported WITH their "
            "leave-one-out range and an explicit estimable flag. The GSM8K coefficient is not "
            "estimable at this cohort size and must not be quoted as one; the argument rests on "
            "the named existence proofs and the descriptive cross/within-family split instead."
        ),
        "headline_band": HEADLINE_BAND,
        "stability_tolerance": STABILITY_TOLERANCE,
        "n_models": len(MODEL_IDS),
        "n_pairs": len(rows),
        "correlations_full_cohort": correlations(rows),
        "stability": leave_one_out(bpb, acc, params),
        "closeness_bands": bands,
        "family_split_at_headline_band": split,
        "witnesses": [
            {"a": DISPLAY[w["a"]], "b": DISPLAY[w["b"]],
             "adapted_bpb_gap_frac": w["adapted_bpb_gap_frac"],
             "gsm8k_gap_pp": w["gsm8k_gap_pp"], "hellaswag_gap_pp": w["hellaswag_gap_pp"],
             "cross_family": w["cross_family"], "size_ratio": w["size_ratio"]}
            for w in witnesses
        ],
        "noise_floor_census": noise_floor_census(rows),
        "size_disagreement": size_disagreement(bpb, acc, params, sorted(MODEL_IDS)),
        "pairs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed artifact differs from a fresh computation")
    args = parser.parse_args()

    report = build_analysis()
    payload = json.dumps(report, indent=2) + "\n"

    if args.check:
        if not OUTPUT_JSON.exists():
            raise SystemExit(f"missing {OUTPUT_JSON.relative_to(ROOT)}; run without --check")
        differences = _diff(json.loads(OUTPUT_JSON.read_text()), report)
        if differences:
            for d in differences[:20]:
                print(f"  {d}")
            raise SystemExit(f"{OUTPUT_JSON.relative_to(ROOT)} is stale ({len(differences)} diffs)")
        print("selection divergence artifact is up to date")
        return

    OUTPUT_JSON.write_text(payload)

    print("=== Does the size of a BPB gap predict the size of a benchmark gap? ===")
    print(f"{'benchmark':<12}{'Pearson':>9}{'Spearman':>10}   leave-one-out range        verdict")
    for bench, stats in report["correlations_full_cohort"].items():
        s = report["stability"][bench]
        verdict = "estimable" if s["estimable"] else "NOT estimable at n=11"
        print(f"{BENCHMARKS[bench]['label']:<12}{stats['pearson']:>+9.3f}{stats['spearman']:>+10.3f}"
              f"   [{s['min_after_deletion']:+.3f}, {s['max_after_deletion']:+.3f}]"
              f"  {verdict}")
    gsm = report["stability"]["gsm8k"]
    print(f"\nGSM8K swings {gsm['max_abs_swing']:.3f} when {gsm['most_influential_model']} alone "
          f"is dropped -- which is why it is reported as a range, not a coefficient.")

    print(f"\n=== Pairs within {int(HEADLINE_BAND * 100)}% on adapted news BPB ===")
    split = report["family_split_at_headline_band"]
    for name in ("cross_family", "within_family"):
        s = split[name]
        print(f"  {name.replace('_', '-'):<14} n={s['n_pairs']:<3} median GSM8K gap "
              f"{s['median_gsm8k_gap_pp']:5.1f} pp   max {s['max_gsm8k_gap_pp']:5.1f} pp")

    print("\n=== Existence proofs: tied on our ruler, far apart on theirs ===")
    for w in report["witnesses"]:
        print(f"  {w['a']:<16} / {w['b']:<16} "
              f"BPB {w['adapted_bpb_gap_frac'] * 100:4.1f}% apart | "
              f"GSM8K {w['gsm8k_gap_pp']:5.1f} pp | HellaSwag {w['hellaswag_gap_pp']:4.1f} pp")

    print("\n=== When size and BPB disagree, which does the benchmark follow? (descriptive) ===")
    for key, d in report["size_disagreement"].items():
        if d["n_disagreeing_pairs"]:
            print(f"  {key:34s} n={d['n_disagreeing_pairs']:2d}  BPB right {d['bpb_correct']:2d} "
                  f"({d['bpb_share']:.0%})")

    print(f"\nwrote {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

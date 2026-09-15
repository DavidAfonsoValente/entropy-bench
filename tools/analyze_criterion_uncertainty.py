#!/usr/bin/env python3
"""How well is the in-domain criterion itself known?

Every rank claim in the paper is measured *against* the cloze criterion, and until now the
criterion's own scores were reported as bare point estimates -- so a reader could not tell whether
an ordering separated by 0.003 (Llama-3.2-1B over Qwen3.5-4B) was an ordering at all.

The resampling unit here is the **item**, not the model, and that is deliberate. The paper's
selection-accuracy intervals resample *models* and answer "would this selector still win on another
draw of models". These intervals resample the 2,500 cloze *items* on a fixed cohort and answer a
different question: "how well do we know the yardstick". Both units appear in the paper and each is
named where it is used.

Because every model answered an identical, identically-ordered item list, the bootstrap is PAIRED:
one resample of items is applied to all models at once, so the criterion's item-difficulty noise
cancels out of a between-model difference instead of being counted twice.

Outputs ``results/criterion_uncertainty.json``. ``--check`` fails if that file is stale or if the
point estimates no longer reproduce the published ones.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_tier_mechanism import _ranks  # noqa: E402  (shared tie convention)

PUBLISHED_CLOZE = ROOT / "results" / "cloze"
EXT_CLOZE = ROOT / "results" / "cohort_ext" / "cloze"
OUTPUT_JSON = ROOT / "results" / "criterion_uncertainty.json"
SCHEMA_VERSION = 1

CORPUS = "news"
N_BOOT = 20_000          # matches the model-level bootstrap elsewhere in the paper
CHUNK = 2_000            # 17x2500 float64 per draw; the login node dies silently past ~1 GB
SEED = 20260911
CONTROL_PREFIX = "CONTROL-"


def _load_items() -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-item correctness for every model, under both scoring conventions.

    Refuses to proceed unless all models share one identical item ordering -- the paired bootstrap
    below is only valid if column j is the same cloze item for every row.
    """
    paths = sorted(PUBLISHED_CLOZE.glob("cloze__*.json"))
    paths += sorted(EXT_CLOZE.glob("cloze__*.json"))
    if not paths:
        raise SystemExit("no cloze sample files found")

    ids_ref: list[str] | None = None
    lenient: dict[str, np.ndarray] = {}
    strict: dict[str, np.ndarray] = {}
    for path in paths:
        row = json.loads(path.read_text())
        label = row["label"]
        if label.startswith(CONTROL_PREFIX):
            continue
        samples = row["samples"]
        ids = [s["id"] for s in samples]
        if ids_ref is None:
            ids_ref = ids
        elif ids != ids_ref:
            raise SystemExit(f"{path.name}: item ordering differs; the paired bootstrap is invalid")
        lenient[label] = np.array([bool(s["lenient"]) for s in samples], dtype=np.float64)
        strict[label] = np.array([bool(s["strict"]) for s in samples], dtype=np.float64)
        for conv, arr, claimed in (("lenient", lenient[label], row["lenient_accuracy"]),
                                   ("strict", strict[label], row["strict_accuracy"])):
            if abs(arr.mean() - claimed) > 5e-9:
                raise SystemExit(
                    f"{label}: {conv} per-item mean {arr.mean()} != reported {claimed}")
    assert ids_ref is not None
    return sorted(lenient), lenient, strict


def _boot_accuracies(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(N_BOOT, n_models) bootstrap accuracies from one shared item resample per draw.

    Multinomial counts rather than index sampling: one matmul per chunk instead of a Python loop,
    and identical in distribution.
    """
    n_items = matrix.shape[1]
    out = np.empty((N_BOOT, matrix.shape[0]), dtype=np.float64)
    done = 0
    while done < N_BOOT:
        size = min(CHUNK, N_BOOT - done)
        counts = rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=size)
        out[done:done + size] = counts @ matrix.T / n_items
        done += size
    return out


def _ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _mean_abs_rank_error(fixed: dict[str, int], criterion_scores: dict[str, float]) -> float:
    crit = _ranks(criterion_scores, lower_is_better=False)
    return sum(abs(fixed[m] - crit[m]) for m in fixed) / len(fixed)


def build() -> dict:
    models, lenient, strict = _load_items()
    rng = np.random.default_rng(SEED)

    # The mechanism claim (Table 2) is the eleven-model published cohort; the headline selector
    # comparison is all seventeen. Both populations are reported so neither has to be inferred.
    eleven = sorted(json.loads((ROOT / "results" / "cohort_extension.json").read_text())
                    ["published_cohort"])

    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "question": "How precisely is the in-domain cloze criterion itself measured, and does it "
                    "resolve the adjacent orderings the paper's rank claims rely on?",
        "method": {
            "resampling_unit": "cloze item (2,500 shared items), NOT model",
            "answers": "how well the yardstick is known; the paper's selection-accuracy intervals "
                       "resample models instead and answer whether a selector generalises",
            "paired": "one item resample is applied to every model per draw, so between-model "
                      "differences do not pay the item-difficulty noise twice",
            "n_boot": N_BOOT, "seed": SEED, "corpus": CORPUS,
            "tie_convention": "analyze_tier_mechanism._ranks; exact ties broken by model order",
        },
        "n_items": int(next(iter(lenient.values())).shape[0]),
        "cohort_17": models,
        "cohort_11": eleven,
    }

    for conv, table in (("lenient", lenient), ("strict", strict)):
        matrix = np.vstack([table[m] for m in models])
        draws = _boot_accuracies(matrix, rng)
        idx = {m: i for i, m in enumerate(models)}

        per_model = {}
        for m in models:
            lo, hi = _ci(draws[:, idx[m]])
            per_model[m] = {"accuracy": float(matrix[idx[m]].mean()),
                            "ci_lo": lo, "ci_hi": hi,
                            "half_width": (hi - lo) / 2}

        # Adjacent pairs in the criterion's own ordering: the question the audit actually asked.
        order = sorted(eleven, key=lambda m: -per_model[m]["accuracy"])
        adjacent = []
        for a, b in zip(order, order[1:]):
            diff = draws[:, idx[a]] - draws[:, idx[b]]
            lo, hi = _ci(diff)
            adjacent.append({
                "better": a, "worse": b,
                "observed": per_model[a]["accuracy"] - per_model[b]["accuracy"],
                "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0),
            })

        # Every pair, so the decision-rule tool can strip the yardstick's noise out of a
        # selection-accuracy figure the same way Appendix C does for benchmark standard errors.
        pairwise = {}
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                diff = draws[:, idx[a]] - draws[:, idx[b]]
                lo, hi = _ci(diff)
                pairwise["%s|%s" % (a, b)] = {
                    "observed": per_model[a]["accuracy"] - per_model[b]["accuracy"],
                    "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0)}

        result[conv] = {"per_model": per_model, "adjacent_pairs": adjacent,
                        "n_adjacent_resolved": sum(p["ci_excludes_zero"] for p in adjacent),
                        "pairwise": pairwise}

        if conv == "lenient":
            result["lenient"]["rank_error"] = _rank_error_block(draws, idx, eleven, per_model)
    return result


def _rank_error_block(draws: np.ndarray, idx: dict[str, int], eleven: list[str],
                      per_model: dict) -> dict:
    """CIs on the paper's 0.36 / 2.00 mean-rank-error claim.

    The two BPB orderings are FIXED (they are what the paper proposes); only the criterion ordering
    is resampled. So this interval answers "how much of the 0.36-vs-2.00 gap could be the yardstick
    moving", which is exactly the audit's objection.
    """
    from analyze_alignment_matrix import load_bpb_matrix
    bpb, _ = load_bpb_matrix()
    fixed = {
        "adapted": _ranks({m: bpb[CORPUS]["adapted_bpb"][m] for m in eleven}, lower_is_better=True),
        "zero_shot": _ranks({m: bpb[CORPUS]["zero_shot_bpb"][m] for m in eleven}, lower_is_better=True),
    }
    observed = {k: _mean_abs_rank_error(v, {m: per_model[m]["accuracy"] for m in eleven})
                for k, v in fixed.items()}

    cols = [idx[m] for m in eleven]
    sub = draws[:, cols]
    errs = {k: np.empty(sub.shape[0]) for k in fixed}
    exact = {k: np.empty(sub.shape[0]) for k in fixed}
    for d in range(sub.shape[0]):
        scores = {m: sub[d, j] for j, m in enumerate(eleven)}
        crit = _ranks(scores, lower_is_better=False)
        for k, ranks in fixed.items():
            diffs = [abs(ranks[m] - crit[m]) for m in eleven]
            errs[k][d] = sum(diffs) / len(diffs)
            exact[k][d] = sum(1 for x in diffs if x == 0)

    out = {"n_models": len(eleven), "note": "criterion ordering resampled; BPB orderings fixed"}
    for k in fixed:
        lo, hi = _ci(errs[k])
        out[k] = {"observed_mean_abs_rank_error": observed[k], "ci_lo": lo, "ci_hi": hi,
                  "median_models_ranked_exactly_right": float(np.median(exact[k]))}
    gap = errs["zero_shot"] - errs["adapted"]
    lo, hi = _ci(gap)
    out["zero_shot_minus_adapted"] = {
        "observed": observed["zero_shot"] - observed["adapted"],
        "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="fail if the artifact is stale or point estimates do not reproduce")
    args = ap.parse_args()

    built = build()
    problems = []

    # Reproduce before you interval: the published mechanism numbers must fall out of this code.
    tier = json.loads((ROOT / "results" / "tier_mechanism.json").read_text())
    for key in ("zero_shot", "adapted"):
        claimed = tier["mean_abs_rank_error"][key]
        got = built["lenient"]["rank_error"][key]["observed_mean_abs_rank_error"]
        if abs(claimed - got) > 1e-9:
            problems.append(f"mean_abs_rank_error[{key}]: {got} != published {claimed}")
    validity = json.loads((ROOT / "results" / "cloze_validity.json").read_text())
    for model, claimed in validity["cloze_accuracy"].items():
        got = built["lenient"]["per_model"][model]["accuracy"]
        if abs(claimed - got) > 1e-9:
            problems.append(f"cloze_accuracy[{model}]: {got} != published {claimed}")

    if args.check:
        if not OUTPUT_JSON.exists():
            problems.append("results/criterion_uncertainty.json is missing")
        else:
            on_disk = json.loads(OUTPUT_JSON.read_text())
            if json.dumps(on_disk, sort_keys=True) != json.dumps(built, sort_keys=True):
                problems.append("results/criterion_uncertainty.json is stale")
        for p in problems:
            print("  " + p)
        print("criterion uncertainty: PROBLEMS", len(problems))
        sys.exit(1 if problems else 0)

    if problems:
        for p in problems:
            print("  " + p)
        print("criterion uncertainty: PROBLEMS", len(problems))
        sys.exit(1)
    OUTPUT_JSON.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n")
    print("wrote", OUTPUT_JSON.relative_to(ROOT))


if __name__ == "__main__":
    main()

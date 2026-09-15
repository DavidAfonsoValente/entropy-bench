#!/usr/bin/env python3
"""Does the recommendation hold for EVERY corpus, or only for the one we looked at first?

`analyze_cloze_validity.py` answers the selector question against the news criterion. That is the
corpus every headline number in the paper is measured on, which makes it the right place to start
and the wrong place to stop: a method sold as "measure the models on YOUR text" has to be shown to
work on more than one kind of text.

This runs the identical comparison once per corpus, each against its OWN in-domain cloze criterion,
with the domain-matched adapted tier as the comparand -- fixed by design in every case, never
chosen by looking at which scored highest. The reported question per corpus is the one the
recommendation actually rests on:

  Among candidates within PRIMARY_BAND in size, does the domain-matched adapted BPB clear chance
  against that corpus's own criterion, and do the public benchmarks?

Nothing here is pooled across corpora. The four criteria are built from four different corpora with
different item counts, span compositions and document counts, and the same eleven models appear in
all four, so a pooled statistic would be neither independent nor comparable. The output is four
separate answers plus a count of how many came out the same way, which is a description and not a
test.

    python tools/analyze_cloze_coverage.py [--check]

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _artifact_check import _diff  # noqa: E402
from analyze_cloze_validity import (  # noqa: E402
    BENCHMARKS, MODEL_IDS, PRIMARY_BAND, REPS, ROOT, SEED, SELECTOR_STATIC,
    _bootstrap_tables, _load_params, _verdicts, load_bpb_matrix, load_cloze, load_raw_static,
)

OUTPUT_JSON = ROOT / "results" / "cloze_coverage.json"
SCHEMA_VERSION = 2
# corpus -> (results directory, filename glob). News keeps the original unprefixed layout.
CORPORA = {
    "news": ("cloze", "cloze__*.json"),
    "math": ("cloze_math", "cloze_math__*.json"),
    "reddit": ("cloze_reddit", "cloze_reddit__*.json"),
    "hackernews": ("cloze_hackernews", "cloze_hackernews__*.json"),
}


def _selectors(bpb, corpus, params, acc):
    """The seven selectors, with the domain-matched adapted tier named for this corpus."""
    return {
        "zero_shot_bpb": (bpb[corpus]["zero_shot_bpb"], True),
        "matched_adapted_bpb": (bpb[corpus]["adapted_bpb"], True),
        "gsm8k": (acc["gsm8k"], False),
        "mmlu_pro": (acc["mmlu_pro"], False),
        "hellaswag": (acc["hellaswag"], False),
        "parameter_count": (params, False),
    }


def build() -> dict:
    bpb, _ = load_bpb_matrix()
    params = _load_params()
    acc = {b: {m: load_raw_static(m, b)[0] for m in MODEL_IDS} for b in BENCHMARKS}

    out = {}
    for corpus, (subdir, pattern) in CORPORA.items():
        d = ROOT / "results" / subdir
        if not d.exists():
            continue
        cloze = {m: load_cloze(d, m, pattern) for m in ("lenient_accuracy", "strict_accuracy")}
        models = sorted(cloze["lenient_accuracy"])
        sels = _selectors(bpb, corpus, params, acc)
        entry = {"n_models": len(models), "models": models,
                 "n_items": None, "comparand": "matched_adapted_bpb"}
        # Item count and composition, recorded per corpus because they are NOT comparable and the
        # prose has to say so wherever two corpora are put side by side.
        first = sorted(d.glob(pattern))[0]
        row = json.loads(first.read_text())
        entry["n_items"] = row["cloze_n_items"]
        entry["items_by_kind"] = {k: v["n"] for k, v in row["by_kind"].items()}

        for metric, key in (("lenient_accuracy", "lenient"), ("strict_accuracy", "strict")):
            for band, scope in ((None, "all_pairs"), (PRIMARY_BAND, f"within_{PRIMARY_BAND:g}x")):
                boot = _bootstrap_tables(sels, cloze[metric], models, params, band,
                                         compared="matched_adapted_bpb")
                t = boot["selectors"]
                entry.setdefault(key, {})[scope] = {
                    "selectors": t,
                    "clears_chance": sorted(k for k, v in t.items() if v["beats_chance"]),
                    # Recorded separately: a selector that orders every in-band pair correctly
                    # produces a [1.000, 1.000] interval, which the naive test would count.
                    "degenerate_intervals": sorted(
                        k for k, v in t.items() if v["interval_is_degenerate"]),
                    "matched_clears_chance": t["matched_adapted_bpb"]["beats_chance"],
                    "any_benchmark_clears_chance": any(t[b]["beats_chance"] for b in SELECTOR_STATIC),
                    "zero_shot_clears_chance": t["zero_shot_bpb"]["beats_chance"],
                    "paired_differences": boot["paired_differences"],
                    "common_pair_mask": boot["common_pair_mask"],
                    "n_pairs": t["matched_adapted_bpb"]["n_pairs"],
                    **{k: v for k, v in _verdicts(
                        t, "matched_adapted_bpb",
                        ("zero_shot_bpb", "matched_adapted_bpb")).items()
                       if k in ("bpb_beats_every_static_benchmark", "bpb_beats_parameter_count")},
                }
        out[corpus] = entry

    scope = f"within_{PRIMARY_BAND:g}x"

    def _corpora(pred):
        return sorted(c for c in out if pred(out[c]))

    summary = {
        "corpora_measured": sorted(out),
        # The eligibility mask is computed per metric, because each convention ties a different set
        # of pairs. Where the two masks differ, "holds under both conventions" compares two
        # different pair populations and not one population scored two ways -- so the fact is
        # recorded rather than assumed away.
        "conventions_share_a_pair_mask": _corpora(
            lambda e: e["lenient"][scope]["n_pairs"] == e["strict"][scope]["n_pairs"]),
        "pairs_by_corpus": {c: {"lenient": out[c]["lenient"][scope]["n_pairs"],
                                "strict": out[c]["strict"][scope]["n_pairs"]} for c in out},
        # Counts, not tests. The same eleven models appear in all four, so these are four views of
        # one cohort and cannot be combined into a significance statement.
        "matched_clears_chance_lenient": _corpora(
            lambda e: e["lenient"][scope]["matched_clears_chance"]),
        "matched_clears_chance_strict": _corpora(
            lambda e: e["strict"][scope]["matched_clears_chance"]),
        "any_benchmark_clears_chance_lenient": _corpora(
            lambda e: e["lenient"][scope]["any_benchmark_clears_chance"]),
        "any_benchmark_clears_chance_strict": _corpora(
            lambda e: e["strict"][scope]["any_benchmark_clears_chance"]),
        "parameter_count_clears_chance_either": _corpora(
            lambda e: any(e[k][scope]["selectors"]["parameter_count"]["beats_chance"]
                          for k in ("lenient", "strict"))),
        # POINT ESTIMATES. Higher observed accuracy than each benchmark, with no requirement that
        # any paired difference exclude zero -- named accordingly so the field cannot be quoted as
        # an inferential result.
        "matched_higher_point_estimate_than_every_benchmark_both_conventions": sorted(
            c for c in out
            if out[c]["lenient"][scope]["bpb_beats_every_static_benchmark"]
            and out[c]["strict"][scope]["bpb_beats_every_static_benchmark"]),
        "matched_higher_point_estimate_than_size_both_conventions": sorted(
            c for c in out
            if out[c]["lenient"][scope]["bpb_beats_parameter_count"]
            and out[c]["strict"][scope]["bpb_beats_parameter_count"]),
        "zero_shot_clears_chance_lenient": _corpora(
            lambda e: e["lenient"][scope]["zero_shot_clears_chance"]),
        "zero_shot_clears_chance_strict": _corpora(
            lambda e: e["strict"][scope]["zero_shot_clears_chance"]),
        # The size comparison as a plain inventory of signs. It is 8 dependent contrasts over one
        # cohort of 11 models, so it is a description and supports no test; the count is recorded
        # exactly so the prose reporting it cannot drift from the data.
        "size_contrast_signs": {
            "positive": sum(1 for c in out for k in ("lenient", "strict")
                            if out[c][k][scope]["paired_differences"]
                            ["bpb_minus_parameter_count"]["observed"] > 0),
            "negative": sum(1 for c in out for k in ("lenient", "strict")
                            if out[c][k][scope]["paired_differences"]
                            ["bpb_minus_parameter_count"]["observed"] < 0),
            "zero": sum(1 for c in out for k in ("lenient", "strict")
                        if out[c][k][scope]["paired_differences"]
                        ["bpb_minus_parameter_count"]["observed"] == 0),
            "any_separates_from_zero": any(
                out[c][k][scope]["paired_differences"]["bpb_minus_parameter_count"]
                ["separates_from_zero"] for c in out for k in ("lenient", "strict")),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "question": "Does the in-band recommendation hold for every corpus, each judged against "
                    "its own in-domain criterion?",
        "method": f"pairwise selection accuracy; cluster bootstrap over models, {REPS} draws, "
                  f"seed {SEED}; one shared stream and one shared eligibility mask per table. "
                  f"NOT pooled across corpora -- four criteria of different sizes and "
                  f"compositions over the same eleven models are four views of one cohort.",
        "primary_band": PRIMARY_BAND,
        "per_corpus": out,
        "summary": summary,
    }


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
        print("cloze coverage artifact is up to date")
        return

    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    scope = f"within_{PRIMARY_BAND:g}x"
    print(f"In-band ({scope}) selector accuracy, each corpus against its OWN criterion\n")
    hdr = f"{'corpus':<12}{'items':>7}{'pairs':>7}  {'matched':>9}{'zero-shot':>11}" \
          f"{'best bench':>12}{'size':>8}"
    print(hdr); print("-" * len(hdr))
    for corpus, e in report["per_corpus"].items():
        t = e["lenient"][scope]["selectors"]
        best_b = max(SELECTOR_STATIC, key=lambda b: t[b]["pairwise_accuracy"])
        star = "*" if t["matched_adapted_bpb"]["beats_chance"] else " "
        print(f"{corpus:<12}{e['n_items']:>7}{e['lenient'][scope]['n_pairs']:>7}  "
              f"{t['matched_adapted_bpb']['pairwise_accuracy']:>8.3f}{star}"
              f"{t['zero_shot_bpb']['pairwise_accuracy']:>11.3f}"
              f"{t[best_b]['pairwise_accuracy']:>12.3f}"
              f"{t['parameter_count']['pairwise_accuracy']:>8.3f}")
    print("\n* interval clears chance (lenient scoring)\n")
    for k, v in report["summary"].items():
        print(f"  {k:<48}{v}")
    print(f"\nwrote {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

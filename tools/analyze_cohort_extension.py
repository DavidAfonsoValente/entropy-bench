#!/usr/bin/env python3
"""E3: does the in-band conclusion survive a cohort large enough to resolve anything?

Every headline in this study is "direction supported, magnitude not", and all of them have one
cause: eleven models give eleven model pairs within 2x in size, and at that size a selector must be
near-perfect for its interval to exclude chance. Across the four-corpus comparison -- 4 corpora x 2
scoring conventions x 7 selectors -- exactly one interval did.

This pools the published eleven with the nine preregistered extension models (docs/RUN_LEDGER.md,
"E3 -- cohort extension: PREREGISTRATION") and re-asks the four questions fixed there, on the news
criterion, in the 2x band. The cohort, the exclusion rule and both scoring conventions were
committed before the first weight was staged.

**The comparability argument, because pooling two cohorts is the thing that could invalidate this.**
Three measurements are pooled and each has its own reason:

* *BPB* is a deterministic forward pass. The run ledger records a Ministral-3-14B cell reproducing
  to four decimals on a different host and date. The extension's ADAPTED BPB additionally depends
  on a training run executed on a different software stack, which is an assumption rather than a
  fact -- so a published cell (Qwen2.5-7B) was re-run through the extension pipeline as a
  reproduction control, and this tool REFUSES to pool unless it reproduces within tolerance.
* *Benchmarks* were scored for the extension on the same H100 host and vLLM build as the published
  eleven, because sec:adaptedacc measures identical weights moving 0.60-0.69 GSM8K points between
  hosts.
* *Cloze* was scored on the same Blackwell host and vLLM build as the published cloze numbers.

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
    _band_pair_count, _bootstrap_tables, _load_params, _verdicts, load_bpb_matrix, load_cloze,
    load_raw_static,
)

OUTPUT_JSON = ROOT / "results" / "cohort_extension.json"
EXT_DIR = ROOT / "results" / "cohort_ext"
SCHEMA_VERSION = 1
CONTROL_LABEL = "CONTROL-Qwen2.5-7B"
CONTROL_OF = "Qwen2.5-7B"
# How far the re-run control may drift from its published adapted BPB before the two cohorts are
# declared incomparable. 1% of the published value: the published cross-host reproduction matched
# to four decimals, so anything approaching a percent is a stack effect worth refusing to pool over.
CONTROL_TOL_FRAC = 0.01


def _load_ext():
    """The extension cells, and nothing inferred: a model appears only if every file exists."""
    dt = {}
    for f in sorted((EXT_DIR / "dt").glob("news__*.json")):
        d = json.loads(f.read_text())
        dt[d["label"]] = d
    bench = {}
    for f in sorted((EXT_DIR / "bench").glob("*__*.json")):
        b, label = f.stem.split("__", 1)
        bench.setdefault(label, {})[b] = json.loads(f.read_text())["score"]
    cloze = {}
    for f in sorted((EXT_DIR / "cloze").glob("cloze__*.json")):
        d = json.loads(f.read_text())
        cloze[d["label"]] = d
    return dt, bench, cloze


def control_check(dt) -> dict:
    """Did a published cell reproduce when re-run through the extension pipeline?"""
    if CONTROL_LABEL not in dt:
        return {"ran": False, "reproduces": None,
                "note": "control cell absent; the cohorts may not be pooled"}
    pub = json.loads((ROOT / "results" / "domain_transfer" /
                      "news__Qwen_Qwen2_5-7B.json").read_text())
    got = dt[CONTROL_LABEL]
    out = {"ran": True, "control_of": CONTROL_OF, "tolerance_frac": CONTROL_TOL_FRAC}
    for k in ("zero_shot_bpb", "adapted_bpb"):
        rel = abs(got[k] - pub[k]) / pub[k]
        out[k] = {"published": pub[k], "re_run": got[k], "rel_diff": rel,
                  "within_tolerance": rel <= CONTROL_TOL_FRAC}
    out["reproduces"] = all(out[k]["within_tolerance"] for k in ("zero_shot_bpb", "adapted_bpb"))
    return out


def build() -> dict:
    dt, bench, cloze_ext = _load_ext()
    ctl = control_check(dt)

    # A model enters the cohort on BPB + cloze alone. Three of the four preregistered questions --
    # whether the matched tier clears chance, whether the free tier does, and the matched-vs-size
    # difference -- involve no public benchmark, so gating the whole analysis on the benchmark
    # column would withhold answers the data already supports. The benchmark SELECTORS are then
    # included only if every model in the pooled cohort has them, because a selector scored on a
    # subset would be compared against selectors scored on everyone.
    ext_labels = sorted(L for L in dt if L != CONTROL_LABEL and L in cloze_ext)
    incomplete = sorted(set(dt) - set(ext_labels) - {CONTROL_LABEL})
    need = ("gsm8k", "hellaswag", "mmlu_pro_1k")
    bench_ready = bool(ext_labels) and all(
        all(b in bench.get(L, {}) for b in need) for L in ext_labels)
    bench_missing = sorted(
        f"{L}:{b}" for L in ext_labels for b in need if b not in bench.get(L, {}))

    bpb_pub, _ = load_bpb_matrix()
    params = dict(_load_params())
    acc = {b: {m: load_raw_static(m, b)[0] for m in MODEL_IDS} for b in BENCHMARKS}
    cl = {m: load_cloze(ROOT / "results" / "cloze", m)
          for m in ("lenient_accuracy", "strict_accuracy")}

    zero = dict(bpb_pub["news"]["zero_shot_bpb"])
    adapt = dict(bpb_pub["news"]["adapted_bpb"])
    lenient = dict(cl["lenient_accuracy"])
    strict = dict(cl["strict_accuracy"])
    bench_map = {"gsm8k": "gsm8k", "hellaswag": "hellaswag", "mmlu_pro_1k": "mmlu_pro"}
    accs = {v: dict(acc[v]) for v in bench_map.values()}

    for L in ext_labels:
        zero[L] = dt[L]["zero_shot_bpb"]
        adapt[L] = dt[L]["adapted_bpb"]
        params[L] = dt[L]["total_params"]
        lenient[L] = cloze_ext[L]["lenient_accuracy"]
        strict[L] = cloze_ext[L]["strict_accuracy"]
        if bench_ready:
            for raw, key in bench_map.items():
                accs[key][L] = bench[L][raw]

    models = sorted(set(lenient) & set(adapt) & set(params))
    selectors = {
        "zero_shot_bpb": (zero, True),
        "matched_adapted_bpb": (adapt, True),
        "parameter_count": (params, False),
    }
    if bench_ready:
        for key in ("gsm8k", "mmlu_pro", "hellaswag"):
            selectors[key] = (accs[key], False)

    out = {
        "schema_version": SCHEMA_VERSION,
        "question": "Does the in-band conclusion survive a cohort large enough to resolve it?",
        "preregistration": 'docs/RUN_LEDGER.md, "E3 -- cohort extension: PREREGISTRATION"',
        "method": f"pairwise selection accuracy against the news in-domain cloze criterion; "
                  f"cluster bootstrap over models, {REPS} draws, seed {SEED}; pairs NOT resampled; "
                  f"one shared stream and one shared eligibility mask per table; degenerate "
                  f"[x, x] intervals do not count as clearing chance.",
        "reproduction_control": ctl,
        "published_cohort": sorted(MODEL_IDS),
        "extension_cohort": ext_labels,
        "extension_incomplete": incomplete,
        "n_models": len(models),
        "pairs_all": len(models) * (len(models) - 1) // 2,
        "benchmarks_included": bench_ready,
        "benchmark_cells_missing": bench_missing,
    }
    if not ctl.get("reproduces"):
        # Refusing to pool is the correct outcome, not an error: without the control the extension
        # cells and the published cells are measurements on different software stacks.
        out["pooled"] = False
        out["reason"] = "reproduction control absent or outside tolerance; cohorts not pooled"
        return out

    out["pooled"] = True
    for band, scope in ((None, "all_pairs"), (PRIMARY_BAND, f"within_{PRIMARY_BAND:g}x")):
        blk = {}
        for metric, key in ((lenient, "lenient"), (strict, "strict")):
            boot = _bootstrap_tables(selectors, metric, models, params, band,
                                     compared="matched_adapted_bpb")
            t = boot["selectors"]
            v = (_verdicts(t, "matched_adapted_bpb",
                           ("zero_shot_bpb", "matched_adapted_bpb")) if bench_ready
                 else {"bpb_beats_every_static_benchmark": None,
                       "bpb_beats_parameter_count":
                           t["matched_adapted_bpb"]["pairwise_accuracy"]
                           > t["parameter_count"]["pairwise_accuracy"]})
            blk[key] = {
                "selectors": t,
                "paired_differences": boot["paired_differences"],
                "common_pair_mask": boot["common_pair_mask"],
                "n_pairs": t["matched_adapted_bpb"]["n_pairs"],
                "clears_chance": sorted(k for k, x in t.items() if x["beats_chance"]),
                "degenerate_intervals": sorted(
                    k for k, x in t.items() if x["interval_is_degenerate"]),
                **{k: v[k] for k in ("bpb_beats_every_static_benchmark",
                                     "bpb_beats_parameter_count")},
            }
        if band is not None:
            blk["n_band_pairs"] = _band_pair_count(models, params, band)
        out[scope] = blk

    scope = f"within_{PRIMARY_BAND:g}x"
    # The four questions, answered in the order the preregistration fixed them.
    out["preregistered_answers"] = {
        "q1_matched_clears_chance": {
            k: out[scope][k]["selectors"]["matched_adapted_bpb"]["beats_chance"]
            for k in ("lenient", "strict")},
        "q2_any_benchmark_clears_chance": ({
            k: any(out[scope][k]["selectors"][b]["beats_chance"] for b in SELECTOR_STATIC)
            for k in ("lenient", "strict")} if bench_ready else
            "PENDING: benchmark column incomplete for the extension cohort"),
        "q3_zero_shot_clears_chance": {
            k: out[scope][k]["selectors"]["zero_shot_bpb"]["beats_chance"]
            for k in ("lenient", "strict")},
        "q4_paired_differences": {
            k: {name: {kk: d[kk] for kk in
                       ("observed", "ci_lo", "ci_hi", "ci_excludes_zero",
                        "supports_positive_effect", "supports_negative_effect")}
                for name, d in out[scope][k]["paired_differences"].items()}
            for k in ("lenient", "strict")},
    }
    return out


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
        print("cohort extension artifact is up to date")
        return

    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    ctl = report["reproduction_control"]
    print(f"reproduction control: ran={ctl.get('ran')} reproduces={ctl.get('reproduces')}")
    if ctl.get("ran"):
        for k in ("zero_shot_bpb", "adapted_bpb"):
            c = ctl[k]
            print(f"  {k:<16}published {c['published']:.4f}  re-run {c['re_run']:.4f}  "
                  f"rel {c['rel_diff']:.2%}  ok={c['within_tolerance']}")
    if not report.get("pooled"):
        print("NOT POOLED:", report.get("reason"))
        print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
        return
    scope = f"within_{PRIMARY_BAND:g}x"
    print(f"\ncohort {report['n_models']} models; in-band pairs "
          f"{report[scope]['n_band_pairs']} (was 11)\n")
    for key in ("lenient", "strict"):
        blk = report[scope][key]
        print(f"--- {key}, {blk['n_pairs']} in-band pairs")
        for n, r in sorted(blk["selectors"].items(), key=lambda kv: -kv[1]["pairwise_accuracy"]):
            mark = "*" if r["beats_chance"] else (" d" if r["interval_is_degenerate"] else "  ")
            print(f"    {n:<22}{r['pairwise_accuracy']:.4f} "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]{mark}")
        for name, d in blk["paired_differences"].items():
            print(f"    {name:<30}{d['observed']:+.4f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] "
                  f"excludes0={d['ci_excludes_zero']}")
    print("\n* interval excludes chance   d degenerate interval (not counted)")
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

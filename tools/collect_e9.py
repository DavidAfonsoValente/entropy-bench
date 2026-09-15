#!/usr/bin/env python3
"""Collect the E9 benchmark column and validate it before anything downstream reads it.

E9 re-scores all 17 models x 3 benchmarks on one host and one build, replacing the July
column in full (docs/RUN_LEDGER.md, "E9"). This does the collection and the checks that
decide whether the new column is trustworthy, and it refuses to publish a partial or mixed
one.

Three checks, in the order they can fail:

  provenance  exactly 51 cells, one per (label, benchmark), no duplicates, no strays, and
              every cell carrying the same lm_eval version and the same GPU string. A cell
              keyed by array position rather than identity is how a correct number ends up
              attached to the wrong model.
  identity    task_hash and task version equal to July's. These are computed from the doc,
              prompt and target text of every logged sample, so they are invariant to host
              and backend and pin the task DEFINITION -- the one thing that can change
              silently and still produce a plausible score.
  protocol    every cell carries a gen_audit.json that passes tools/check_gen_audit.py --
              the decode protocol was IMPOSED on this column (eval/lm_eval_uniform.py), and
              a cell scored without it may have inherited a checkpoint's own max_new_tokens.
              Per-cell gating already happens in the runner; this is the column-level check
              that no cell reached results/ by another route.
  agreement   the 11 models present in both columns, gated at the PREREGISTERED tolerance:
              1.0 absolute point on gsm8k and hellaswag, HALT if more than one model is
              outside it. The backend changed (vLLM -> hf, forced: no vLLM wheel supporting
              transformers 5 runs on this driver) and the amendment names that deviation,
              but the tolerance and the HALT rule were both fixed in advance and are not
              renegotiable once scores exist. mmlu_pro_1k is excluded from the tolerance
              gate because the preregistration excluded it.

    python tools/collect_e9.py --dry-run     # validate, print, write nothing
    python tools/collect_e9.py               # validate, then stage into results/
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = json.loads((ROOT / "eval" / "july_reproduction_spec.json").read_text())
BENCHES = ("gsm8k", "hellaswag", "mmlu_pro_1k")
METRIC = {"gsm8k": "exact_match,flexible-extract",
          "hellaswag": "acc_norm,none",
          "mmlu_pro_1k": "exact_match,custom-extract"}


def load_cells(cell_root: Path) -> dict[tuple[str, str], dict]:
    cells: dict[tuple[str, str], dict] = {}
    for results in sorted(cell_root.glob("*/*/**/results_*.json")):
        # <cell_root>/<label>/<bench>/<...>/results_*.json
        rel = results.relative_to(cell_root).parts
        label, bench = rel[0], rel[1]
        if bench not in BENCHES:
            raise SystemExit(f"FATAL: unexpected benchmark dir {bench} in {results}")
        key = (label, bench)
        if key in cells:
            raise SystemExit(f"FATAL: duplicate cell for {label} x {bench}; "
                             f"provenance must be one file per cell")
        cells[key] = json.loads(results.read_text())
    return cells


def july_scores() -> dict[tuple[str, str], float]:
    out = {}
    for bench in BENCHES:
        for d in sorted((ROOT / "results" / bench).glob("*")):
            if not d.is_dir():
                continue
            files = sorted(d.glob("*/results_*.json"))
            if not files:
                continue
            data = json.loads(files[-1].read_text())
            task = "mmlu_pro_1k" if bench == "mmlu_pro_1k" else bench
            res = data["results"].get(task)
            if res and METRIC[bench] in res:
                out[(d.name, bench)] = res[METRIC[bench]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None, help="E9 cell root (default $SCRATCH/e9/cells)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import os
    root = Path(a.cells) if a.cells else Path(os.environ.get("SCRATCH", "/tmp")) / "e9" / "cells"
    if not root.is_dir():
        raise SystemExit(f"FATAL: no cell root at {root}")
    cells = load_cells(root)

    # --- provenance -------------------------------------------------------------
    labels = sorted({k[0] for k in cells})
    expected = {(l, b) for l in labels for b in BENCHES}
    missing = sorted(expected - set(cells))
    print(f"cells found: {len(cells)}  models: {len(labels)}")
    if missing:
        print(f"MISSING {len(missing)} cells: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    versions = {c.get("lm_eval_version") for c in cells.values()}
    # lm_eval writes this as "GPU models and configuration: GPU 0: NVIDIA A100-SXM-64GB", so a
    # startswith("GPU 0:") test never matches and the one-host check silently passes on an
    # empty set. That is the same failure that left this repo believing the July column came
    # from an H100 for four documents; match the marker anywhere in the line instead.
    gpus, drivers = set(), set()
    for c in cells.values():
        for line in (c.get("pretty_env_info") or "").splitlines():
            line = line.strip()
            for n in range(8):
                marker = f"GPU {n}:"
                if marker in line:
                    name = line.split(marker, 1)[1].strip()
                    # An empty name is an unparsed line, not a GPU. Adding it would make the
                    # set non-empty and restore exactly the vacuous check this fix removes.
                    if name:
                        gpus.add(name)
            if "Nvidia driver version:" in line:
                drivers.add(line.split("Nvidia driver version:", 1)[1].strip())
    print(f"lm_eval versions: {versions or '{}'}")
    print(f"GPU strings: {gpus or '{}'}")
    print(f"driver versions: {drivers or '{}'}")
    problems = []
    if len(versions) > 1:
        problems.append(f"mixed lm_eval versions {versions}")
    if not gpus:
        problems.append("no GPU could be parsed from any cell -- the one-host check would be "
                        "vacuous, and a silently vacuous provenance check is how the July "
                        "column was misattributed to an H100")
    if len(gpus) > 1:
        problems.append(f"mixed GPUs {gpus} -- the column must come from ONE host")
    if len(drivers) > 1:
        problems.append(f"mixed drivers {drivers}")

    # A --limit cell is a throughput probe, never a score. Probe B's mmlu_pro_1k cell has
    # limit=5.0 and sits in the same directory layout, so without this it collects cleanly
    # and lands in the column as a 70-item "result".
    limited = sorted(f"{l}x{b}" for (l, b), c in cells.items()
                     if c.get("config", {}).get("limit") is not None)
    if limited:
        print(f"  LIMITED CELLS: {limited[:6]}{' ...' if len(limited) > 6 else ''}")
        problems.append(f"{len(limited)} cells were run with --limit and are not scores")

    # batch_size is a declared deviation from July's pinned value (see the amendment); it is
    # allowed to differ from July, but not to differ BETWEEN cells of this column.
    batches = {c.get("config", {}).get("batch_size") for c in cells.values()}
    print(f"batch sizes: {batches or '{}'}")
    if len(batches) > 1:
        problems.append(f"mixed batch sizes {batches} -- under the hf backend batch size "
                        "affects numerics, so the column must use one value")

    # --- identity ---------------------------------------------------------------
    print("\n-- task identity vs July --")
    for bench in BENCHES:
        want_h = SPEC["benchmarks"][bench].get("task_hashes") or {}
        want_v = SPEC["benchmarks"][bench].get("task_versions") or {}
        got = [(l, c) for (l, b), c in cells.items() if b == bench]
        if not got:
            print(f"  {bench}: no cells yet")
            continue
        bad_h = [l for l, c in got if want_h and c.get("task_hashes", {}) != want_h]
        bad_v = [l for l, c in got if want_v and c.get("versions", {}) != want_v]
        if want_h:
            print(f"  {bench}: task_hash matches July for {len(got)-len(bad_h)}/{len(got)}"
                  + (f"  MISMATCH: {bad_h[:4]}" if bad_h else ""))
            if bad_h:
                problems.append(f"{bench} task_hash mismatch on {len(bad_h)} cells")
        else:
            print(f"  {bench}: July recorded no task_hash (custom task); version check only")
        if bad_v:
            print(f"  {bench}: task VERSION mismatch: {bad_v[:4]}")
            problems.append(f"{bench} task version mismatch on {len(bad_v)} cells")

    # --- protocol ---------------------------------------------------------------
    # A cell can satisfy every check above and still have been scored under a checkpoint's
    # own generation_config, which is the whole reason this column is being re-run.
    print("\n-- decode protocol (gen_audit.json per cell) --")
    checker = ROOT / "tools" / "check_gen_audit.py"
    no_audit, bad_audit = [], []
    for (label, bench) in sorted(cells):
        audit = root / label / bench / "gen_audit.json"
        if not audit.exists():
            no_audit.append(f"{label}x{bench}")
            continue
        if subprocess.call([sys.executable, str(checker), str(audit), bench],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            bad_audit.append(f"{label}x{bench}")
    print(f"  {len(cells) - len(no_audit) - len(bad_audit)}/{len(cells)} cells carry a passing "
          f"protocol audit")
    if no_audit:
        print(f"  NO AUDIT: {no_audit[:6]}{' ...' if len(no_audit) > 6 else ''}")
        problems.append(f"{len(no_audit)} cells have no gen_audit.json -- protocol unverifiable")
    if bad_audit:
        print(f"  FAILED AUDIT: {bad_audit[:6]}{' ...' if len(bad_audit) > 6 else ''}")
        problems.append(f"{len(bad_audit)} cells failed the protocol audit")

    # --- agreement (gated at the PREREGISTERED tolerance) -----------------------
    print("\n-- overlap with July (backend vLLM -> hf; prereg tolerance 1.0 pp, HALT at >1) --")
    july = july_scores()
    # docs/RUN_LEDGER.md, "E9 ... PREREGISTRATION": each of the 11 overlapping models on
    # gsm8k and hellaswag within 1.0 absolute point; more than one outside -> HALT.
    TOLERANCE_PP = 1.0
    GATED = ("gsm8k", "hellaswag")
    EXPECTED_OVERLAP = 11
    outliers = []
    # The gate is only as strong as the set it runs on. A renamed label would not error --
    # it would quietly shrink the overlap and weaken the HALT rule, which is the same
    # silently-vacuous-check failure as the GPU parsing above. Assert the count.
    overlap_labels = {l for (l, b) in cells if (l, b) in july}
    print(f"  overlap set: {len(overlap_labels)} models (preregistration expects "
          f"{EXPECTED_OVERLAP})")
    if len(cells) == 51 and len(overlap_labels) != EXPECTED_OVERLAP:
        problems.append(
            f"overlap set is {len(overlap_labels)} models, not {EXPECTED_OVERLAP} -- a label "
            f"mismatch would silently weaken the preregistered agreement gate. Present: "
            f"{sorted(overlap_labels)}")
    for bench in BENCHES:
        rows = []
        for (l, b), c in sorted(cells.items()):
            if b != bench or (l, bench) not in july:
                continue
            task = "mmlu_pro_1k" if bench == "mmlu_pro_1k" else bench
            new = c["results"].get(task, {}).get(METRIC[bench])
            if new is None:
                continue
            rows.append((l, july[(l, bench)], new))
        if not rows:
            print(f"  {bench}: no overlapping cells yet")
            continue
        deltas = [100 * (n - j) for _, j, n in rows]
        print(f"  {bench}: {len(rows)} overlapping models, "
              f"delta pp min {min(deltas):+.2f} max {max(deltas):+.2f} "
              f"mean {sum(deltas)/len(deltas):+.2f}")
        for l, j, n in rows:
            d = 100 * (n - j)
            flag = ""
            if bench in GATED and abs(d) > TOLERANCE_PP:
                outliers.append(f"{l}x{bench} ({d:+.2f} pp)")
                flag = "  <-- OUTSIDE PREREG TOLERANCE"
            print(f"      {l:<20} July {j:.4f}  new {n:.4f}  {d:+.2f} pp{flag}")

    # The preregistration allows exactly one outlier; two is a HALT, not a judgement call.
    if outliers:
        print(f"\n  {len(outliers)} model(s) outside the {TOLERANCE_PP} pp tolerance: "
              + ", ".join(outliers))
        if len(outliers) > 1:
            problems.append(
                f"HALT (preregistered): {len(outliers)} models outside the {TOLERANCE_PP} pp "
                "tolerance -- the re-score stops and nothing is written to the paper until "
                "it is explained")
        else:
            print("  within the preregistered allowance of one; not a HALT, but state it "
                  "wherever the column is reported")

    print("\n" + ("PROBLEMS: " + "; ".join(problems) if problems else "PROBLEMS: none"))
    if missing:
        print(f"INCOMPLETE: {len(missing)} cells missing -- not staging a partial column")
        return 1
    if problems:
        return 1
    if a.dry_run:
        print("dry-run: nothing written")
        return 0

    # --- stage ------------------------------------------------------------------
    legacy = ROOT / "results" / "legacy_july_bench"
    legacy.mkdir(exist_ok=True)
    for bench in BENCHES:
        src = ROOT / "results" / bench
        if src.is_dir():
            shutil.move(str(src), str(legacy / bench))
    print(f"July column moved to {legacy.relative_to(ROOT)}")
    # Per-example samples are excluded on purpose: 2.9 GB across the 17 models, against 383 MB
    # for the entire July column, and $HOME has a 50 GB quota this repo has already hit. They are
    # archived to $WORK/e9_column_samples/ (not $SCRATCH, which is purged ~40d). Everything the
    # docs and the paper quote lives in results_*.json; the samples are forensic evidence.
    staged = 0
    for (label, bench), _ in sorted(cells.items()):
        for results in sorted((root / label / bench).glob("**/results_*.json")):
            dst = ROOT / "results" / bench / label / results.parent.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in results.parent.iterdir():
                if f.is_file() and not f.name.startswith("samples_"):
                    shutil.copy2(f, dst / f.name)
            audit = root / label / bench / "gen_audit.json"
            if audit.exists():
                shutil.copy2(audit, dst / "gen_audit.json")
            staged += 1
    print(f"staged {staged} cells into results/{{gsm8k,hellaswag,mmlu_pro_1k}}/ "
          "(samples excluded -- see $WORK/e9_column_samples/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Recompute the tokenizer-blocking bias bound for the 17-model extended cohort.

The E3 preregistration (docs/RUN_LEDGER.md) fixed that "the byte denominator and the
tokenizer-bias bound must be recomputed for the extended cohort", and it was not done:
`tokenizer_bias_bound` exists only in `results/alignment_matrix.json`, the 11-model artifact.
This matters because the extension widens the bytes-per-token spread -- SmolLM2's 49,152-entry
vocabulary sits outside the original cohort's range -- and the bound scales with that spread.

The bound itself is unchanged from `tools/analyze_alignment_matrix.py`, deliberately: this is a
recomputation on a larger cohort, not a new method. Eval blocks are 512 *tokens*, so a model whose
tokenizer packs fewer bytes per token scores less text per block and pays the expensive
block-opening positions over fewer bytes. Treat the bytes-per-block ratio as a change in available
byte-context and apply the measured context sensitivity.

The question it answers: is any adjacent pair's adapted-BPB gap small enough that the bias bound
could account for it -- i.e. could tokenizer blocking, rather than model quality, explain a rank?

    python tools/analyze_tokenizer_bias_ext.py [--check]
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.analyze_alignment_matrix import BPB_PER_CONTEXT_DOUBLING, tokenizer_bias_bound  # noqa: E402

PUBLISHED_DIR = ROOT / "results" / "domain_transfer"
EXTENSION_DIR = ROOT / "results" / "cohort_ext" / "dt"
OUT = ROOT / "results" / "tokenizer_bias_ext.json"

# Excluded from E3 by the preregistered rule; excluded here for the same reason, so the bound is
# computed on exactly the cohort the +0.250 result rests on.
E3_EXCLUDED = {"OLMo-2-7B", "OLMo-2-13B", "Qwen2.5-32B"}


# The two cell directories name their files differently (the published cells encode the full
# HF id with separators flattened), so map both onto the slugs cohort_extension.json uses.
_SLUG_FIXUPS = {
    "Qwen_Qwen2_5-0_5B": "Qwen2.5-0.5B", "Qwen_Qwen2_5-1_5B": "Qwen2.5-1.5B",
    "Qwen_Qwen2_5-7B": "Qwen2.5-7B", "Qwen_Qwen3_5-4B-Base": "Qwen3.5-4B",
    "Qwen_Qwen3_5-9B-Base": "Qwen3.5-9B", "Qwen_Qwen3_5-35B-A3B-Base": "Qwen3.5-35B-MoE",
    "meta-llama_Llama-3_2-1B": "Llama-3.2-1B", "google_gemma-4-12B": "gemma-4-12B",
    "google_gemma-4-31B": "gemma-4-31B", "LiquidAI_LFM2_5-1_2B-Base": "LFM2.5-1.2B",
    "mistralai_Ministral-3-14B-Base-2512": "Ministral-3-14B",
}


def _slug(path: Path) -> str:
    """news__Qwen_Qwen2_5-0_5B.json -> Qwen2.5-0.5B, matching cohort_extension.json naming."""
    stem = path.stem.split("__", 1)[1]
    return _SLUG_FIXUPS.get(stem, stem)


def load_cells() -> dict[str, dict]:
    cells: dict[str, dict] = {}
    for directory in (PUBLISHED_DIR, EXTENSION_DIR):
        if not directory.is_dir():
            raise SystemExit(f"FATAL: missing cell directory {directory}")
        for path in sorted(directory.glob("news__*.json")):
            slug = _slug(path)
            if slug in E3_EXCLUDED or slug.startswith("CONTROL-"):
                continue
            data = json.loads(path.read_text())
            for field in ("avg_bytes_per_token", "adapted_bpb", "n_test_blocks_scored"):
                if data.get(field) is None:
                    raise SystemExit(f"FATAL: {path} has no {field}")
            if slug in cells:
                raise SystemExit(f"FATAL: duplicate cell for {slug}")
            cells[slug] = data
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the committed artifact matches a fresh recomputation")
    args = ap.parse_args()

    cells = load_cells()
    if len(cells) != 17:
        raise SystemExit(f"FATAL: expected the 17-model E3 cohort, found {len(cells)}: "
                         f"{sorted(cells)}")

    meta = {k: {"avg_bytes_per_token": v["avg_bytes_per_token"],
                "n_test_blocks_scored": v["n_test_blocks_scored"]}
            for k, v in cells.items()}
    bpb = {k: v["adapted_bpb"] for k, v in cells.items()}
    bound = tokenizer_bias_bound(meta, bpb)

    bpts = {k: v["avg_bytes_per_token"] for k, v in cells.items()}
    lo_model, hi_model = min(bpts, key=bpts.get), max(bpts, key=bpts.get)
    report = {
        "schema_version": 1,
        "question": ("Could token-aligned eval blocks, rather than model quality, account for any "
                     "adjacent rank in the 17-model extended cohort?"),
        "cohort": sorted(cells),
        "n_models": len(cells),
        "excluded": sorted(E3_EXCLUDED),
        "corpus": "news",
        "tier": "adapted",
        "context_sensitivity_per_doubling": BPB_PER_CONTEXT_DOUBLING,
        "bytes_per_token": bpts,
        "bytes_per_token_range": {"min_model": lo_model, "min": bpts[lo_model],
                                  "max_model": hi_model, "max": bpts[hi_model]},
        "bound": bound,
    }

    if args.check:
        if not OUT.exists():
            print(f"FAIL: {OUT.relative_to(ROOT)} does not exist", file=sys.stderr)
            return 1
        committed = json.loads(OUT.read_text())
        if committed != report:
            print(f"FAIL: {OUT.relative_to(ROOT)} does not match a fresh recomputation",
                  file=sys.stderr)
            return 1
        print(f"OK: {OUT.relative_to(ROOT)} reproduces ({len(cells)} models)")
        return 0

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    worst = bound["worst_case"]
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(cells)} models)")
    print(f"bytes/token spread: {bpts[lo_model]:.4f} ({lo_model}) .. "
          f"{bpts[hi_model]:.4f} ({hi_model})")
    print(f"worst bias/gap ratio: {worst['bias_over_gap']:.4f} "
          f"({worst['better']} vs {worst['worse']}); "
          f"overturns_any_rank={bound['overturns_any_rank']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

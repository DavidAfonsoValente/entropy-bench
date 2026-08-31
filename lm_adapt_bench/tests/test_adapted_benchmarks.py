"""Guards on tools/analyze_adapted_benchmarks.py's base-accuracy provenance.

E2 measures deltas of ~1 pp on GSM8K, and the cross-host/engine drift between the July A100 run
and the H100 that produced the adapted numbers was measured at -0.60 to +0.69 pp on 2026-08-28 --
i.e. the confound is the size of the effect. Phase 2 re-scores the released weights on the adapted
run's own host to remove it. These tests pin the behaviour that keeps that meaningful: the
same-host number wins when it exists, the July fallback is recorded rather than silently mixed,
and the verdict refuses to resolve while any delta is still unmatched.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

ab = pytest.importorskip("analyze_adapted_benchmarks")

TASK, METRIC = "gsm8k", "exact_match,flexible-extract"


def _write_results(path: Path, value: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "results_2026-01-01T00-00-00.000000.json").write_text(
        json.dumps({"results": {TASK: {METRIC: value}}}))


def _run_root(tmp_path: Path, label: str, adapted: float, base: float | None) -> Path:
    root = tmp_path / "adapted_bench"
    _write_results(root / "math" / "gsm8k" / f"{label}__math-adapted" / "x", adapted)
    if base is not None:
        _write_results(root / "base" / "gsm8k" / label / "x", base)
    return root


def test_same_host_base_is_preferred_over_july(tmp_path):
    label = "Qwen2.5-1.5B"
    root = _run_root(tmp_path, label, adapted=0.590, base=0.6126)
    row = ab.build(root)["cells"]["math__gsm8k"]["per_model"][label]
    assert row["base_source"] == "same_host"
    assert row["base_accuracy"] == pytest.approx(0.6126)
    assert row["delta_pp"] == pytest.approx((0.590 - 0.6126) * 100)


def test_missing_base_cell_falls_back_and_is_marked(tmp_path):
    label = "Qwen2.5-1.5B"
    july = float(json.loads(ab.STATIC_PATH.read_text())[label]["gsm8k"])
    root = _run_root(tmp_path, label, adapted=0.590, base=None)
    cell = ab.build(root)["cells"]["math__gsm8k"]
    row = cell["per_model"][label]
    assert row["base_source"] == "july_a100"
    assert row["base_accuracy"] == pytest.approx(july)
    assert cell["n_matched"] == 0 and cell["n_models"] == 1


def test_unmatched_row_is_excluded_from_the_cell_aggregates(tmp_path):
    """The July fallback must not contaminate the mean -- that is the confound phase 2 removes."""
    root = tmp_path / "adapted_bench"
    _write_results(root / "math" / "gsm8k" / "Qwen2.5-1.5B__math-adapted" / "x", 0.500)
    _write_results(root / "base" / "gsm8k" / "Qwen2.5-1.5B" / "x", 0.510)   # matched, -1.0 pp
    _write_results(root / "math" / "gsm8k" / "Qwen2.5-7B__math-adapted" / "x", 0.100)  # unmatched
    cell = ab.build(root)["cells"]["math__gsm8k"]
    assert cell["n_models"] == 2 and cell["n_matched"] == 1 and cell["n_aggregated"] == 1
    assert cell["aggregate_basis"] == "same_host"
    assert cell["mean_delta_pp"] == pytest.approx(-1.0)      # not the mean of both rows
    assert set(cell["per_model"]) == {"Qwen2.5-1.5B", "Qwen2.5-7B"}


def test_cell_with_no_matched_row_falls_back_and_says_so(tmp_path):
    root = _run_root(tmp_path, "Qwen2.5-1.5B", adapted=0.590, base=None)
    cell = ab.build(root)["cells"]["math__gsm8k"]
    assert cell["aggregate_basis"] == "july_a100_fallback"
    assert cell["n_matched"] == 0 and cell["n_aggregated"] == 1


def test_verdict_refuses_to_resolve_on_an_unmatched_delta():
    """A full cohort is not enough: an unmatched base makes the delta uninterpretable."""
    n = len(ab.MODEL_IDS)
    per_model = {label: {"base_source": "same_host"} for label in ab.MODEL_IDS}
    per_model[next(iter(ab.MODEL_IDS))]["base_source"] = "july_a100"
    cell = {"complete": True, "n_models": n, "n_matched": n - 1,
            "mean_delta_pp": -3.0, "per_model": per_model}
    out = ab.verdict({"math__gsm8k": cell})
    assert out["resolved"] is False
    assert "July A100" in out["reason"]

    cell["n_matched"] = n
    assert ab.verdict({"math__gsm8k": cell})["resolved"] is True

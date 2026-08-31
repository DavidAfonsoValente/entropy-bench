"""Guards on tools/analyze_selection_divergence.py.

This tool exists to stop an anecdote being promoted into a statistic. Its whole value is the
refutation it carries: the |BPB gap| vs |GSM8K gap| correlation is not estimable at eleven models,
and the artifact has to keep saying so. These tests fail if that self-refutation is ever weakened,
or if the existence proofs stop being existence proofs.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

sd = pytest.importorskip("analyze_selection_divergence")
ARTIFACT = ROOT / "results" / "selection_divergence.json"


@pytest.fixture(scope="module")
def report():
    return json.loads(ARTIFACT.read_text())


def test_every_model_has_a_family():
    from analyze_static_benchmarks import MODEL_IDS
    for label in MODEL_IDS:
        assert sd.family(label)


def test_all_pairs_are_covered_exactly_once(report):
    n = report["n_models"]
    assert report["n_pairs"] == n * (n - 1) // 2 == 55
    seen = {(r["a"], r["b"]) for r in report["pairs"]}
    assert len(seen) == 55


def test_gaps_are_absolute(report):
    for row in report["pairs"]:
        assert row["adapted_bpb_gap_frac"] >= 0
        assert row["gsm8k_gap_pp"] >= 0
        assert row["size_ratio"] >= 1.0


def test_gsm8k_correlation_is_reported_as_not_estimable(report):
    """The refutation is the point. If this ever flips, the prose must change with it."""
    gsm = report["stability"]["gsm8k"]
    assert gsm["estimable"] is False
    assert gsm["max_abs_swing"] > sd.STABILITY_TOLERANCE
    # A single deletion must move it by more than the coefficient's own distance from zero,
    # which is what makes quoting the coefficient indefensible.
    assert gsm["max_abs_swing"] > abs(gsm["full_cohort_pearson"])


def test_hellaswag_correlation_is_stable(report):
    hs = report["stability"]["hellaswag"]
    assert hs["estimable"] is True
    assert hs["min_after_deletion"] > 0.8


def test_leave_one_out_covers_every_model(report):
    from analyze_static_benchmarks import MODEL_IDS
    for bench in report["stability"]:
        assert set(report["stability"][bench]["per_deletion_pearson"]) == set(MODEL_IDS)


def test_witnesses_are_bpb_close_and_benchmark_far(report):
    """Existence proofs must actually be close on BPB, or they prove nothing."""
    assert report["witnesses"]
    for w in report["witnesses"]:
        assert w["adapted_bpb_gap_frac"] <= report["headline_band"]
        assert w["gsm8k_gap_pp"] > w["hellaswag_gap_pp"]


def test_family_split_is_the_claimed_direction(report):
    """Cross-family pairs diverge on GSM8K where same-family pairs do not."""
    split = report["family_split_at_headline_band"]
    assert split["cross_family"]["n_pairs"] > 0 and split["within_family"]["n_pairs"] > 0
    assert (split["cross_family"]["median_gsm8k_gap_pp"]
            > split["within_family"]["median_gsm8k_gap_pp"])


def test_bands_are_nested(report):
    bands = report["closeness_bands"]
    counts = [bands[f"within_{int(t * 100)}pct"]["n_pairs"] for t in sd.CLOSENESS_BANDS]
    assert counts == sorted(counts), "wider closeness bands must contain at least as many pairs"


def test_check_mode_passes_against_the_committed_artifact(report):
    from _artifact_check import _diff
    assert _diff(report, sd.build_analysis()) == []

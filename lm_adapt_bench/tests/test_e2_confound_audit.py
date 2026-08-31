"""Guards on tools/audit_e2_confounds.py.

The audit exists to stop E2's headline contrast being read as settled, so its statistics have to
be right in the direction that matters: the paired test must not overstate significance, and the
leverage check must actually drop the model that dominates the fit.
"""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

au = pytest.importorskip("audit_e2_confounds")


def _cells(math_deltas, news_deltas):
    return {
        "math__gsm8k": {"per_model": {m: {"delta_pp": d} for m, d in math_deltas.items()}},
        "news__gsm8k": {"per_model": {m: {"delta_pp": d} for m, d in news_deltas.items()}},
    }


def test_slug_matches_the_pipeline_slugify():
    """The audit reads results/domain_transfer/<corpus>__<slug>.json written by the pipeline."""
    utils = pytest.importorskip("lm_adapt_bench.utils")
    for model_id in ("Qwen/Qwen2.5-7B", "LiquidAI/LFM2.5-1.2B-Base", "meta-llama/Llama-3.2-1B"):
        assert au._slug(model_id) == utils.slugify(model_id)


def test_paired_contrast_is_the_per_model_difference():
    cells = _cells({"a": -3.0, "b": -1.0}, {"a": -1.0, "b": -1.0})
    out = au.paired_contrast(cells, ["a", "b"])
    assert out["per_model_pp"] == {"a": -2.0, "b": 0.0}
    assert out["mean_pp"] == pytest.approx(-1.0)
    assert out["n_negative"] == 1
    assert out["df"] == 1


def test_required_cohort_grows_as_the_effect_shrinks():
    """A smaller mean at the same spread must demand more models, not fewer."""
    wide = au.paired_contrast(_cells({"a": -3.0, "b": 1.0}, {"a": 0.0, "b": 0.0}), ["a", "b"])
    small = au.paired_contrast(_cells({"a": -1.5, "b": 2.5}, {"a": 0.0, "b": 0.0}), ["a", "b"])
    assert small["n_models_for_80pct_power"] > wide["n_models_for_80pct_power"]
    # sd is identical by construction; only the mean moved.
    assert small["sd_pp"] == pytest.approx(wide["sd_pp"])


def test_power_formula_matches_the_closed_form():
    out = au.paired_contrast(_cells({"a": -3.0, "b": 1.0}, {"a": 0.0, "b": 0.0}), ["a", "b"])
    expected = au.Z_SUM_SQ * (out["sd_pp"] / abs(out["mean_pp"])) ** 2
    assert out["n_models_for_80pct_power"] == pytest.approx(expected)
    assert math.isclose(au.Z_SUM_SQ, 7.848, rel_tol=1e-3)


def test_correlation_helpers_agree_on_a_monotone_nonlinear_series():
    xs = [1.0, 2.0, 3.0, 10.0]
    ys = [1.0, 4.0, 9.0, 100.0]
    assert au._spearman(xs, ys) == pytest.approx(1.0)   # rank-perfect
    assert au._pearson(xs, ys) < 1.0                    # but not linear


def test_check_tolerates_last_bit_float_drift_but_not_real_drift():
    """The committed-artefact gate must not fail on which interpreter ran it.

    Measured: the same correlation came out as -0.9345753609945784 on one CPython and ...83 on
    another, so a byte-exact comparison reported a stale artefact spuriously.
    """
    committed = {"a": {"pearson": -0.9345753609945784}, "n": 10, "who": "LFM2.5-1.2B"}
    fresh_ok = {"a": {"pearson": -0.9345753609945783}, "n": 10, "who": "LFM2.5-1.2B"}
    assert au._diff(committed, fresh_ok) == []

    fresh_bad = {"a": {"pearson": -0.93}, "n": 10, "who": "LFM2.5-1.2B"}
    assert au._diff(committed, fresh_bad)          # a real change is still caught
    assert au._diff(committed, {"a": {"pearson": -0.9345753609945784}, "n": 11,
                                "who": "LFM2.5-1.2B"})
    assert au._diff(committed, {"a": {"pearson": -0.9345753609945784}, "n": 10, "who": "other"})


def test_diff_reports_missing_and_extra_keys():
    assert au._diff({"a": 1}, {}) and au._diff({}, {"a": 1})
    assert au._diff({"a": [1, 2]}, {"a": [1]})


def test_counts_and_types_compare_exactly():
    """The float tolerance must not leak onto integers or hide a type change."""
    assert au._diff({"n": 1_000_000_000}, {"n": 1_000_000_001})   # counts are exact
    assert au._diff({"n": 1}, {"n": 1.0})                          # int vs float is a real change
    assert au._diff({"x": 1.0}, {"x": 1.0000000005}) == []         # float drift is tolerated
    assert au._diff({"x": float("inf")}, {"x": float("inf")}) == []
    assert au._diff({"x": float("inf")}, {"x": 1.0})


def test_selection_accuracy_survives_model_level_resampling():
    """The claim the leaderboard makes must hold at the unit the design randomises over."""
    ps = pytest.importorskip("pairwise_significance")
    report = ps.build(reps=400)                      # small but seeded; the count is stable
    sel = report["selection_accuracy"]
    assert len(sel) == 24
    assert report["cells_beating_chance"] >= 22
    assert all(0.0 <= v["ci_lo"] <= v["accuracy"] <= v["ci_hi"] <= 1.0 for v in sel.values())


def test_pairwise_accuracy_is_computed_over_comparable_pairs_only():
    """Ties carry no ordering information and must not be scored either way."""
    ps = pytest.importorskip("pairwise_significance")
    bpb = {("c", "t", "a"): 0.1, ("c", "t", "b"): 0.2, ("c", "t", "d"): 0.2}
    acc = {"a": {"x": 0.9}, "b": {"x": 0.8}, "d": {"x": 0.8}}
    # a<b and a<d are comparable and correct; b vs d ties on BOTH axes and is skipped.
    assert ps.pairwise_accuracy(bpb, acc, ["a", "b", "d"], "c", "t", "x") == pytest.approx(1.0)
    # A repeated model from a bootstrap draw must not compare against itself.
    assert ps.pairwise_accuracy(bpb, acc, ["a", "a"], "c", "t", "x") is None


def test_the_unresolved_contrasts_are_reported_as_unresolved():
    """The paper's honesty depends on these intervals containing zero; pin that they are computed."""
    ps = pytest.importorskip("pairwise_significance")
    report = ps.build(reps=400)
    for r in report["corpus_steering_contrast"].values():
        assert r["ci_lo"] <= r["estimate"] <= r["ci_hi"]
        assert "excludes_zero" in r

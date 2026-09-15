"""Guards on tools/analyze_tier_mechanism.py.

This artifact carries the paper's explanation for demoting the free tier, and an explanation is
exactly the kind of claim that drifts: it is a story about eleven models, and a story survives edits
that a number would not. The tests pin the two things the prose asserts -- that the models zero-shot
under-rates are the models needing the most adaptation, and that the pattern does not rest on the
single most vivid example -- and pin the hedge that keeps it honest.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_tier_mechanism as tm  # noqa: E402

ARTIFACT = ROOT / "results" / "tier_mechanism.json"


@pytest.fixture(scope="module")
def report():
    if not ARTIFACT.exists():
        pytest.skip("tier mechanism artifact not built")
    return json.loads(ARTIFACT.read_text())


def test_the_underrated_are_the_ones_that_needed_adapting(report):
    """The whole mechanism claim, as set equality so no reordering can weaken it."""
    assert set(report["most_underrated_by_zero_shot"]) == set(report["largest_reductions"])


def test_adaptation_removes_the_relationship_it_is_supposed_to(report):
    """If adaptation did not remove the convention mismatch, both correlations would be similar."""
    pr = report["pearson_reduction_vs_rank_error"]
    assert pr["zero_shot"] > 0.5, pr
    assert abs(pr["adapted"]) < 0.2, pr
    err = report["mean_abs_rank_error"]
    assert err["adapted"] < err["zero_shot"] / 2, err


def test_rank_error_sign_convention_is_what_the_prose_says(report):
    """Positive rank error must mean the tier ranks the model WORSE than the criterion does."""
    per = report["per_model"]
    lfm = per["LFM2.5-1.2B"]
    assert lfm["rank_zero_shot"] > lfm["rank_cloze"]
    assert lfm["rank_error_zero_shot"] == lfm["rank_zero_shot"] - lfm["rank_cloze"] > 0


def test_the_named_example_is_as_described(report):
    """LFM last zero-shot, ahead of Qwen2.5-0.5B once adapted, and genuinely ahead on the
    criterion -- all three are needed for the sentence to be true."""
    per = report["per_model"]
    lfm, q05 = per["LFM2.5-1.2B"], per["Qwen2.5-0.5B"]
    assert lfm["rank_zero_shot"] == len(per)
    assert lfm["zero_shot_bpb"] > q05["zero_shot_bpb"]
    assert lfm["adapted_bpb"] < q05["adapted_bpb"]
    assert lfm["cloze"] > q05["cloze"], "the criterion must actually agree with the adapted tier"


def test_the_pattern_does_not_rest_on_the_vivid_example(report):
    """Gemma, not LFM, carries the numbers -- the prose says so and must stay true."""
    flips = report["pairs_zero_shot_gets_wrong_that_adapted_gets_right"]
    assert len(flips) == report["n_such_pairs"] >= 5
    lfm_pairs = [p for p in flips if "LFM2.5-1.2B" in p]
    gemma_pairs = [p for p in flips if any("gemma" in m for m in p)]
    assert len(lfm_pairs) < len(gemma_pairs), (len(lfm_pairs), len(gemma_pairs))


def test_the_hedge_survives(report):
    """Eleven models cannot support an estimated effect, and the artifact has to keep saying so."""
    assert "not an estimate" in report["caveat"]
    assert "eleven models" in report["caveat"]


def test_ranks_are_a_permutation(report):
    per = report["per_model"]
    n = len(per)
    for key in ("rank_zero_shot", "rank_adapted", "rank_cloze"):
        assert sorted(p[key] for p in per.values()) == list(range(1, n + 1)), key

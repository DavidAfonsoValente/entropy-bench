"""Guards on tools/analyze_cloze_steering.py.

This tool exists to test the steering claim with no public benchmark anywhere in it. Two ways it
could quietly stop doing that, and neither would show up as a crash:

1. **Reporting a main effect as an interaction.** If news-adapted BPB were simply the better
   selector everywhere, the difference-in-differences would still come out positive off one arm
   alone. Steering predicts BOTH arms positive -- each selector doing relatively better on its own
   corpus's criterion -- so the arms are checked separately from the DiD.
2. **Mixing the two criteria.** The two runs are written under different prefixes precisely so a
   glob cannot pick up the wrong one; if that broke, both "criteria" would be the same numbers and
   every difference would collapse to zero.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

# NOT importorskip: these modules are in this repository, so a failure to import them is a bug to
# surface, not an environment to skip. Only the ARTIFACT may legitimately be absent.
import analyze_cloze_steering as cs  # noqa: E402
import analyze_cloze_validity as cv  # noqa: E402

ARTIFACT = ROOT / "results" / "cloze_steering.json"
needs_artifact = pytest.mark.skipif(not ARTIFACT.exists(),
                                    reason="mathematics cloze run not present")


@pytest.fixture(scope="module")
def report():
    if not ARTIFACT.exists():
        pytest.skip("mathematics cloze run not present")
    return json.loads(ARTIFACT.read_text())


def test_did_is_the_interaction_not_a_main_effect():
    """Hand-built cells: the DiD must be computed as (a-c)-(b-d) and nothing else."""
    vals = {"news_adapted_bpb__vs__news_cloze": 0.9,
            "math_adapted_bpb__vs__news_cloze": 0.6,
            "news_adapted_bpb__vs__math_cloze": 0.5,
            "math_adapted_bpb__vs__math_cloze": 0.8}
    assert cs._did(vals) == pytest.approx((0.9 - 0.6) - (0.5 - 0.8))


def test_a_constant_advantage_cancels_in_the_did():
    """The interaction is the right statistic precisely because a uniform main effect vanishes."""
    constant = {"news_adapted_bpb__vs__news_cloze": 0.9,
                "math_adapted_bpb__vs__news_cloze": 0.6,
                "news_adapted_bpb__vs__math_cloze": 0.8,
                "math_adapted_bpb__vs__math_cloze": 0.5}
    assert cs._did(constant) == pytest.approx(0.0)


def test_did_is_positive_even_when_only_one_arm_moves():
    """Why both_arms_positive_point_estimate exists as a separate field.

    Not a *constant* main effect -- that cancels, as the test above shows. This is the narrower
    and real hazard: a selector that is better on one criterion and merely tied on the other
    produces a positive interaction while only one matched inequality actually holds.
    """
    one_arm = {"news_adapted_bpb__vs__news_cloze": 0.9,
               "math_adapted_bpb__vs__news_cloze": 0.5,
               "news_adapted_bpb__vs__math_cloze": 0.9,
               "math_adapted_bpb__vs__math_cloze": 0.9}
    assert cs._did(one_arm) > 0, "a positive DiD alone cannot establish steering"


def test_did_is_none_when_any_cell_is_undefined():
    vals = {"news_adapted_bpb__vs__news_cloze": None,
            "math_adapted_bpb__vs__news_cloze": 0.6,
            "news_adapted_bpb__vs__math_cloze": 0.5,
            "math_adapted_bpb__vs__math_cloze": 0.8}
    assert cs._did(vals) is None


@needs_artifact
def test_the_two_criteria_are_distinct_runs():
    """Checked on the INPUTS, not on an accuracy inequality.

    Two aggregate accuracies can differ while the inputs are misrouted, and can coincide while
    they are fine, so comparing them proves nothing either way. The result files record which
    cloze set produced them; that is the thing to check.
    """
    news = json.loads(next((ROOT / "results" / "cloze").glob("cloze__*.json")).read_text())
    math = json.loads(next((ROOT / "results" / "cloze_math").glob("cloze_math__*.json")).read_text())
    assert news["cloze_set"] != math["cloze_set"], (news["cloze_set"], math["cloze_set"])
    assert news["cloze_n_items"] != math["cloze_n_items"]


def test_all_four_cells_share_one_pair_list(report):
    """The composition confound: subtracting cell accuracies taken over different dyads.

    A tie in one criterion drops a pair from two cells and not the others. Sharing bootstrap draws
    does not repair that -- the cells must be built on a common eligible-pair list, and this was a
    real defect in the first version of this analysis (55/54/55/54).
    """
    for key in ("lenient", "strict"):
        for scope, blk in report[key].items():
            counts = {c["n_pairs"] for c in blk["cells"].values()}
            assert len(counts) == 1, (key, scope, counts)


def test_common_mask_actually_drops_the_tied_pair():
    """Exercised directly, not only through the shipped artifact."""
    models = ["a", "b", "c"]
    params = {m: 1.0e9 for m in models}
    bpb = {c: {"adapted_bpb": {"a": 0.1, "b": 0.2, "c": 0.3}} for c in cs.CRITERIA}
    cloze = {"news": {"m": {"a": 0.1, "b": 0.2, "c": 0.3}},
             "math": {"m": {"a": 0.1, "b": 0.1, "c": 0.3}}}   # a and b tie on the maths criterion
    pairs = cs._eligible_pairs(bpb, cloze, models, params, "m", None)
    assert ("a", "b") not in pairs and len(pairs) == 2


def test_selectors_are_fixed_by_design(report):
    assert cs.SELECTORS == ("news_adapted_bpb", "math_adapted_bpb")
    assert cs.CRITERIA == ("news", "math")
    assert report["primary_estimand"].startswith("difference_in_differences over all pairs")


def test_every_cell_is_scored_on_the_same_resample():
    """The shared stream is the whole basis for reading the interaction directly.

    Verified by construction rather than by inspection: two selectors that are IDENTICAL must give
    exactly zero on every bootstrap draw, which is only true if both saw the same cohort.
    """
    models = ["a", "b", "c", "d"]
    params = {m: 1.0e9 for m in models}
    same = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    bpb = {c: {"adapted_bpb": dict(same)} for c in cs.CRITERIA}
    cloze = {"news": {"m": {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}},
             "math": {"m": {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}}}
    out = cs.analyse(bpb, cloze, models, params, "m")
    for arm in ("news_arm", "math_arm"):
        assert out["arms"][arm]["observed"] == 0.0
        assert out["arms"][arm]["ci_lo"] == 0.0 and out["arms"][arm]["ci_hi"] == 0.0
    assert out["difference_in_differences"]["observed"] == 0.0


def test_both_conventions_and_every_band_are_reported(report):
    for key in ("lenient", "strict"):
        assert "all_pairs" in report[key]
        for band in cv.SIZE_BANDS:
            assert f"within_{band:g}x" in report[key]
            blk = report[key][f"within_{band:g}x"]
            assert set(blk["cells"]) == set(report[key]["all_pairs"]["cells"])


def test_a_negative_interval_is_never_reported_as_unresolved(report):
    """The bug an independent review found: one flag merged three different outcomes.

    `lo > 0` is false for an interval containing zero, for one lying entirely BELOW zero, and for
    no interval at all. "Does not separate from zero" was then used to mean "no effect either way",
    which a strongly negative result would also have satisfied.
    """
    for key in ("lenient", "strict"):
        for scope, blk in report[key].items():
            for d in [blk["difference_in_differences"]] + list(blk["arms"].values()):
                assert {"ci_excludes_zero", "supports_positive_effect",
                        "supports_negative_effect", "ci_contains_zero",
                        "inference_available"} <= set(d), (key, scope)
                if d["inference_available"]:
                    assert d["ci_excludes_zero"] == (d["supports_positive_effect"]
                                                     or d["supports_negative_effect"])
                    assert d["ci_contains_zero"] != d["ci_excludes_zero"]
                    # Nothing in this study may be negative-resolved without the gate noticing.
                    assert d["supports_negative_effect"] is False, (key, scope)


def test_verdict_field_is_named_for_what_it_measures(report):
    """It reads point-estimate signs only, and its name has to say so."""
    assert "sign_pattern_consistent_with_steering" in report
    assert "steering_holds_under_both_conventions" not in report
    for key in ("lenient", "strict"):
        for blk in report[key].values():
            assert "both_arms_positive_point_estimate" in blk
            assert blk["both_arms_positive_point_estimate"] == all(
                a["observed"] > 0 for a in blk["arms"].values())


def test_the_degeneracy_guard_is_wired(report):
    for key in ("lenient", "strict"):
        for scope, blk in report[key].items():
            if blk["degenerate_draw_frac"] > cv.MAX_DEGENERATE_FRAC:
                assert blk["inference_available"] is False, (key, scope)
                for d in [blk["difference_in_differences"]] + list(blk["arms"].values()):
                    assert d["ci_lo"] is None and d["inference_available"] is False, (key, scope)
                    assert d["ci_excludes_zero"] is False, (key, scope)


def test_the_sixteen_contrasts_are_not_treated_as_sixteen_tests(report):
    """No combined significance may be derivable from the artifact.

    The two conventions score the same generations and the bands are nested, so the sixteen arm
    contrasts are one observation under sixteen specifications. The artifact must therefore carry
    a single declared primary estimand and no count-based verdict field.
    """
    assert report["primary_estimand"]
    flat = json.dumps(report)
    for forbidden in ("sign_test", "combined_p", "n_confirmations", "fisher"):
        assert forbidden not in flat


def test_max_per_doc_defaults_to_independent_items():
    import build_domain_cloze as b
    docs = ["x" * 600 + " The Riemann Hypothesis holds for 42 cases here. " + "y" * 200] * 4
    one = b.build_items(docs, 100, 42, b.PATTERNS)
    many = b.build_items(docs, 100, 42, b.PATTERNS, max_per_doc=3)
    assert len(one) <= len(docs), "the default must take at most one item per document"
    assert len(many) >= len(one)
    with pytest.raises(ValueError):
        b.build_items(docs, 10, 42, b.PATTERNS, max_per_doc=0)

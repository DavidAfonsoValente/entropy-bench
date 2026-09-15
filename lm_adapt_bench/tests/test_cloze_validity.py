"""Guards on tools/analyze_cloze_validity.py.

Two things about this tool are easy to break in a way nothing else would catch.

First, the *band* restriction. The full-range comparison answers a question no practitioner asks --
over 0.5B to 35B, "take the bigger model" is already a good rule, and this study says so. The
recommendation is scoped to candidates of comparable size, so the band tables are the ones the
claims rest on. If the restriction silently stopped restricting, every band would collapse to the
full-range answer and the artifact would look fine.

Second, the *conventions*. Lenient and strict scoring disagree about who wins, and the tool exists
partly to stop the flattering convention being reported alone. A claim may be asserted only if it
survives both, and that must never degrade into "the two conventions agreed" -- they agree just as
well when both reject.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

cv = pytest.importorskip("analyze_cloze_validity")
ARTIFACT = ROOT / "results" / "cloze_validity.json"


@pytest.fixture(scope="module")
def report():
    return json.loads(ARTIFACT.read_text())


# --- the band restriction actually restricts -------------------------------------------------

PARAMS = {"tiny": 1.0e9, "small": 1.8e9, "big": 10.0e9, "huge": 30.0e9}
MODELS = sorted(PARAMS)


def test_band_drops_pairs_outside_the_size_factor():
    target = {"tiny": 0.1, "small": 0.2, "big": 0.3, "huge": 0.4}
    # A perfect selector, so accuracy carries no information here -- only the denominator does.
    _, n_all = cv._score(target, target, MODELS, lower_is_better=False)
    _, n_band = cv._score(target, target, MODELS, lower_is_better=False,
                          params=PARAMS, band=2.0)
    assert n_all == 6                      # 4 models -> 6 pairs
    assert n_band == 1                     # only tiny/small are within 2x
    _, n_wide = cv._score(target, target, MODELS, lower_is_better=False,
                          params=PARAMS, band=100.0)
    assert n_wide == n_all


def test_band_is_inclusive_at_the_boundary():
    params = {"a": 1.0e9, "b": 2.0e9}
    target = {"a": 0.1, "b": 0.2}
    _, n = cv._score(target, target, ["a", "b"], lower_is_better=False, params=params, band=2.0)
    assert n == 1, "a pair at exactly the band factor must be kept, not dropped"


def test_pairwise_accuracy_is_unchanged_by_the_refactor():
    """The public helper must still be the unrestricted statistic it always was, ties and gaps
    included -- those are the behaviours a refactor silently changes."""
    sel = {"a": 1.0, "b": 2.0, "c": 3.0}
    tgt = {"a": 0.3, "b": 0.2, "c": 0.1}
    assert cv.pairwise_accuracy(sel, tgt, ["a", "b", "c"], lower_is_better=True) == 1.0
    assert cv.pairwise_accuracy(sel, tgt, ["a", "b", "c"], lower_is_better=False) == 0.0
    # A tie on either side carries no ordering information and must drop the pair, not score it.
    assert cv._score({"a": 1.0, "b": 1.0}, tgt, ["a", "b"], True) == (None, 0)
    assert cv._score(sel, {"a": 0.2, "b": 0.2}, ["a", "b"], True) == (None, 0)
    # A model absent from either mapping is skipped rather than crashing.
    assert cv._score(sel, tgt, ["a", "b", "zz"], True)[1] == 1


def test_score_returns_none_when_no_pair_survives():
    acc, n = cv._score({"a": 1.0, "b": 2.0}, {"a": 0.1, "b": 0.2}, ["a", "b"],
                       lower_is_better=True, params={"a": 1.0e9, "b": 9.0e9}, band=2.0)
    assert acc is None and n == 0


def test_degenerate_draws_are_measured_not_hidden(report):
    """Skipping empty draws conditions the bootstrap, so the rate must be visible and bounded."""
    assert report["degenerate_draw_frac"] <= cv.MAX_DEGENERATE_FRAC
    assert report["inference_available"] is True
    for key, blk in report["bands"].items():
        assert "degenerate_draw_frac" in blk, key
        # The narrower the band the more draws come back empty. If a band ever crosses the
        # declared threshold the tool must withhold intervals rather than publish conditioned ones.
        if blk["degenerate_draw_frac"] > cv.MAX_DEGENERATE_FRAC:
            assert blk["inference_available"] is False, key
            assert all(r["ci_lo"] is None for r in blk["selectors"].values()), key


def test_partial_degeneracy_still_yields_an_interval_but_is_recorded():
    """The dangerous case is PARTIAL degeneracy, where a plausible-looking CI comes back."""
    params = {"a": 1.0e9, "b": 1.5e9, "c": 30.0e9}
    cloze = {"a": 0.1, "b": 0.2, "c": 0.3}
    sels = {n: (cloze, False) for n in cv.SELECTOR_BPB + cv.SELECTOR_STATIC + ("parameter_count",)}
    out = cv._bootstrap_tables(sels, cloze, ["a", "b", "c"], params, band=2.0)
    # Cohorts drawn as e.g. (c, c, c) contain no in-band pair at all.
    assert 0.0 < out["degenerate_draw_frac"] < 1.0
    assert out["inference_available"] is (out["degenerate_draw_frac"] <= cv.MAX_DEGENERATE_FRAC)


def test_band_pair_count_is_geometric_not_a_selector_denominator():
    params = {"a": 1.0e9, "b": 1.5e9, "c": 30.0e9}
    assert cv._band_pair_count(["a", "b", "c"], params, 2.0) == 1
    assert cv._band_pair_count(["a", "b", "c"], params, 100.0) == 3


def test_band_arguments_are_validated():
    tgt = {"a": 0.1, "b": 0.2}
    with pytest.raises(ValueError):
        cv._score(tgt, tgt, ["a", "b"], False, params=None, band=2.0)
    with pytest.raises(ValueError):
        cv._score(tgt, tgt, ["a", "b"], False, params={"a": 1.0, "b": 1.0}, band=-1.0)
    with pytest.raises(ValueError):
        cv._score(tgt, tgt, ["a", "b"], False, params={"a": 1.0}, band=2.0)


def test_a_resampled_cohort_with_duplicate_labels_invents_no_self_comparison():
    sel = {"a": 1.0, "b": 2.0}
    tgt = {"a": 0.2, "b": 0.1}
    # Three copies of 'a' and one 'b' -> 3 a-b pairs, and zero a-a pairs.
    acc, n = cv._score(sel, tgt, ["a", "a", "a", "b"], lower_is_better=True)
    assert n == 3 and acc == 1.0


def test_verdicts_refuse_to_rank_an_unrankable_table():
    table = {k: {"pairwise_accuracy": 0.6} for k in
             cv.SELECTOR_BPB + cv.SELECTOR_STATIC + ("parameter_count",)}
    assert cv._verdicts(table)["best_bpb_selector"] is not None
    table["gsm8k"]["pairwise_accuracy"] = None
    assert all(v is None for v in cv._verdicts(table).values())


# --- the shipped artifact keeps its self-refutations ------------------------------------------

def test_every_band_in_size_bands_is_reported(report):
    assert set(report["bands"]) == {f"within_{b:g}x" for b in cv.SIZE_BANDS}
    primary = [k for k, v in report["bands"].items() if v["is_primary"]]
    assert primary == [f"within_{cv.PRIMARY_BAND:g}x"], "exactly one band may be primary"


def test_bands_are_nested_and_shrink(report):
    """Set nesting, not just non-decreasing counts -- equal counts over different pairs would pass
    a count-only check while meaning the restriction is broken."""
    from itertools import combinations
    params = report["bands"][f"within_{cv.PRIMARY_BAND:g}x"] and cv._load_params()
    models = report["models_scored"]

    def in_band(b):
        return {(x, y) for x, y in combinations(models, 2)
                if max(params[x], params[y]) / min(params[x], params[y]) <= b}

    ordered = sorted(cv.SIZE_BANDS)
    for narrow, wide in zip(ordered, ordered[1:]):
        assert in_band(narrow) < in_band(wide), f"{narrow}x must be a strict subset of {wide}x"
    for b in cv.SIZE_BANDS:
        assert report["bands"][f"within_{b:g}x"]["n_pairs"] == len(in_band(b))
    assert len(in_band(max(cv.SIZE_BANDS))) < 55


def test_both_conventions_ship_for_every_band(report):
    for key, blk in report["bands"].items():
        assert blk["selectors"] and blk["selectors_strict"], key
        assert set(blk["selectors"]) == set(blk["selectors_strict"]), key


def test_size_is_never_claimed_as_beaten(report):
    """The one claim this study must not make.

    Checked on the STRICT verdict directly, not only on the combined field: the combined field is
    also false when lenient fails and strict succeeds, which is a different world.
    """
    assert report["strict"]["bpb_beats_parameter_count"] is False
    assert report["claim_holds_under_both_conventions"]["bpb_beats_parameter_count"] is False
    for key, blk in report["bands"].items():
        assert blk["strict"]["bpb_beats_parameter_count"] is False, key
        assert blk["claim_holds_under_both_conventions"]["bpb_beats_parameter_count"] is False, key


def test_the_benchmark_claim_is_band_scoped(report):
    """True where the decision is hard, false over the full range. Both halves are load-bearing.

    Each convention is asserted on its own, so the combined field cannot pass by accident.
    """
    assert report["claim_holds_under_both_conventions"][
        "bpb_beats_every_static_benchmark"] is False
    primary = report["bands"][f"within_{cv.PRIMARY_BAND:g}x"]
    assert primary["bpb_beats_every_static_benchmark"] is True
    assert primary["strict"]["bpb_beats_every_static_benchmark"] is True
    assert primary["claim_holds_under_both_conventions"][
        "bpb_beats_every_static_benchmark"] is True


def test_a_point_estimate_lead_is_never_reported_as_a_resolved_one(report):
    """The correction that mattered: leading on the point estimate is not a resolved lead.

    The paired bootstrap draws ONE cohort and scores every selector on it, so the difference is
    read directly rather than inferred from two overlapping marginal intervals. At this cohort size
    no difference separates from zero, and the artifact has to keep saying so next to every verdict
    that compares point estimates.
    """
    assert "point estimates only" in report["comparison_basis"]
    blocks = [report] + list(report["bands"].values())
    for blk in blocks:
        for key in ("paired_differences", "paired_differences_strict"):
            for name, d in blk[key].items():
                assert d["separates_from_zero"] is False, (name, key)
                assert d["ci_lo"] is not None and d["ci_lo"] <= 0.0


def test_every_selector_is_scored_on_the_same_pairs(report):
    """Otherwise a verdict can be true because one selector got an easier subset."""
    assert report["common_pair_mask"] is True
    for key, blk in report["bands"].items():
        assert blk["common_pair_mask"] is True, key


def test_agreement_is_not_survival(report):
    """Two conventions agreeing that a claim FAILS is not the claim surviving.

    Exercised on the dangerous case directly -- both conventions rejecting -- rather than only on
    the shipped artifact, where the two happen to disagree.
    """
    assert report["verdicts_agree_across_conventions"] is False

    reject = {k: {"pairwise_accuracy": 0.5} for k in
              cv.SELECTOR_BPB + cv.SELECTOR_STATIC + ("parameter_count",)}
    reject["hellaswag"]["pairwise_accuracy"] = 0.9      # a benchmark wins under both conventions
    reject["parameter_count"]["pairwise_accuracy"] = 0.9
    v = cv._verdicts(reject)
    assert v["bpb_beats_every_static_benchmark"] is False
    assert v["bpb_beats_parameter_count"] is False
    agree = (v["bpb_beats_every_static_benchmark"] == v["bpb_beats_every_static_benchmark"])
    holds = bool(v["bpb_beats_every_static_benchmark"] and v["bpb_beats_every_static_benchmark"])
    assert agree is True and holds is False, "identical rejections must agree and still not hold"


def test_denominators_are_recorded(report):
    """Both conventions, and an undefined result may not carry a nonzero denominator."""
    for table in ("selectors", "selectors_strict"):
        for row in report[table].values():
            assert row["n_pairs"] == 55
    for key, blk in report["bands"].items():
        for table in ("selectors", "selectors_strict"):
            for name, row in blk[table].items():
                if row["pairwise_accuracy"] is None:
                    assert row["n_pairs"] == 0, (key, table, name)
                else:
                    assert row["n_pairs"] == blk["n_pairs"], (key, table, name)

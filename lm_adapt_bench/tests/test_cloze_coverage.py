"""Guards on tools/analyze_cloze_coverage.py.

This tool exists to stop one corpus's result being sold as the method's result. Its job is to run
the identical comparison per corpus and then NOT combine the answers -- four criteria of different
sizes and span compositions, scored by the same eleven models, are four views of one cohort. The
failure this guards against is a pooled number appearing later because it looked tidier.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_cloze_coverage as cc  # noqa: E402
import analyze_cloze_validity as cv  # noqa: E402

ARTIFACT = ROOT / "results" / "cloze_coverage.json"


@pytest.fixture(scope="module")
def report():
    if not ARTIFACT.exists():
        pytest.skip("cloze coverage artifact not built")
    return json.loads(ARTIFACT.read_text())


def test_every_corpus_uses_its_own_matched_tier(report):
    """The comparand must be the corpus's OWN adapted tier, never a globally chosen winner."""
    for corpus, e in report["per_corpus"].items():
        assert e["comparand"] == "matched_adapted_bpb", corpus


def test_nothing_is_pooled_across_corpora(report):
    """Lists of corpus names, plus two explicitly descriptive inventories. No pooled statistic.

    `pairs_by_corpus` and `size_contrast_signs` are dicts by design: the first records per-corpus
    denominators, the second is a plain count of signs that the prose quotes. Neither is a
    statistic over corpora, and everything else must remain a list of names.
    """
    summary = report["summary"]
    descriptive = {"pairs_by_corpus", "size_contrast_signs"}
    for key, value in summary.items():
        if key in descriptive:
            assert isinstance(value, dict), key
            continue
        assert isinstance(value, list), f"{key} is not a list of corpus names"
    assert "NOT pooled across corpora" in report["method"]
    flat = json.dumps(report)
    for forbidden in ("pooled_p", "combined_p", "mean_across_corpora", "fisher"):
        assert forbidden not in flat


def test_the_news_result_matches_the_authoritative_artifact(report):
    """Coverage must reproduce analyze_cloze_validity's news numbers, not approximate them.

    Different selector NAMES (matched_adapted_bpb vs news_adapted_bpb) but the same underlying
    series, the same seed and the same shared-stream construction, so the accuracies must agree
    exactly. If they ever diverge, one of the two tools has drifted.
    """
    validity = json.loads((ROOT / "results" / "cloze_validity.json").read_text())
    scope = f"within_{cv.PRIMARY_BAND:g}x"
    v = validity["bands"][scope]["selectors"]
    c = report["per_corpus"]["news"]["lenient"][scope]["selectors"]
    assert c["matched_adapted_bpb"]["pairwise_accuracy"] == \
        v["news_adapted_bpb"]["pairwise_accuracy"]
    assert c["zero_shot_bpb"]["pairwise_accuracy"] == v["zero_shot_news_bpb"]["pairwise_accuracy"]
    for b in ("gsm8k", "mmlu_pro", "hellaswag", "parameter_count"):
        assert c[b]["pairwise_accuracy"] == v[b]["pairwise_accuracy"], b


def test_item_counts_are_recorded_per_corpus_because_they_differ(report):
    """The four criteria are not interchangeable measurements and the artifact must show it."""
    counts = {c: e["n_items"] for c, e in report["per_corpus"].items()}
    assert len(set(counts.values())) > 1, "identical item counts would mean a mis-wired glob"
    for corpus, e in report["per_corpus"].items():
        assert sum(e["items_by_kind"].values()) == e["n_items"], corpus


def test_both_conventions_and_both_scopes_are_reported(report):
    for corpus, e in report["per_corpus"].items():
        for key in ("lenient", "strict"):
            assert set(e[key]) == {"all_pairs", f"within_{cv.PRIMARY_BAND:g}x"}, corpus
            for scope, blk in e[key].items():
                assert blk["common_pair_mask"] is True, (corpus, key, scope)


def test_the_size_comparison_is_unresolved_not_won(report):
    """The claim about parameter count, stated as the data supports it.

    An earlier version of this test asserted that the point estimate never favours us. That was an
    assumption written before the mathematics and Reddit criteria existed, and it is false: on
    those two corpora the matched tier leads size under both conventions. What is actually true --
    and what the paper may say -- is that NO paired difference against size separates from zero on
    any corpus under either convention, and the point-estimate signs are mixed.
    """
    signs = []
    for corpus, e in report["per_corpus"].items():
        for key in ("lenient", "strict"):
            for scope, blk in e[key].items():
                d = blk["paired_differences"]["bpb_minus_parameter_count"]
                assert d["separates_from_zero"] is False, (corpus, key, scope)
                if scope.startswith("within_"):
                    signs.append(d["observed"])
    assert any(v > 0 for v in signs) and any(v < 0 for v in signs), \
        "mixed signs are the honest finding; a uniform sign would need different prose"


def test_no_public_benchmark_clears_chance_on_any_corpus(report):
    """A FAILURE-TO-REJECT, and the tests must not let it be read as more.

    At ten or eleven in-band pairs almost nothing is resolvable: exactly one cell in the whole
    comparison has an interval excluding chance. So this says only that no benchmark's interval
    excluded chance -- not that the benchmarks are uninformative. Checked under BOTH conventions
    and per selector, because an empty summary list could otherwise be empty for the wrong reason.
    """
    for key in ("lenient", "strict"):
        assert report["summary"][f"any_benchmark_clears_chance_{key}"] == []
    scope = f"within_{cv.PRIMARY_BAND:g}x"
    for corpus, e in report["per_corpus"].items():
        for key in ("lenient", "strict"):
            for b in cv.SELECTOR_STATIC:
                assert e[key][scope]["selectors"][b]["beats_chance"] is False, (corpus, key, b)


def test_the_free_tier_clears_chance_on_no_corpus(report):
    """The retraction, on all four corpora and BOTH conventions, per selector not just per list."""
    scope = f"within_{cv.PRIMARY_BAND:g}x"
    for key in ("lenient", "strict"):
        assert report["summary"][f"zero_shot_clears_chance_{key}"] == []
        for corpus, e in report["per_corpus"].items():
            assert e[key][scope]["selectors"]["zero_shot_bpb"]["beats_chance"] is False, \
                (corpus, key)


def test_a_perfect_score_is_not_counted_as_clearing_chance(report):
    """The false-precision trap, and it fired on real data.

    A selector that orders every in-band pair correctly makes every bootstrap resample score 1.000,
    so the percentile interval collapses to [1.000, 1.000] and a naive lower-bound test passes on
    what is a point estimate at the boundary over ten pairs. Reddit's matched tier and Hacker
    News's parameter count both hit this. Degenerate intervals must be marked and excluded.
    """
    scope = f"within_{cv.PRIMARY_BAND:g}x"
    seen = False
    for corpus, e in report["per_corpus"].items():
        for key in ("lenient", "strict"):
            blk = e[key][scope]
            for name, v in blk["selectors"].items():
                if v["interval_is_degenerate"]:
                    seen = True
                    assert v["beats_chance"] is False, (corpus, key, name)
                    assert name in blk["degenerate_intervals"]
    assert seen, "the guard is untested if no degenerate interval exists in the data"


def test_exactly_one_cell_in_the_whole_comparison_clears_chance(report):
    """The honest headline, asserted so prose cannot drift from it."""
    assert report["summary"]["matched_clears_chance_lenient"] == ["news"]
    assert report["summary"]["matched_clears_chance_strict"] == []
    assert report["summary"]["parameter_count_clears_chance_either"] == []


def test_the_size_sign_split_is_recorded_exactly(report):
    """'5 of 8' must come from the artifact, not from prose that could drift."""
    signs = report["summary"]["size_contrast_signs"]
    assert signs["positive"] + signs["negative"] + signs["zero"] == 8
    assert signs["positive"] == 5 and signs["negative"] == 3 and signs["zero"] == 0
    assert signs["any_separates_from_zero"] is False


def test_cross_convention_pair_masks_are_recorded(report):
    """Where the conventions score different pair sets, 'holds under both' compares two
    populations rather than one population scored twice. The artifact has to say which."""
    shared = report["summary"]["conventions_share_a_pair_mask"]
    pairs = report["summary"]["pairs_by_corpus"]
    for corpus, counts in pairs.items():
        assert (counts["lenient"] == counts["strict"]) == (corpus in shared), corpus
    assert set(shared) != set(pairs), "this guard is vacuous if every mask agrees"


def test_clears_chance_lists_are_consistent_with_the_flags(report):
    scope = f"within_{cv.PRIMARY_BAND:g}x"
    for corpus, e in report["per_corpus"].items():
        blk = e["lenient"][scope]
        assert blk["matched_clears_chance"] == ("matched_adapted_bpb" in blk["clears_chance"])
        assert blk["any_benchmark_clears_chance"] == any(
            b in blk["clears_chance"] for b in cv.SELECTOR_STATIC)
        assert (corpus in report["summary"]["matched_clears_chance_lenient"]) == \
            blk["matched_clears_chance"]

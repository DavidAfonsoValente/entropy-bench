"""Guards on the steering difference-in-differences and its resampling units.

The DiD is the statistic the paper's steering claim leads with, and until 2026-08-30 it was
hand-carried into the LaTeX and only string-matched by the verification script -- no committed
artifact regenerated it. These tests pin the two properties that make the regenerated version
trustworthy: the statistic is the joint movement it claims to be, and the family-level resampling
actually clusters (rather than quietly behaving like the model-level one it is meant to correct).
"""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

ps = pytest.importorskip("pairwise_significance")


def test_did_is_the_joint_movement_not_a_single_contrast():
    """DiD = (GSM8K - HellaSwag | math) - (GSM8K - HellaSwag | general), by construction."""
    bpb, acc = ps._load_bpb(), __import__("json").loads(ps.STATIC_PATH.read_text())
    models = list(ps.MODEL_IDS)

    def pa(corpus, bench):
        return ps.pairwise_accuracy(bpb, acc, models, corpus, "adapted", bench)

    expected = (pa("math", "gsm8k") - pa("math", "hellaswag")) - (
        pa("news", "gsm8k") - pa("news", "hellaswag"))
    assert ps.steering_did(bpb, acc, models, "news") == pytest.approx(expected)


def test_pooled_did_averages_every_general_corpus():
    bpb, acc = ps._load_bpb(), __import__("json").loads(ps.STATIC_PATH.read_text())
    models = list(ps.MODEL_IDS)
    parts = [ps.steering_did(bpb, acc, models, g) for g in ps.GENERAL_CORPORA]
    assert len(ps.GENERAL_CORPORA) == 3
    assert ps.pooled_did(bpb, acc, models) == pytest.approx(sum(parts) / 3)


def test_family_maps_cover_every_model_exactly_once():
    assert set(ps.PUBLISHER) == set(ps.MODEL_IDS)
    assert set(ps.GENERATION) == set(ps.MODEL_IDS)
    # The finer unit must genuinely split Qwen; if it did not, the two "family" rows would be the
    # same analysis reported twice.
    assert len(set(ps.GENERATION.values())) > len(set(ps.PUBLISHER.values()))
    assert ps.GENERATION["Qwen2.5-7B"] != ps.GENERATION["Qwen3.5-9B"]
    assert ps.PUBLISHER["Qwen2.5-7B"] == ps.PUBLISHER["Qwen3.5-9B"]


def test_sign_split_draws_whole_clusters():
    """A family draw must take every member of the family, or it is not a cluster bootstrap."""
    seen = []
    units = {"a1": "A", "a2": "A", "b1": "B"}
    ps.sign_split(lambda cohort: seen.append(sorted(cohort)) or 1.0, units, 12, random.Random(0))
    for cohort in seen:
        assert cohort.count("a1") == cohort.count("a2"), cohort


def test_sign_split_reports_a_three_way_split_that_sums_to_one():
    out = ps.sign_split(lambda c: len(c) - 3, {"a": "A", "b": "B", "c": "C"}, 200,
                        random.Random(1))
    total = out["frac_positive"] + out["frac_zero"] + out["frac_negative"]
    assert total == pytest.approx(1.0)
    assert out["n_clusters"] == 3


def test_tie_conventions_are_ordered_and_bracket_the_mid_p():
    """Counting ties against must be the most conservative, ignoring them the least."""
    out = ps.sign_split(lambda c: (len(set(c)) - 2), {"a": "A", "b": "B", "c": "C"}, 400,
                        random.Random(2))
    assert (out["p_one_sided_ties_ignored"] <= out["p_one_sided_mid_p"]
            <= out["p_one_sided_ties_against"])


def test_committed_artifact_reports_both_units_and_the_jackknife():
    import json
    payload = json.loads((ROOT / "results" / "pairwise_significance.json").read_text())
    did = payload["steering_difference_in_differences"]
    assert set(did["resampling"]) == {"model", "publisher_family", "publisher_and_generation"}
    assert did["resampling"]["publisher_family"]["n_clusters"] == 5
    assert set(did["leave_one_family_out"]) == set(ps.PUBLISHER.values())
    # The point of the whole addition: the conservative unit must be reported, and it must be
    # weaker than the model-level one. If this ever inverts, the prose is wrong somewhere.
    assert (did["resampling"]["publisher_family"]["frac_positive"]
            < did["resampling"]["model"]["frac_positive"])


def test_did_does_not_share_the_rng_with_the_rest_of_the_build():
    """Its own seeded stream, so upstream draws cannot silently move the published figure."""
    import json
    a = ps.build_steering_did(ps._load_bpb(), json.loads(ps.STATIC_PATH.read_text()), 400)
    b = ps.build_steering_did(ps._load_bpb(), json.loads(ps.STATIC_PATH.read_text()), 400)
    assert a["resampling"]["model"] == b["resampling"]["model"]

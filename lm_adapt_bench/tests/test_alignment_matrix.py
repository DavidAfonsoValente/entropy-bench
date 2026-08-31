"""Guards on tools/analyze_alignment_matrix.py.

The alignment matrix is what the paper's central claim now rests on, so the statistics get
property tests on synthetic input (where the right answer is known by construction) plus a
consistency check against the committed artifacts. Without these, a silent change to the rank or
regret definitions would sail through -- verify_paper_numbers.py compares the paper against these
same functions, so it cannot catch a bug that moves both together.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

am = pytest.importorskip("analyze_alignment_matrix")


def test_perfect_agreement_scores_one():
    """Lower BPB must mean higher accuracy for rho = 1 and zero regret."""
    bpb = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    acc = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}
    out = am.alignment(bpb, acc)
    assert out["spearman"] == pytest.approx(1.0)
    assert out["kendall_tau"] == pytest.approx(1.0)
    assert out["exact_ranks"] == 4
    assert out["max_shift"] == 0
    assert out["pairs_correct"] == out["pairs_comparable"] == 6
    assert out["top1_pick"] == out["top1_oracle"] == "a"
    assert out["top1_regret_pp"] == pytest.approx(0.0)


def test_perfect_disagreement_is_negative_and_costly():
    bpb = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    acc = {"a": 0.6, "b": 0.7, "c": 0.8, "d": 0.9}
    out = am.alignment(bpb, acc)
    assert out["spearman"] == pytest.approx(-1.0)
    assert out["pairs_correct"] == 0
    # Picks the worst model; regret is the full spread, in accuracy points.
    assert out["top1_pick"] == "a" and out["top1_oracle"] == "d"
    assert out["top1_regret_pp"] == pytest.approx(30.0)


def test_regret_is_zero_whenever_the_top_pick_is_the_oracle():
    """Regret must depend only on the top pick, not on the rest of the ordering."""
    acc = {"a": 0.9, "b": 0.5, "c": 0.8, "d": 0.6}
    # 'a' ranks first on BPB but the tail is scrambled relative to accuracy.
    bpb = {"a": 0.1, "b": 0.2, "c": 0.4, "d": 0.3}
    out = am.alignment(bpb, acc)
    assert out["top1_regret_pp"] == pytest.approx(0.0)
    assert out["spearman"] < 1.0, "the tail really is misordered"


def test_bootstrap_is_deterministic_and_brackets_the_point_estimate():
    acc = {m: v for m, v in zip("abcdefghijk", [0.9, 0.85, 0.8, 0.75, 0.7,
                                                0.65, 0.6, 0.55, 0.5, 0.45, 0.4])}
    good = {m: i / 100 for i, m in enumerate(acc)}          # perfectly aligned
    bad = {m: -i / 100 for i, m in enumerate(acc)}          # perfectly anti-aligned
    first = am.bootstrap_delta(bad, good, acc)
    second = am.bootstrap_delta(bad, good, acc)
    assert first == second, "seeded bootstrap must be reproducible"
    assert first["ci_low"] <= first["delta_spearman"] <= first["ci_high"]
    assert first["delta_spearman"] == pytest.approx(2.0)
    assert first["excludes_zero"] is True
    assert first["resamples_used"] > 0.9 * am.BOOTSTRAP_RESAMPLES


def test_tokenizer_bias_bound_is_reported_relative_to_the_gap_it_must_overturn():
    meta = {
        "wide": {"avg_bytes_per_token": 5.0, "n_test_blocks_scored": 100, "n_train_blocks": 1},
        "narrow": {"avg_bytes_per_token": 4.0, "n_test_blocks_scored": 100, "n_train_blocks": 1},
    }
    # A huge BPB gap cannot be overturned by a tokenizer difference.
    out = am.tokenizer_bias_bound(meta, {"wide": 0.5, "narrow": 1.5})
    assert out["overturns_any_rank"] is False
    # A vanishing gap can be.
    out = am.tokenizer_bias_bound(meta, {"wide": 0.5, "narrow": 0.5001})
    assert out["overturns_any_rank"] is True


def test_committed_artifacts_match_a_fresh_computation():
    """The same gate `make check` runs, so a stale table fails the suite too."""
    report = am.build()
    assert json.loads(am.OUTPUT_JSON.read_text()) == json.loads(
        json.dumps(report, sort_keys=True)
    )
    assert am.OUTPUT_TEX.read_text() == am.render_matrix_tex(report)
    assert am.OUTPUT_REGRET_TEX.read_text() == am.render_regret_tex(report)


def test_every_corpus_and_tier_is_present_and_complete():
    report = am.build()
    assert len(report["cells"]) == len(am.CORPORA) * len(am.TIERS) == 8
    for name, cell in report["cells"].items():
        assert len(cell["bpb"]) == 11, f"{name} is missing models"
        for task, stats in cell["alignment"].items():
            assert -1.0 <= stats["spearman"] <= 1.0
            assert stats["top1_regret_pp"] >= 0.0, f"{name}/{task} regret must be non-negative"
            assert stats["pairs_correct"] <= stats["pairs_comparable"] == 55

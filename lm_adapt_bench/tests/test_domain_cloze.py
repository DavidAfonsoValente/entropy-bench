"""Guards on the in-domain cloze criterion (tools/build_domain_cloze.py, score_domain_cloze.py).

This set exists to answer the circularity objection, so its validity rests entirely on four
construction properties. Each is load-bearing and each is tested here, because a silent regression
in any one of them would turn the experiment back into the thing it was built to escape:

* items come from the pipeline's own held-out split (nothing was adapted on them);
* the answer is not already visible in the prompt (prediction, not retrieval);
* scoring is string match on generated text, never log-likelihood;
* dates are excluded (a single-day snapshot makes the publication date the modal answer, which
  adaptation teaches for free).
"""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

build = pytest.importorskip("build_domain_cloze")
score = pytest.importorskip("score_domain_cloze")


def _doc(filler: str, tail: str) -> str:
    return (filler * 40)[:900] + tail


def test_held_out_split_matches_the_pipeline():
    """Must reproduce DataModule._split_data exactly, or 'never adapted on' is not true."""
    texts = [f"doc{i}" for i in range(1000)]
    got = build.held_out_texts(texts, seed=42, test_split=0.1)

    indices = list(range(1000))
    random.seed(42)
    random.shuffle(indices)
    expected = [texts[i] for i in indices[:100]]
    assert got == expected
    assert len(got) == 100


def test_split_is_deterministic():
    texts = [f"doc{i}" for i in range(500)]
    assert build.held_out_texts(texts, 42, 0.1) == build.held_out_texts(texts, 42, 0.1)


def test_answer_never_appears_in_its_own_context():
    """The core anti-retrieval property, checked on the real generator."""
    docs = [_doc("The council met to discuss the budget. ", "Revenue rose by 47 percent overall.")
            for _ in range(30)]
    items = build.build_items(docs, n_items=10, seed=1, patterns=build.PATTERNS)
    for it in items:
        assert it["answer"] not in it["context"], it


def test_dates_are_excluded_by_default_and_restorable():
    kinds = {k for k, _ in build.PATTERNS}
    assert "date" not in kinds
    assert build.DATE_PATTERN[0] == "date"
    with_dates = {k for k, _ in build.PATTERNS + [build.DATE_PATTERN]}
    assert "date" in with_dates


def test_items_have_real_left_context():
    docs = [_doc("Market conditions remained volatile through the quarter. ",
                 "Output climbed to 731 units.") for _ in range(20)]
    items = build.build_items(docs, n_items=5, seed=2, patterns=build.PATTERNS)
    assert items
    for it in items:
        assert len(it["context"]) > 0
        assert len(it["context"]) <= build.CONTEXT_CHARS


def test_one_item_per_document_at_most():
    docs = [_doc("Numbers everywhere 11 22 33 44 55 66 77 88. ", "Totals reached 991 today.")
            for _ in range(12)]
    items = build.build_items(docs, n_items=50, seed=3, patterns=build.PATTERNS)
    assert len(items) <= len(docs)


# ---------------------------------------------------------------- scoring
def test_strict_requires_the_answer_first():
    assert score.score_one("126 satellites", "126") == (True, True)
    assert score.score_one("about 126 satellites", "126") == (False, True)


def test_lenient_window_is_bounded():
    """Lenient must not credit a model for mentioning the answer far downstream."""
    far = "x" * 200 + " 126"
    strict, lenient = score.score_one(far, "126")
    assert strict is False and lenient is False


def test_scoring_is_whitespace_insensitive():
    assert score.score_one("  126\n", "126")[0] is True


def test_wrong_answer_scores_zero_both_ways():
    assert score.score_one("Moodys said", "Fitch Ratings") == (False, False)


def test_scoring_uses_no_logprobs():
    """The criterion must not be likelihood-based, or it is convergent validity with BPB."""
    src = (ROOT / "tools" / "score_domain_cloze.py").read_text()
    for banned in ("logprob", "logprobs", "prompt_logprobs"):
        assert banned not in src, f"{banned} would make the criterion likelihood-based"

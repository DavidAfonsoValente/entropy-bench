#!/usr/bin/env python3
"""Pin the hand-rolled ROUGE-L in tools/analyze_downstream.py.

rouge_score is not installed on this cluster, so the metric that decides E11's verdict is written
here rather than imported. A metric nobody checked is a verdict nobody checked.

    python tools/test_downstream_metric.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_downstream import first_paragraph, lcs_length, rouge_l, tokenize  # noqa: E402


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def main() -> None:
    assert lcs_length([], ["a"]) == 0
    assert lcs_length(list("abcde"), list("ace")) == 3
    assert lcs_length(list("abc"), list("abc")) == 3
    assert lcs_length(list("abc"), list("xyz")) == 0

    # Identical text is 1.0; disjoint text is 0.0.
    assert close(rouge_l("the cat sat", "the cat sat"), 1.0)
    assert close(rouge_l("alpha beta", "gamma delta"), 0.0)

    # Worked by hand: LCS("the cat sat on the mat", "the cat sat") = 3 tokens.
    # precision 3/6, recall 3/3, F1 = 2*0.5*1/1.5 = 2/3.
    assert close(rouge_l("the cat sat on the mat", "the cat sat"), 2 / 3)

    # Order matters -- a bag-of-words metric would score these the same and ROUGE-L must not.
    assert rouge_l("a b c d", "d c b a") < rouge_l("a b c d", "a b c d")

    # Case and punctuation are normalised away; empty candidate scores zero rather than crashing.
    assert close(rouge_l("The Cat, sat.", "the cat sat"), 1.0)
    assert close(rouge_l("", "the cat"), 0.0)
    assert tokenize("Reuters -- 12 dead") == ["reuters", "12", "dead"]

    # The stop rule keeps the first paragraph only, whichever blank-line spelling the model emits.
    assert first_paragraph("lede line\n\ntrailing waffle") == "lede line"
    assert first_paragraph("lede line\n   \nmore") == "lede line"
    assert first_paragraph("single paragraph only") == "single paragraph only"

    print("downstream metric: PROBLEMS 0")


if __name__ == "__main__":
    main()

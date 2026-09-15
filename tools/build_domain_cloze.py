#!/usr/bin/env python3
"""Build an in-domain cloze set: the outcome criterion this study does not otherwise have.

Every result in this paper is validated against MMLU-Pro, HellaSwag and GSM8K -- the same public
suites whose preparability motivates the work. That circularity is the sharpest objection to the
whole approach (see paper_sota.tex, "The validation is circular"), and no amount of additional
analysis on the existing artifacts can answer it: it needs a criterion that is *not* one of those
suites.

This builds one, with four properties that together make it non-circular:

1. **In-domain.** Items come from the target corpus itself, so it measures the thing a practitioner
   actually cares about rather than a domain-general proxy.
2. **Uncontaminated by construction.** Items are drawn from the pipeline's own held-out TEST split,
   reproduced here with the same shuffle and seed as ``lm_adapt_bench.data.DataModule``, so no model
   was adapted on them; and the corpora postdate every candidate's release, so no model was
   pretrained on them either.
3. **Not a likelihood metric.** The model must *generate* the missing span and be scored by exact
   match. HellaSwag agreement is partly convergent-validity -- it scores continuations by
   log-likelihood, the same object BPB measures -- so a likelihood-based criterion would be
   tautological here. Greedy generation plus string match is not.
4. **Prediction, not copying.** A span that already appears earlier in the prompt can be recovered
   by in-context retrieval alone. Those are excluded, so a correct answer reflects domain knowledge
   rather than string matching.

    python tools/build_domain_cloze.py --corpus <jsonl> --out <json> [--n-items 600]

Standard library only. Deterministic: same corpus and seed give byte-identical output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

# Mirrors lm_adapt_bench.config.DataConfig. Imported rather than duplicated where possible; these
# are the fallbacks so the script also runs on a box without the package installed.
DEFAULT_SEED = 42
DEFAULT_TEST_SPLIT = 0.1

# Characters of left context handed to the model. Long enough that the answer is genuinely
# predictable from the article, short enough to fit every candidate's context window at the
# 512-token block length the rest of the study uses.
CONTEXT_CHARS = 1200
# Require this much preceding text so the model is never asked to predict from a cold start.
MIN_PREFIX_CHARS = 500

# Span types. Numbers and entities are the sharpest test -- they are the content positions
# Section 5.2 shows adaptation improves least, and they cannot be guessed from syntax.
#
# DATES ARE DELIBERATELY EXCLUDED. Inspecting the first build showed two unrelated articles whose
# masked span was the same date, because a single-day news snapshot makes the publication date the
# modal answer. A model that has merely learned "this corpus is 8 June" answers those for free, and
# adaptation teaches exactly that -- so date items would credit the adapted tier for calendar
# memorisation rather than domain knowledge, biasing the experiment toward the result we are hoping
# to find. --include-dates restores them for anyone who wants to check that claim.
PATTERNS = [
    ("number", re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})*(?:\.\d+)?%?(?![\w.])")),
    ("entity", re.compile(r"\b(?:[A-Z][a-z]{2,})(?:\s+[A-Z][a-z]{2,}){1,2}\b")),
]
DATE_PATTERN = ("date", re.compile(r"\b(?:January|February|March|April|May|June|July|August|"
                                   r"September|October|November|December)\s+\d{1,2}\b"))

# Sentence-initial capitals are grammar, not knowledge; a span right after a full stop is
# predictable from punctuation alone and would inflate every model equally.
SENTENCE_START = re.compile(r"[.!?]\s+$")


def load_corpus(path: Path, text_field: str = "text") -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue          # a truncated final line in a streamed sample
            value = row.get(text_field)
            if isinstance(value, str) and value.strip():
                texts.append(value)
    if not texts:
        raise SystemExit(f"no '{text_field}' records in {path}")
    return texts


def held_out_texts(texts: list[str], seed: int, test_split: float) -> list[str]:
    """The pipeline's own test split, reproduced exactly.

    Mirrors DataModule._split_data: seed, shuffle the index list, take the first ``test_split``
    fraction. Reproducing it rather than approximating it is what lets us say no model was adapted
    on these documents.
    """
    indices = list(range(len(texts)))
    random.seed(seed)
    random.shuffle(indices)
    n_test = int(len(texts) * test_split)
    return [texts[i] for i in indices[:n_test]]


def candidate_spans(doc: str, patterns):
    """Yield (kind, start, end) for spans deep enough in the document to have real context."""
    for kind, pattern in patterns:
        for m in pattern.finditer(doc):
            if m.start() < MIN_PREFIX_CHARS:
                continue
            yield kind, m.start(), m.end()


def build_items(docs: list[str], n_items: int, seed: int, patterns,
                max_per_doc: int = 1) -> list[dict]:
    """Sample masked spans, at most ``max_per_doc`` from any one document.

    One per document is the default and the right choice whenever the corpus can supply enough
    documents: it makes items independent, so no single article can dominate a model's score.

    It is not always affordable. The arXiv mathematics corpus holds 3,207 documents against the
    news corpus's tens of thousands, and the held-out split is 10% of that, so one item per document
    yields ~320 items -- a standard error near 0.02 on an accuracy around 0.15, which is wider than
    the gaps between adjacent models in this cohort and would order them by noise. Raising the cap
    buys items at the cost of within-document dependence, and that trade is stated in the artifact
    (``max_per_doc`` and ``n_held_out_docs``) rather than hidden: the items are no longer
    independent, so an item-level interval would be too narrow. The uncertainty this study reports
    is bootstrapped over MODELS, not items, which is why the trade is acceptable here.
    """
    if max_per_doc < 1:
        raise ValueError(f"max_per_doc must be at least 1, got {max_per_doc}")
    rng = random.Random(seed)
    items: list[dict] = []
    for doc_i, doc in enumerate(docs):
        taken = 0
        spans = list(candidate_spans(doc, patterns))
        rng.shuffle(spans)
        for kind, start, end in spans:
            answer = doc[start:end]
            prefix = doc[:start]
            if SENTENCE_START.search(prefix[-8:]):
                continue
            context = prefix[-CONTEXT_CHARS:]
            # The whole point: if the answer is already in the visible context, a correct
            # response proves retrieval, not domain prediction.
            if answer in context:
                continue
            # Keep the continuation unambiguous: the model must produce the span, so the answer
            # should not be a fragment of a longer token run.
            if end < len(doc) and doc[end].isalnum():
                continue
            items.append({
                "id": f"cloze_{doc_i}_{hashlib.sha256(answer.encode()).hexdigest()[:8]}",
                "kind": kind,
                "context": context,
                "answer": answer,
                "answer_chars": len(answer),
            })
            taken += 1
            if taken >= max_per_doc:
                break
        if len(items) >= n_items:
            break
    return items[:n_items]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-items", type=int, default=600)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--test-split", type=float, default=DEFAULT_TEST_SPLIT)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--include-dates", action="store_true",
                    help="restore date spans; excluded by default, see PATTERNS")
    ap.add_argument("--max-per-doc", type=int, default=1,
                    help="items to take from one document. 1 keeps items independent and is "
                         "correct whenever the corpus has enough documents; raise it only when it "
                         "does not, and expect within-document dependence (see build_items)")
    a = ap.parse_args()

    texts = load_corpus(a.corpus, a.text_field)
    docs = held_out_texts(texts, a.seed, a.test_split)
    patterns = PATTERNS + ([DATE_PATTERN] if a.include_dates else [])
    if a.max_per_doc < 1:
        raise SystemExit("--max-per-doc must be at least 1")
    items = build_items(docs, a.n_items, a.seed, patterns, a.max_per_doc)
    if len(items) < a.n_items:
        print(f"warning: only {len(items)} items from {len(docs)} held-out docs", file=sys.stderr)

    by_kind: dict[str, int] = {}
    for it in items:
        by_kind[it["kind"]] = by_kind.get(it["kind"], 0) + 1

    payload = {
        "schema_version": 1,
        "corpus": a.corpus.name,
        "n_source_docs": len(texts),
        "n_held_out_docs": len(docs),
        "split": {"seed": a.seed, "test_split": a.test_split,
                  "note": "reproduces lm_adapt_bench.data.DataModule._split_data"},
        "context_chars": CONTEXT_CHARS,
        "min_prefix_chars": MIN_PREFIX_CHARS,
        "max_per_doc": a.max_per_doc,
        # Narrow claim: one item per document removes WITHIN-document repetition. It does not make
        # documents independent of each other -- they may still be topically or temporally
        # clustered -- so this field is about span repetition and nothing more.
        "no_within_document_repetition": a.max_per_doc == 1,
        "exclusions": [
            "answer already present in the visible context (retrieval, not prediction)",
            "span at a sentence start (predictable from punctuation)",
            "span abutting an alphanumeric character (ambiguous boundary)",
        ] + ([] if a.include_dates else
             ["date spans (a single-day snapshot makes the publication date the modal answer, "
              "which adaptation teaches for free)"]),
        "n_items": len(items),
        "items_by_kind": by_kind,
        "items": items,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(items)} items from {len(docs)} held-out docs of {len(texts)}  {by_kind}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

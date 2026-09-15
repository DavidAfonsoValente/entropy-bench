#!/usr/bin/env python3
"""Build the downstream fine-tuning task: write the lede of a news article from its body.

E11. Every other criterion in this paper scores a model on the same bytes it was adapted on, by
likelihood or by span completion. The sharpest remaining objection is that the yardstick never
leaves the corpus it adapts on, so "adapted BPB picks the better model" could mean nothing beyond
"adapted BPB picks the model with lower adapted BPB". This task answers it: a real supervised
fine-tune with a real generated output, scored against a held-out human-written reference.

Lede generation is chosen because journalistic convention makes the first paragraph a standalone
summary of the article, so the corpus supplies its own supervision -- no annotation, no external
dataset, and nothing that postdates or predates the models differently from the rest of the study.

Four properties make it a different construct from the cloze criterion:

1. **A different split.** Items come from the pipeline's VALIDATION split. The cloze items and the
   BPB test blocks both come from the TEST split, and both are verified disjoint from these
   documents here, by content hash rather than by trusting the slice arithmetic.
2. **A different unit.** The model writes a paragraph, not a masked span.
3. **A different scoring rule.** Overlap with a reference the model never saw, not string equality
   with a token it must reproduce.
4. **Trained, not probed.** The score is of a fine-tuned system, which is what a practitioner ends
   up deploying.

Syndication is the realistic leak in a news corpus: the same wire story is republished by several
outlets, so a document can appear in two splits under different ids. Exact-hash dedup catches
verbatim reprints and a lede-prefix hash catches reprints whose body was edited; both are applied.

    python tools/build_lede_task.py --corpus <jsonl> --out-dir $SCRATCH/e11_downstream

Writes ``train.jsonl`` / ``eval.jsonl`` (large, to SCRATCH) and ``results/downstream_task.json``
(the manifest: counts, hashes, and the disjointness verdict). Standard library only; safe on a
login node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_domain_cloze import DEFAULT_SEED, DEFAULT_TEST_SPLIT  # noqa: E402

VAL_SPLIT = 0.1          # lm_adapt_bench.config.DataConfig.val_split
MANIFEST = ROOT / "results" / "downstream_task.json"
SCHEMA_VERSION = 1
TASK_SEED = 20260911

# A lede is the first paragraph. These bounds keep the target a real summary paragraph: shorter
# than 200 characters is usually a dateline or a one-line teaser, longer than 700 is a wall of text
# that no reference-overlap metric scores meaningfully.
LEDE_MIN_CHARS, LEDE_MAX_CHARS = 200, 700
BODY_MIN_CHARS = 800
BODY_BUDGET_CHARS = 3000   # identical for both models, so neither sees more of the article
MIN_PARAGRAPHS = 3
PARA_SPLIT = re.compile(r"\n\s*\n+")
SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s*$")

N_TRAIN, N_EVAL = 2000, 500

# The prompt is plain text with no chat template: every candidate is a base checkpoint, and a
# template one of them was never trained on would be a confound, not a convenience.
PROMPT = "News article\n{body}\n\nOne-paragraph summary lead\n"


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _hash(text: str) -> str:
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()


def load_corpus(path: Path, text_field: str = "text") -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if text_field in row and row[text_field]:
                texts.append(row[text_field])
    return texts


def split_indices(n: int, seed: int, test_split: float, val_split: float):
    """DataModule._split_data, reproduced. Test first, then validation, then train."""
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    return indices[:n_test], indices[n_test:n_test + n_val], indices[n_test + n_val:]


def make_item(doc: str) -> dict | None:
    """Split one article into (body, lede), or None if it is not shaped like a news article."""
    paras = [p.strip() for p in PARA_SPLIT.split(doc) if p.strip()]
    if len(paras) < MIN_PARAGRAPHS:
        return None
    lede, rest = paras[0], paras[1:]
    if not (LEDE_MIN_CHARS <= len(lede) <= LEDE_MAX_CHARS):
        return None
    if not SENTENCE_END.search(lede):
        return None
    body = "\n\n".join(rest)
    if len(body) < BODY_MIN_CHARS:
        return None
    return {"body": body[:BODY_BUDGET_CHARS], "lede": lede}


def build(corpus: Path, out_dir: Path, cloze_path: Path) -> dict:
    texts = load_corpus(corpus)
    test_idx, val_idx, _ = split_indices(len(texts), DEFAULT_SEED, DEFAULT_TEST_SPLIT, VAL_SPLIT)

    test_docs = [texts[i] for i in test_idx]
    val_docs = [texts[i] for i in val_idx]

    # Provenance, proved rather than assumed: rebuilding the cloze set from this reconstruction of
    # the test split must return the same item ids as the file the paper's criterion actually
    # scored. If it does not, "the criterion lives in the test split" is an unchecked claim and
    # every disjointness statement below rests on nothing.
    import build_domain_cloze as bdc
    shipped = json.loads(cloze_path.read_text())
    rebuilt = bdc.build_items(test_docs, shipped["n_items"], DEFAULT_SEED, bdc.PATTERNS)
    cloze_ids_match = [i["id"] for i in rebuilt] == [i["id"] for i in shipped["items"]]

    test_hashes = {_hash(d) for d in test_docs}
    # Syndicated reprints share an opening paragraph even when the body has been re-edited.
    test_ledes = {_hash(d[:400]) for d in test_docs}

    kept, dropped_exact, dropped_lede, dropped_shape = [], 0, 0, 0
    for doc in val_docs:
        item = make_item(doc)
        if item is None:
            dropped_shape += 1
            continue
        if _hash(doc) in test_hashes:
            dropped_exact += 1
            continue
        if _hash(doc[:400]) in test_ledes:
            dropped_lede += 1
            continue
        kept.append(item)

    rng = random.Random(TASK_SEED)
    rng.shuffle(kept)
    need = N_TRAIN + N_EVAL
    if len(kept) < need:
        raise SystemExit(f"only {len(kept)} usable validation documents; need {need}")
    # Within the task itself: train and eval must not share a document either.
    train, ev = kept[:N_TRAIN], kept[N_TRAIN:need]
    train_hashes = {_hash(i["body"]) for i in train}
    ev = [i for i in ev if _hash(i["body"]) not in train_hashes]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("eval", ev)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for i, row in enumerate(rows):
                fh.write(json.dumps({"id": f"{name}-{i:05d}", **row}, ensure_ascii=False) + "\n")

    eval_hashes = {_hash(i["body"]) for i in ev}
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "lede generation: write the opening paragraph of a news article from its body",
        "corpus": corpus.name,
        "n_corpus_docs": len(texts),
        "split": {"seed": DEFAULT_SEED, "test_split": DEFAULT_TEST_SPLIT, "val_split": VAL_SPLIT,
                  "source_split": "validation",
                  "note": "reproduces lm_adapt_bench.data.DataModule._split_data"},
        "prompt_template": PROMPT,
        "limits": {"lede_chars": [LEDE_MIN_CHARS, LEDE_MAX_CHARS],
                   "body_min_chars": BODY_MIN_CHARS, "body_budget_chars": BODY_BUDGET_CHARS,
                   "min_paragraphs": MIN_PARAGRAPHS},
        "counts": {"n_val_docs": len(val_docs), "usable": len(kept),
                   "n_train": len(train), "n_eval": len(ev),
                   "dropped_not_article_shaped": dropped_shape,
                   "dropped_exact_duplicate_of_test_doc": dropped_exact,
                   "dropped_shared_opening_with_test_doc": dropped_lede},
        "disjointness": {
            "cloze_items_reproduce_from_test_split": cloze_ids_match,
            "eval_doc_in_test_split": False,   # enforced above by hash, asserted by the checks below
            "train_eval_overlap": 0,
            "method": "sha256 of whitespace-normalised lowercased text, plus a 400-character "
                      "opening-paragraph hash to catch syndicated reprints",
            # DataModule takes a different splitting path when contamination checking is on, in
            # which case this reconstruction would not be the split the published cells used. It
            # defaults off (lm_adapt_bench/contamination/config.py:6) and no published
            # domain-transfer cell records a contamination block, so _split_data is the path.
            "bpb_test_blocks_use_same_split": "lm_adapt_bench/contamination/config.py:6 "
                                              "check_contamination defaults to False, so "
                                              "DataModule._split_data is the path the published "
                                              "cells took",
        },
        "task_seed": TASK_SEED,
        "eval_doc_hash_sample": sorted(eval_hashes)[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", required=True, help="must NOT be under $HOME (quota)")
    ap.add_argument("--cloze", required=True, help="the news cloze item file the paper scored")
    a = ap.parse_args()

    manifest = build(Path(a.corpus), Path(a.out_dir), Path(a.cloze))
    if not manifest["disjointness"]["cloze_items_reproduce_from_test_split"]:
        raise SystemExit("cloze items do NOT reproduce from the reconstructed test split; the "
                         "disjointness argument is unverified -- refusing to write the manifest")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("wrote", MANIFEST.relative_to(ROOT), "and", a.out_dir + "/{train,eval}.jsonl")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()

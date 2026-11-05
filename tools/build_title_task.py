#!/usr/bin/env python3
"""Build the second downstream fine-tuning task: write a mathematics paper's title from its abstract.

E13. The news lede task (tools/build_lede_task.py) is the paper's only fine-tuned outcome, so every
claim about predicting fine-tuning rests on one domain. This task moves to the arXiv mathematics
corpus with the same construction: the corpus supplies its own reference (the title the authors
wrote), nothing is annotated, and the fine-tune and its scoring are the news task's, unchanged.

A corpus row is several papers in a row, each a title paragraph followed by an abstract paragraph.
Items are (abstract, title) pairs taken from the train and validation splits only. The test split is
the text BPB is scored on, and it is reconstructed with the pipeline's own DataModule, whose test
texts reproduce the scored byte count exactly (asserted below), so disjointness from it is checked
against the real split, by hash of every test paragraph.

Rows carry the field names tools/downstream_sft.py reads, ``body`` (the input) and ``lede`` (the
target). ``--direction title`` (E13) writes the title from the abstract; ``--direction abstract``
(E13b) writes the abstract from the title, from the same pairs and the same shuffle.

    python tools/build_title_task.py --corpus corpora/math.jsonl --out-dir $SCRATCH/e13_title
    python tools/build_title_task.py --corpus corpora/math.jsonl --out-dir $SCRATCH/e13b_abstract \
        --direction abstract

Writes ``train.jsonl`` / ``eval.jsonl`` to --out-dir and the manifest
(``results/downstream_math_task.json`` or ``results/downstream_math_abstract_task.json``).
CPU only; the corpus is 15 MB.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_lede_task import PARA_SPLIT, _hash  # noqa: E402
from lm_adapt_bench.config import DataConfig  # noqa: E402
from lm_adapt_bench.data import DataModule  # noqa: E402

MANIFESTS = {"title": ROOT / "results" / "downstream_math_task.json",
             "abstract": ROOT / "results" / "downstream_math_abstract_task.json"}
SCHEMA_VERSION = 1
TASK_SEED = 20261103
# Bytes the document reading scored on arXiv maths (results/doc_protocol_cells/math_doc__*.json).
SCORED_TEST_BYTES = 1_507_154
N_TRAIN, N_EVAL = 2000, 500
# A title is one short line; an abstract is a paragraph several times longer than its title.
TITLE_CHARS, ABSTRACT_CHARS = (10, 250), (200, 2000)
PROMPTS = {"title": "Mathematics paper abstract\n{body}\n\nTitle\n",
           "abstract": "Mathematics paper title\n{body}\n\nAbstract\n"}
TASKS = {"title": "title generation: write a mathematics paper's title from its abstract",
         "abstract": "abstract generation: write a mathematics paper's abstract from its title"}


def pairs_of(doc: str) -> list[dict]:
    paras = [p.strip() for p in PARA_SPLIT.split(doc) if p.strip()]
    out = []
    for title, abstract in zip(paras[0::2], paras[1::2]):
        if "\n" in title or not TITLE_CHARS[0] <= len(title) <= TITLE_CHARS[1]:
            continue
        if not ABSTRACT_CHARS[0] <= len(abstract) <= ABSTRACT_CHARS[1]:
            continue
        if len(abstract) < 3 * len(title):
            continue
        out.append({"body": abstract, "lede": title})
    return out


def build(corpus: Path, out_dir: Path, direction: str = "title") -> dict:
    dm = DataModule(DataConfig(dataset_path=str(corpus), max_seq_len=512, max_samples=40000,
                               mask_injected_special_tokens=True))
    test_bytes = sum(len(t.encode("utf-8")) for t in dm.test_texts)
    if test_bytes != SCORED_TEST_BYTES:
        raise SystemExit(f"test split is {test_bytes} bytes, scored {SCORED_TEST_BYTES}: "
                         "this is not the split BPB was measured on")
    test_paras = {_hash(p) for t in dm.test_texts for p in PARA_SPLIT.split(t) if p.strip()}

    candidates, dropped_in_test = [], 0
    for doc in dm.train_texts + dm.val_texts:
        for item in pairs_of(doc):
            if _hash(item["body"]) in test_paras or _hash(item["lede"]) in test_paras:
                dropped_in_test += 1
                continue
            candidates.append(item)
    # One item per abstract: a paper cross-listed twice must not land in both train and eval.
    seen, kept = set(), []
    for item in candidates:
        h = _hash(item["body"])
        if h not in seen:
            seen.add(h)
            kept.append(item)

    rng = random.Random(TASK_SEED)
    rng.shuffle(kept)
    if len(kept) < N_TRAIN + N_EVAL:
        raise SystemExit(f"only {len(kept)} usable pairs; need {N_TRAIN + N_EVAL}")
    train, ev = kept[:N_TRAIN], kept[N_TRAIN:N_TRAIN + N_EVAL]
    if direction == "abstract":
        train, ev = ([{"body": r["lede"], "lede": r["body"]} for r in rows] for rows in (train, ev))

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("eval", ev)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for i, row in enumerate(rows):
                fh.write(json.dumps({"id": f"{name}-{i:05d}", **row}, ensure_ascii=False) + "\n")

    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASKS[direction],
        "corpus": corpus.name,
        "source_splits": ["train", "validation"],
        "split": {"n_train_docs": len(dm.train_texts), "n_val_docs": len(dm.val_texts),
                  "n_test_docs": len(dm.test_texts), "test_bytes": test_bytes,
                  "note": "lm_adapt_bench.data.DataModule, the loader the BPB cells used"},
        "prompt_template": PROMPTS[direction],
        "limits": {"title_chars": list(TITLE_CHARS), "abstract_chars": list(ABSTRACT_CHARS),
                   "abstract_at_least_x_title": 3},
        "counts": {"pairs_found": len(candidates) + dropped_in_test, "usable": len(kept),
                   "dropped_paragraph_in_test_split": dropped_in_test,
                   "n_train": len(train), "n_eval": len(ev)},
        "disjointness": {"eval_or_train_paragraph_in_test_split": False, "train_eval_overlap": 0,
                         "method": "sha256 of whitespace-normalised lowercased paragraphs"},
        "task_seed": TASK_SEED,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", required=True, help="must NOT be under $HOME (quota)")
    ap.add_argument("--direction", choices=sorted(PROMPTS), default="title")
    a = ap.parse_args()
    manifest = build(Path(a.corpus), Path(a.out_dir), a.direction)
    MANIFESTS[a.direction].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("wrote", MANIFESTS[a.direction].relative_to(ROOT), "and", a.out_dir + "/{train,eval}.jsonl")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()

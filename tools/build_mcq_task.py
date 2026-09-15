#!/usr/bin/env python3
"""Build the in-domain multiple-choice task: which continuation belongs to this article?

E12. Every criterion in this paper so far -- the cloze set and the lede fine-tune -- scores a model
on producing in-domain prose, and both readers of the final draft made the same objection: a metric
that rewards in-domain text prediction is close kin to the selector, so agreement may be definitional
rather than informative.

This task is built to break that kinship while keeping everything else fixed:

* **The output is a label, not prose.** The model emits one letter; the metric is accuracy, with no
  n-gram overlap anywhere in it.
* **The format is the one the paper criticises.** Four options, one correct, exactly the shape of
  MMLU-Pro -- but written from the target corpus and postdating nothing, so it cannot be prepared for.
* **The answer cannot be produced by fluent writing.** It must be selected, which is what a
  discriminative product task actually asks.

Items come from the pipeline's VALIDATION split, like the lede task, and disjointness from the test
split (where the cloze items and the BPB blocks live) is verified by content hash rather than assumed
-- the same two checks, reused from tools/build_lede_task.py.

    python tools/build_mcq_task.py --corpus <jsonl> --out-dir $SCRATCH/e12_mcq --cloze <cloze json>

Writes train.jsonl / eval.jsonl (to SCRATCH) and results/mcq_task.json. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_domain_cloze import DEFAULT_SEED, DEFAULT_TEST_SPLIT  # noqa: E402
from build_lede_task import (  # noqa: E402  (one splitter and one hash rule, not two)
    PARA_SPLIT, VAL_SPLIT, _hash, load_corpus, split_indices,
)

MANIFEST = ROOT / "results" / "mcq_task.json"
SCHEMA_VERSION = 1
TASK_SEED = 20260912
N_OPTIONS = 4
LETTERS = "ABCD"
# The opening the model is given, and the candidate paragraphs it chooses between. Options are
# length-matched within a band so the answer cannot be picked by length alone.
OPENING_MIN, OPENING_MAX = 200, 900
OPTION_MIN, OPTION_MAX = 180, 600
LENGTH_BAND = 1.6
MIN_PARAGRAPHS = 3
N_TRAIN, N_EVAL = 2000, 500
# Random distractors are usually off-topic, so the answer can be found by topic match alone -- the
# first build put both pilot models near ceiling. Hard mode picks distractors that SHARE rare
# vocabulary with the opening, so every option is about the same event and the choice needs
# coherence rather than topic matching.
WORD = re.compile(r"[a-z][a-z'-]{2,}")
RARE_DF_FRAC = 0.01        # a word in under 1% of articles counts as rare (a name, a place)
CANDIDATE_POOL = 60        # rank this many topical neighbours, then length-filter

PROMPT = ("News article opening\n{opening}\n\nWhich paragraph continues this article?\n{options}\n"
          "Answer with A, B, C or D.\nAnswer:")


def candidates(doc: str) -> tuple[str, str] | None:
    paras = [p.strip() for p in PARA_SPLIT.split(doc) if p.strip()]
    if len(paras) < MIN_PARAGRAPHS:
        return None
    opening, cont = paras[0], paras[1]
    if not (OPENING_MIN <= len(opening) <= OPENING_MAX):
        return None
    if not (OPTION_MIN <= len(cont) <= OPTION_MAX):
        return None
    return opening, cont


def _rare_index(openings: list[str], conts: list[str]):
    """Inverted index from rare word -> continuations containing it, plus each opening's rare set."""
    df = Counter()
    toks = []
    for t in conts:
        w = set(WORD.findall(t.lower()))
        toks.append(w)
        df.update(w)
    cutoff = max(2, int(RARE_DF_FRAC * len(conts)))
    index: dict[str, list[int]] = {}
    for i, w in enumerate(toks):
        for x in w:
            if df[x] <= cutoff:
                index.setdefault(x, []).append(i)
    open_rare = [{x for x in set(WORD.findall(o.lower())) if 0 < df[x] <= cutoff} for o in openings]
    return index, open_rare


def build(corpus: Path, out_dir: Path, cloze_path: Path, hard: bool = False) -> dict:
    texts = load_corpus(corpus)
    test_idx, val_idx, _ = split_indices(len(texts), DEFAULT_SEED, DEFAULT_TEST_SPLIT, VAL_SPLIT)
    test_docs = [texts[i] for i in test_idx]
    val_docs = [texts[i] for i in val_idx]

    import build_domain_cloze as bdc
    shipped = json.loads(cloze_path.read_text())
    rebuilt = bdc.build_items(test_docs, shipped["n_items"], DEFAULT_SEED, bdc.PATTERNS)
    cloze_ids_match = [i["id"] for i in rebuilt] == [i["id"] for i in shipped["items"]]

    test_hashes = {_hash(d) for d in test_docs}
    test_openings = {_hash(d[:400]) for d in test_docs}

    kept, dropped_shape, dropped_dup = [], 0, 0
    for doc in val_docs:
        c = candidates(doc)
        if c is None:
            dropped_shape += 1
            continue
        if _hash(doc) in test_hashes or _hash(doc[:400]) in test_openings:
            dropped_dup += 1
            continue
        kept.append(c)

    rng = random.Random(TASK_SEED)
    rng.shuffle(kept)
    need = N_TRAIN + N_EVAL
    if len(kept) < need + 200:
        raise SystemExit(f"only {len(kept)} usable validation documents; need {need}+")

    pool = [c for _, c in kept]          # every continuation is a possible distractor
    index, open_rare = _rare_index([o for o, _ in kept], pool) if hard else (None, None)
    items = []
    for i, (opening, cont) in enumerate(kept[:need]):
        lo, hi = len(cont) / LENGTH_BAND, len(cont) * LENGTH_BAND
        pick: list[str] = []
        if hard:
            # Rank other articles' continuations by how much rare vocabulary they share with this
            # opening; the top ones are about the same event, so topic no longer gives the answer.
            share: Counter = Counter()
            for x in open_rare[i]:
                for j in index.get(x, ()):
                    if j != i:
                        share[j] += 1
            for j, _ in share.most_common(CANDIDATE_POOL):
                cand = pool[j]
                if cand is cont or not (lo <= len(cand) <= hi) or cand in pick:
                    continue
                pick.append(cand)
                if len(pick) == N_OPTIONS - 1:
                    break
        tries = 0
        # Length-matched random fill, for hard mode too when an article has few topical neighbours.
        while len(pick) < N_OPTIONS - 1 and tries < 400:
            tries += 1
            cand = pool[rng.randrange(len(pool))]
            if cand is cont or not (lo <= len(cand) <= hi) or cand in pick:
                continue
            pick.append(cand)
        if len(pick) < N_OPTIONS - 1:
            continue
        options = pick + [cont]
        rng.shuffle(options)
        answer = LETTERS[options.index(cont)]
        block = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(options))
        items.append({"opening": opening, "options": block, "answer": answer})

    train, ev = items[:N_TRAIN], items[N_TRAIN:need]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("eval", ev)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for i, r in enumerate(rows):
                # Same field names the fine-tuning tool already reads: the prompt body and target.
                fh.write(json.dumps({"id": f"{name}-{i:05d}",
                                     "body": PROMPT.format(opening=r["opening"],
                                                           options=r["options"]),
                                     "lede": " " + r["answer"], "answer": r["answer"]},
                                    ensure_ascii=False) + "\n")
    dist = {L: sum(1 for r in ev if r["answer"] == L) for L in LETTERS}
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "in-domain multiple choice: which paragraph continues this article",
        "corpus": corpus.name,
        "split": {"seed": DEFAULT_SEED, "test_split": DEFAULT_TEST_SPLIT, "val_split": VAL_SPLIT,
                  "source_split": "validation",
                  "note": "reproduces lm_adapt_bench.data.DataModule._split_data"},
        # The tool templates {body}; the whole question is already in that field.
        "prompt_template": "{body}",
        "n_options": N_OPTIONS,
        "distractors": ("topical: other articles' continuations ranked by shared rare vocabulary "
                        "with the opening" if hard else "random, length-matched"),
        "chance_accuracy": 1.0 / N_OPTIONS,
        "counts": {"usable": len(kept), "n_train": len(train), "n_eval": len(ev),
                   "dropped_not_article_shaped": dropped_shape,
                   "dropped_duplicate_of_test_doc": dropped_dup},
        "eval_answer_distribution": dist,
        "disjointness": {
            "cloze_items_reproduce_from_test_split": cloze_ids_match,
            "method": "sha256 of whitespace-normalised lowercased text, plus a 400-character "
                      "opening hash to catch syndicated reprints",
        },
        "task_seed": TASK_SEED,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", required=True, help="must NOT be under $HOME (quota)")
    ap.add_argument("--cloze", required=True)
    ap.add_argument("--hard-distractors", action="store_true",
                    help="pick distractors that share rare vocabulary with the opening")
    a = ap.parse_args()
    m = build(Path(a.corpus), Path(a.out_dir), Path(a.cloze), a.hard_distractors)
    if not m["disjointness"]["cloze_items_reproduce_from_test_split"]:
        raise SystemExit("cloze items do NOT reproduce from the reconstructed test split")
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    print("wrote", MANIFEST.relative_to(ROOT), "and", a.out_dir + "/{train,eval}.jsonl")
    print(json.dumps({"counts": m["counts"], "answers": m["eval_answer_distribution"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E13: which signal, read before fine-tuning, predicts the fine-tuned maths title writers?

The second downstream task (tools/build_title_task.py): each of the fifteen models is fine-tuned to
write a mathematics paper's title from its abstract with the news task's recipe
(tools/downstream_sft.py), and scored by ROUGE-L against the authors' title. Pairs, bootstrap and
selector table are tools/analyze_downstream.py's, unchanged; only the target and the scorer differ.

**Fixed before any cell returned (ledger, "E13").**
- Guard, from E12: UNINFORMATIVE if any fine-tuned model averages ROUGE-L above 0.95, the spread
  across models is under 0.02, or fewer than 12 of the 15 fine-tuned systems beat their own base
  weights. Nothing is then claimed either way.
- Primary: pairwise accuracy over size-matched pairs of maths document BPB, news document BPB,
  GSM8K, MMLU-Pro, HellaSwag and parameter count. "Maths text predicts the maths fine-tune" is
  claimed if maths document BPB orders at least as many pairs as every benchmark.
- The two-task test: "your domain's text, specifically" is claimed if maths document BPB beats news
  document BPB on this task and news document BPB beats maths document BPB on the news task.
Reported whatever the outcome.

A title is one line, so title generations are cut at the first line break; an abstract is one
paragraph, so abstract generations are cut at the first blank line, as the news task's are. The
same cut applies to every model and to the base-weights control.

``--variant abstract`` (E13b, preregistered after E13's pilot was stopped by the guard) scores the
reverse task, writing the abstract from the title, with the same guard and rules.

    python tools/analyze_downstream_math.py [--variant title|abstract] [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from analyze_downstream import _selectors_17, build_cohort, doc_bpb, first_paragraph, rouge_l  # noqa: E402

SCRATCH = Path(os.environ.get("SCRATCH", "/tmp"))
VARIANTS = {  # cells directory, artifact, task manifest, cut
    "title": (Path(os.environ.get("E13_DIR", str(SCRATCH / "e13_title"))),
              ROOT / "results" / "downstream_math.json", "results/downstream_math_task.json",
              lambda t: t.strip().split("\n", 1)[0].strip()),
    "abstract": (Path(os.environ.get("E13B_DIR", str(SCRATCH / "e13b_abstract"))),
                 ROOT / "results" / "downstream_math_abstract.json",
                 "results/downstream_math_abstract_task.json", first_paragraph),
}
CELLS, OUTPUT_JSON, TASK, CUT = VARIANTS["title"]
GUARD_CEILING, GUARD_SPREAD, GUARD_MIN_IMPROVED = 0.95, 0.02, 12


def score_title(cell: dict, key: str) -> dict[str, float]:
    return {g["id"]: rouge_l(CUT(g["generated"]), g["reference"]) for g in cell[key]}


def selectors() -> dict:
    base, _ = _selectors_17()
    keep = ("gsm8k", "mmlu_pro", "hellaswag", "parameter_count", "math_doc_bpb")
    sel = {k: base[k] for k in keep}
    sel["news_doc_bpb"] = (doc_bpb("news"), True)
    return sel


def guard(cohort: list[str]) -> dict:
    tuned, base = {}, {}
    for m in cohort:
        cell = json.loads((CELLS / f"sft__{m}.json").read_text())
        t, b = score_title(cell, "generations"), score_title(cell, "base_generations")
        tuned[m], base[m] = sum(t.values()) / len(t), sum(b.values()) / len(b)
    improved = sum(tuned[m] > base[m] for m in cohort)
    reasons = [r for r, bad in (
        ("a model is above the ceiling", max(tuned.values()) > GUARD_CEILING),
        ("the spread is under the floor", max(tuned.values()) - min(tuned.values()) < GUARD_SPREAD),
        ("too few systems beat their base weights", improved < GUARD_MIN_IMPROVED)) if bad]
    return {"base_rouge_l": base, "n_improved_over_base": improved, "uninformative": bool(reasons),
            "reasons": reasons}


def pairs_right(result: dict, name: str) -> int:
    return sum(p.get(f"{name}_right", False) for p in result["pairs"])


def build() -> dict:
    sel = selectors()
    out = build_cohort(cells=CELLS, score=score_title, selectors=sel, compared="math_doc_bpb",
                       off_domain=False)
    if out.get("verdict") == "INCOMPLETE":
        return out
    out["task"] = TASK
    out["guard"] = guard(out["cohort"])
    right = {n: pairs_right(out, n) for n in sel}
    out["pairs_right"] = right
    news = json.loads((ROOT / "results" / "downstream.json").read_text())["cohort"]["pairs"]
    news_doc, math_doc = sel["news_doc_bpb"][0], sel["math_doc_bpb"][0]

    def on_news(vals):
        return sum((min(p["pair"].split("|"), key=vals.get)) == p["winner"] for p in news)

    out["two_task"] = {"news_task": {"news_doc_bpb": on_news(news_doc),
                                     "math_doc_bpb": on_news(math_doc), "n_pairs": len(news)},
                       "math_task": {"news_doc_bpb": right["news_doc_bpb"],
                                     "math_doc_bpb": right["math_doc_bpb"],
                                     "n_pairs": len(out["pairs"])}}
    tt = out["two_task"]
    out["verdict"] = "UNINFORMATIVE" if out["guard"]["uninformative"] else {
        "maths_text_predicts": all(right["math_doc_bpb"] >= right[b]
                                   for b in ("gsm8k", "mmlu_pro", "hellaswag")),
        "own_domain_text": (tt["math_task"]["math_doc_bpb"] > tt["math_task"]["news_doc_bpb"]
                            and tt["news_task"]["news_doc_bpb"] > tt["news_task"]["math_doc_bpb"]),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="fail if the artifact is stale")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="title")
    args = ap.parse_args()
    global CELLS, OUTPUT_JSON, TASK, CUT
    CELLS, OUTPUT_JSON, TASK, CUT = VARIANTS[args.variant]
    built = json.loads(json.dumps(build(), sort_keys=True))
    if args.check:
        if not OUTPUT_JSON.exists() or json.loads(OUTPUT_JSON.read_text()) != built:
            sys.exit(f"{OUTPUT_JSON.relative_to(ROOT)} is stale")
        print(f"{OUTPUT_JSON.name} current")
        return
    OUTPUT_JSON.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n")
    print("wrote", OUTPUT_JSON.relative_to(ROOT))
    print(json.dumps({k: built.get(k) for k in ("verdict", "pairs_right", "two_task")}, indent=2))


if __name__ == "__main__":
    main()

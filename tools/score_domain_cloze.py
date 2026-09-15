#!/usr/bin/env python3
"""Score one model on the in-domain cloze set built by tools/build_domain_cloze.py.

This is the outcome criterion the study otherwise lacks. Everything else in the paper is validated
against MMLU-Pro / HellaSwag / GSM8K, which is circular: those are the suites whose preparability
motivates the work. Cloze is in-domain, drawn from the pipeline's own held-out split, postdates
every model's release, and -- critically -- is scored by GREEDY GENERATION AND STRING MATCH rather
than by log-likelihood. A likelihood criterion would agree with BPB for the same reason HellaSwag
does (convergent validity on the same object), and would prove nothing.

    python tools/score_domain_cloze.py --cloze cloze_news.json --model <path> --out <json>

Reports two metrics, deliberately, for the same reason lm-eval reports strict and flexible GSM8K:

* ``strict``  -- the continuation begins with the answer, after whitespace normalisation.
* ``lenient`` -- the answer appears anywhere in a short window of the continuation.

A base model has no instruction following and no stop convention, so it will often continue past
the answer into the rest of the sentence; ``strict`` alone would then measure stopping behaviour
rather than knowledge. Reporting both means a reader can see whether any conclusion depends on the
convention, and the analysis uses ``lenient`` as the headline for exactly that reason.

Per-item generations are written out so a reader can audit what was actually produced.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

WS = re.compile(r"\s+")
# How far past the start of the continuation the lenient metric looks. The answer is a single span
# a few characters long; a window much wider than that starts crediting the model for eventually
# mentioning the right entity somewhere in a paragraph, which is not the task.
LENIENT_WINDOW = 24


def normalise(text: str) -> str:
    return WS.sub(" ", text).strip()


def score_one(generated: str, answer: str) -> tuple[bool, bool]:
    gen = normalise(generated)
    ans = normalise(answer)
    strict = gen.startswith(ans)
    lenient = ans in gen[: len(ans) + LENIENT_WINDOW]
    return strict, lenient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cloze", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default=None, help="short name recorded in the output")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    # 0 = leave vLLM's own default alone. Needed for hybrid-attention models: Qwen3.5-35B-A3B
    # carries Mamba layers, which reserve one cache block PER CONCURRENT SEQUENCE, so the default
    # cap of 1024 cannot be satisfied at any utilization below 1.0 and CUDA graph capture aborts.
    # This is a scheduler cap, not a decoding parameter -- generation stays greedy and per-prompt.
    ap.add_argument("--max-num-seqs", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N items (smoke test)")
    a = ap.parse_args()

    payload = json.loads(a.cloze.read_text())
    items = payload["items"][: a.limit] if a.limit else payload["items"]

    from vllm import LLM, SamplingParams          # imported late so --help works without vLLM

    extra = {"max_num_seqs": a.max_num_seqs} if a.max_num_seqs else {}
    llm = LLM(model=a.model, tensor_parallel_size=a.tensor_parallel,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
              dtype="bfloat16", trust_remote_code=True, **extra)
    # Greedy and seeded: this must be reproducible, and sampling would add a variance term to a
    # comparison whose whole point is ranking models a few points apart.
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.max_new_tokens, seed=0)

    t0 = time.time()
    outputs = llm.generate([it["context"] for it in items], params)
    wall = time.time() - t0

    records, n_strict, n_lenient = [], 0, 0
    by_kind: dict[str, dict[str, int]] = {}
    for it, out in zip(items, outputs):
        gen = out.outputs[0].text
        strict, lenient = score_one(gen, it["answer"])
        n_strict += strict
        n_lenient += lenient
        k = by_kind.setdefault(it["kind"], {"n": 0, "strict": 0, "lenient": 0})
        k["n"] += 1
        k["strict"] += strict
        k["lenient"] += lenient
        records.append({"id": it["id"], "kind": it["kind"], "answer": it["answer"],
                        "generated": gen[:120], "strict": strict, "lenient": lenient})

    n = len(items)
    result = {
        "schema_version": 1,
        "model": a.model,
        "label": a.label or Path(a.model).name,
        "cloze_set": payload.get("corpus"),
        "cloze_n_items": n,
        "max_new_tokens": a.max_new_tokens,
        "lenient_window_chars": LENIENT_WINDOW,
        "strict_accuracy": n_strict / n,
        "lenient_accuracy": n_lenient / n,
        "by_kind": {k: {"n": v["n"], "strict_accuracy": v["strict"] / v["n"],
                        "lenient_accuracy": v["lenient"] / v["n"]}
                    for k, v in sorted(by_kind.items())},
        "wall_seconds": wall,
        "samples": records,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['label']}: strict {result['strict_accuracy']:.4f}  "
          f"lenient {result['lenient_accuracy']:.4f}  ({n} items, {wall:.0f}s)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

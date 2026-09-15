#!/usr/bin/env python3
"""Fine-tune one model on the lede task and generate its held-out ledes.

E11, the measurement half. ``tools/build_lede_task.py`` makes the data and
``tools/analyze_downstream.py`` computes the verdict; this script does exactly one model and writes
raw generations, so the metric can be changed afterwards without re-running a GPU job.

**Identical preparation is the whole point.** Both members of a pair get the same LoRA
configuration, the same number of optimiser steps, the same effective batch, the same sequence
budget and the same decoding rule. Nothing is tuned on either model -- the recipe is the paper's own
adaptation recipe (rank 16, alpha 32, dropout 0.05, lr 1e-4, effective batch 32, cosine schedule),
carried over unchanged rather than re-searched here, because a per-model search is precisely the
advantage this paper argues a practitioner should not have to buy.

Tokenizers differ, so the matched budget is in update steps x effective batch x sequence length,
and the bytes each model actually saw is recorded in the artifact for the reader to check.

Decoding follows the rule E9 established: the checkpoint's own ``generation_config`` is kept, and
only the decode policy this run sets centrally is overridden. Stripping it wholesale cost
Gemma-4-12B 9.86 GSM8K points once already (docs/RUN_LEDGER.md, E9); it is not repeated here.

    python tools/downstream_sft.py --model-id <hf-id> --task-dir <dir> --out <json>

Run under slurm/downstream_sft.sbatch; never on a login node.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lm_adapt_bench.config import RunConfig  # noqa: E402
from lm_adapt_bench.utils import get_lora_target_modules, get_model_and_tokeniser  # noqa: E402

# The paper's adaptation recipe, verbatim (paper_sota.tex, "The Protocol").
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LEARNING_RATE = 1e-4
EFFECTIVE_BATCH = 32
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0
SEED = 20260911
# Only the decode policy this run owns; everything else in the checkpoint's generation_config
# (bos/eos/pad ids, suppress_tokens) is left alone. See E9.
MAX_NEW_TOKENS = 180
# How many articles had their body cut to fit. Reported per model so a reader can see that the two
# members of a pair saw comparable amounts of each article, rather than taking it on trust.
TRUNCATED = [0]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_prompt_ids(tok, head_ids: list[int], tail_ids: list[int], body: str,
                     budget: int) -> list[int]:
    """Prompt token ids with the body truncated, never the instruction.

    Right-truncating the assembled prompt would cut the trailing "One-paragraph summary lead" line
    off exactly the longest articles, and it would do so at a different article count for each
    tokenizer -- a per-model difference on the pair being compared. Measured over the 500 eval
    articles: one item overflows for Ministral and one for Qwen, none for Gemma or Llama. Small,
    but it is an asymmetry between the two things being compared, so the body is cut instead and
    the instruction always survives. Ids are sliced directly; decoding and re-tokenizing would
    silently change the token stream.
    """
    body_ids = tok(body, add_special_tokens=False)["input_ids"]
    room = budget - len(head_ids) - len(tail_ids)
    if room < 1:
        raise SystemExit("prompt scaffolding alone exceeds the sequence budget")
    if len(body_ids) > room:
        TRUNCATED[0] += 1
    return head_ids + body_ids[:room] + tail_ids


def encode(tok, head_ids, tail_ids, body: str, target: str, max_len: int):
    """Prompt tokens are masked out of the loss: the model is trained to write the lede, not to
    reconstruct the article it was given."""
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    if tok.eos_token_id is not None:
        t_ids = t_ids + [tok.eos_token_id]
    # Truncate the BODY, never the target -- a clipped target would train the model to stop early.
    p_ids = build_prompt_ids(tok, head_ids, tail_ids, body, max_len - len(t_ids))
    return p_ids + t_ids, [-100] * len(p_ids) + list(t_ids)


def collate(batch, pad_id: int):
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a)
        labels[i, :len(b)] = torch.tensor(b)
        mask[i, :len(a)] = 1
    return ids, labels, mask


def generate_all(model, tok, rows, head_ids, tail_ids, max_seq_len, gen_batch, pad_id, tag, t0,
                 max_new=MAX_NEW_TOKENS):
    model.eval()
    out_rows = []
    budget = max_seq_len - max_new
    with torch.no_grad():
        for i in range(0, len(rows), gen_batch):
            chunk = rows[i:i + gen_batch]
            seqs = [build_prompt_ids(tok, head_ids, tail_ids, r["body"], budget) for r in chunk]
            n = max(len(x) for x in seqs)
            # Left padding: right padding would put pad tokens between the prompt and the answer.
            ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), n), dtype=torch.long)
            for j, sq in enumerate(seqs):
                ids[j, n - len(sq):] = torch.tensor(sq)
                mask[j, n - len(sq):] = 1
            dev = next(model.parameters()).device
            out = model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                                 max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=pad_id)
            new = out[:, n:]
            for row, seq in zip(chunk, tok.batch_decode(new, skip_special_tokens=True)):
                out_rows.append({"id": row["id"], "reference": row["lede"], "generated": seq})
            if i % (gen_batch * 20) == 0:
                print(f"[gen {tag} {i}/{len(rows)}] ({time.time() - t0:.0f}s)", flush=True)
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--label", required=True, help="the paper's label for this model")
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1280)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=500)
    # E11c: a DIFFERENT recipe, so the downstream outcome is not produced by the same adaptation
    # the selector uses. Defaults are the paper's recipe (E11/E11b); the sbatch sets recipe B.
    ap.add_argument("--recipe", default="A", help="label written into the artifact")
    # E12 reuses this trainer for a different task: a manifest naming the prompt, and a short
    # generation cap because the answer is one letter rather than a paragraph.
    ap.add_argument("--manifest", default="results/downstream_task.json")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--lora-r", type=int, default=LORA_R)
    ap.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--effective-batch", type=int, default=EFFECTIVE_BATCH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-base", action="store_true",
                    help="skip the unadapted control pass (it is what shows the fine-tune moved "
                         "anything, so only skip it to debug)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("refusing to run on CPU; this belongs in an sbatch job")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    task_dir = Path(args.task_dir)
    manifest = json.loads((ROOT / args.manifest).read_text())
    prompt_tpl = manifest["prompt_template"]
    head, tail = prompt_tpl.split("{body}")
    train_rows = load_jsonl(task_dir / "train.jsonl")
    eval_rows = load_jsonl(task_dir / "eval.jsonl")[:args.n_eval]

    run_cfg = RunConfig(model_id=args.model_id, output_dir=str(task_dir), dtype="bfloat16")
    device = torch.device("cuda")
    model, tok = get_model_and_tokeniser(args.model_id, run_cfg, device, torch.bfloat16)
    gen_cfg_before = model.generation_config.to_dict()

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    head_ids = tok(head, add_special_tokens=True)["input_ids"]
    tail_ids = tok(tail, add_special_tokens=False)["input_ids"]
    t0 = time.time()

    encoded = []
    for row in train_rows:
        encoded.append(encode(tok, head_ids, tail_ids, row["body"], row["lede"],
                              args.max_seq_len))
    if len(encoded) < args.steps * args.effective_batch // 4:
        raise SystemExit(f"only {len(encoded)} usable training examples")
    train_tokens = sum(len(e[0]) for e in encoded)

    base_generations = []
    if not args.skip_base:
        # Generated BEFORE any adapter exists: without it "the fine-tune worked" is an assertion,
        # and a pair could be ordered entirely by how well each base model already writes ledes.
        base_generations = generate_all(model, tok, eval_rows, head_ids, tail_ids,
                                        args.max_seq_len, args.gen_batch, pad_id, "base", t0,
                                        args.max_new_tokens)

    targets = get_lora_target_modules(model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=targets, bias="none"))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # Gemma-4's 256k vocabulary makes the logits tensor the memory peak, not the weights. Recompute
    # activations so both members of a pair fit one 64 GB A100 and neither needs a different
    # hardware shape from the other.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    accum = max(1, args.effective_batch // args.micro_batch)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(args.steps * WARMUP_RATIO), args.steps)

    order = list(range(len(encoded)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    cursor = 0
    losses = []
    model.train()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            if cursor + args.micro_batch > len(order):
                rng.shuffle(order)
                cursor = 0
            batch = [encoded[i] for i in order[cursor:cursor + args.micro_batch]]
            cursor += args.micro_batch
            ids, labels, mask = collate(batch, pad_id)
            dev = next(model.parameters()).device
            out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), labels=labels.to(dev))
            (out.loss / accum).backward()
            total += out.loss.item() / accum
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                       GRAD_CLIP)
        opt.step()
        sched.step()
        losses.append(total)
        if step % 25 == 0:
            print(f"[step {step:3d}] loss {total:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    model.config.use_cache = True
    generations = generate_all(model, tok, eval_rows, head_ids, tail_ids, args.max_seq_len,
                               args.gen_batch, pad_id, "sft", t0, args.max_new_tokens)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "schema_version": 1,
        "model_id": args.model_id,
        "label": args.label,
        "recipe": {"name": args.recipe, "seed": args.seed,
                   "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                   "lora_dropout": LORA_DROPOUT, "lr": args.lr,
                   "effective_batch": args.effective_batch, "steps": args.steps,
                   "max_seq_len": args.max_seq_len, "warmup_ratio": WARMUP_RATIO,
                   "lora_targets": targets if isinstance(targets, str) else sorted(targets)},
        # The matched budget is steps x batch x length; what that buys in tokens and bytes differs
        # per tokenizer, so both are recorded rather than assumed equal.
        "budget": {"usable_train_examples": len(encoded),
                   "train_tokens_in_pool": train_tokens,
                   "optimiser_steps": args.steps,
                   "examples_consumed": args.steps * args.effective_batch,
                   "articles_whose_body_was_truncated": TRUNCATED[0]},
        "loss": {"first": losses[0], "last": losses[-1],
                 "mean_last_25": sum(losses[-25:]) / len(losses[-25:]), "curve": losses},
        # E9: the decode policy is reported, not trusted. A reader can check that nothing but
        # max_new_tokens and greedy sampling was imposed.
        "task_manifest": args.manifest,
        "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                       "generation_config_at_load": {
                           k: v for k, v in gen_cfg_before.items()
                           if k in ("bos_token_id", "eos_token_id", "pad_token_id",
                                    "suppress_tokens", "max_new_tokens", "max_length",
                                    "do_sample", "temperature", "top_p", "top_k")}},
        "n_eval": len(generations),
        "base_generations": base_generations,
        "wall_seconds": time.time() - t0,
        "generations": generations,
    }, indent=2, ensure_ascii=False) + "\n")
    print("wrote", out_path, "| loss", f"{losses[0]:.3f} -> {losses[-1]:.3f}",
          "| n_eval", len(generations))


if __name__ == "__main__":
    main()

"""Zero-shot and adapted BPB for one model on one corpus, under the corrected packing.

Answers the question the team raised: does the BPB ordering hold on domains other than
news? Deliberately reuses lm_adapt_bench's own DataModule and compute_bpb, so the splits,
the packing and the BPB definition are identical to the published run -- except that the
injected document-start token is now masked out of the loss
(DataConfig.mask_injected_special_tokens, see docs/TOKEN_GAIN_FINDINGS.md).

Two departures from the news protocol, both deliberate and both stated in the paper:

  * One fixed LoRA configuration for every model instead of a per-model sweep. The news
    run's sweep budget was unequal (4 completed trials for the largest models against 18
    for the smallest), which biases adapted BPB by model size. A single shared
    configuration removes that confound; it is not tuned to any model's advantage.
  * A fixed step budget rather than training to convergence, so cost per model is
    predictable. This measures adaptation under a fixed budget, not an asymptote.

  python tools/domain_transfer_eval.py --model-id Qwen/Qwen2.5-1.5B \\
      --dataset $WORK/corpora/reddit_2026-08.jsonl --out out/reddit_qwen15.json \\
      --max-train-steps 400
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--dataset", required=True, help="JSONL with a 'text' field")
    p.add_argument("--out", required=True)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap documents read from the corpus, for cost control")
    p.add_argument("--eval-blocks", type=int, default=3000)
    p.add_argument("--eval-batch-size", type=int, default=8)
    # one shared LoRA configuration for every model
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-train-steps", type=int, default=400)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--target-modules", default="all-linear",
                   help="LoRA target spec; identical for every model by design")
    p.add_argument("--zero-shot-only", action="store_true")
    p.add_argument("--device", default="cuda", help="cuda, or cpu for smoke tests")
    return p.parse_args()


def load_model(model_id, device_map="auto"):
    """Load a causal LM, tolerating two things transformers is fussy about.

    * `torch_dtype` was renamed to `dtype` in transformers v5.
    * Some checkpoints in this cohort are multimodal wrappers -- Ministral-3-14B carries a
      Mistral3Config, which AutoModelForCausalLM refuses -- so fall back to the
      image-text-to-text auto class and score its language backbone. This mirrors the
      pipeline's own note that adapters are restricted to the language backbone.
    """
    import transformers
    from transformers import AutoModelForCausalLM
    common = dict(device_map=device_map, trust_remote_code=True)

    def _try(cls):
        for kw in ("dtype", "torch_dtype"):
            try:
                return cls.from_pretrained(model_id, **{kw: torch.bfloat16}, **common)
            except TypeError:
                if kw == "torch_dtype":
                    raise
        return None

    try:
        return _try(AutoModelForCausalLM)
    except ValueError as e:
        if "Unrecognized configuration class" not in str(e):
            raise
        print(f"[warn] {model_id} is not a plain causal LM ({e.__class__.__name__}); "
              f"trying the multimodal auto class", flush=True)
    cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if cls is None:
        raise RuntimeError(f"cannot load {model_id}: no multimodal auto class available")
    m = _try(cls)
    lm = getattr(m, "language_model", None)
    if lm is not None and hasattr(lm, "forward"):
        print(f"[info] scoring the language backbone of {model_id}", flush=True)
    return m


def main():
    a = parse_args()
    from transformers import AutoTokenizer
    from lm_adapt_bench.config import DataConfig
    from lm_adapt_bench.data import DataModule
    from lm_adapt_bench.evaluate import compute_bpb

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model_id, trust_remote_code=True)

    cfg = DataConfig(dataset_path=a.dataset, max_seq_len=a.max_seq_len,
                     max_samples=a.max_samples, mask_injected_special_tokens=True)
    dm = DataModule(cfg)
    # DataModule.__init__ loads and splits; contamination gating is skipped here because
    # both new corpora were scraped in August 2026, after every model in the cohort was
    # released, so the audit has nothing to find by construction. Wikipedia would not
    # have that property -- see the paper's discussion.
    train_ds, val_ds, test_ds = dm.tokenize(a.model_id, tok, a.max_seq_len, os.environ.get("DT_CACHE", "./dt_cache"))
    avg_bpt = dm.calculate_avg_bytes_per_token(tok)
    print(f"[info] {a.model_id} on {Path(a.dataset).name}: "
          f"{len(train_ds)} train / {len(test_ds)} test blocks, "
          f"{avg_bpt:.3f} bytes/token", flush=True)

    device = torch.device(a.device)
    test_subset = torch.utils.data.Subset(test_ds, range(min(a.eval_blocks, len(test_ds))))

    model = load_model(a.model_id, "auto" if a.device == "cuda" else None)
    if a.device != "cuda":
        model = model.to(device)
    zs = compute_bpb(model, test_subset, avg_bpt, device, a.eval_batch_size)
    print(f"[result] zero-shot BPB = {zs:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    result = {
        "model_id": a.model_id,
        "dataset": str(a.dataset),
        "corpus": Path(a.dataset).stem,
        "avg_bytes_per_token": avg_bpt,
        "n_train_blocks": len(train_ds),
        "n_test_blocks_scored": len(test_subset),
        "max_seq_len": a.max_seq_len,
        "mask_injected_special_tokens": True,
        "zero_shot_bpb": zs,
    }

    if not a.zero_shot_only:
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import DataLoader

        # PEFT can infer target_modules for some architectures and not others: it worked
        # for Qwen-2.5 and raised "Please specify `target_modules`" for LFM2.5, Llama-3.2,
        # Qwen-3.5, Gemma-4 and Ministral. "all-linear" is architecture-agnostic and, more
        # importantly, gives every model the *same* adapter scope -- which is required for
        # the cross-model comparison to mean anything.
        lc = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                        bias="none", task_type="CAUSAL_LM",
                        target_modules=a.target_modules)
        model = get_peft_model(model, lc)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[info] LoRA target={a.target_modules}: {n_train:,} trainable of "
              f"{n_all:,} ({100*n_train/n_all:.3f}%)", flush=True)
        result["trainable_params"] = n_train
        result["total_params"] = n_all
        result["trainable_pct"] = 100.0 * n_train / n_all
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=a.learning_rate)
        total = a.max_train_steps
        warm = max(1, int(total * a.warmup_ratio))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm
            else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

        dl = DataLoader(train_ds, batch_size=a.train_batch_size, shuffle=True)
        step = 0
        opt.zero_grad(set_to_none=True)
        done = False
        while not done:
            for micro, batch in enumerate(dl):
                out = model(input_ids=batch["input_ids"].to(device),
                            labels=batch["labels"].to(device))
                (out.loss / a.grad_accum).backward()
                if (micro + 1) % a.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 25 == 0:
                        print(f"  step {step}/{total} loss={out.loss.item():.4f}", flush=True)
                    if step >= total:
                        done = True; break
            else:
                continue
        model.eval()
        ad = compute_bpb(model, test_subset, avg_bpt, device, a.eval_batch_size)
        red = 100.0 * (zs - ad) / zs
        print(f"[result] adapted BPB = {ad:.4f}   reduction = {red:.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        result.update({"adapted_bpb": ad, "reduction_pct": red,
                       "train_steps": step,
                       "lora": {"r": a.lora_r, "alpha": a.lora_alpha,
                                "dropout": a.lora_dropout, "lr": a.learning_rate,
                                "effective_batch": a.train_batch_size * a.grad_accum}})

    result["wall_seconds"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

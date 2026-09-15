#!/usr/bin/env python3
"""Zero-shot BPB as a function of how much of each block's cold start is discarded.

This is the control the mechanism argument needs. Section "Why it works" claims that block-averaged
zero-shot BPB mis-ranks because it is largely scoring cold starts, and that a short equal adaptation
supplies the missing familiarity. The competing explanation is cheaper: the in-domain criterion
supplies 1,200 characters of context, so it scores the *warmed-up* regime, and adapted BPB is warm
too, while block-averaged zero-shot BPB is cold -- so the agreement could be two warm measurements
meeting rather than adaptation recovering capability.

That has a decisive test which needs no adaptation at all. Score each model zero-shot but discard
the first K token positions of every 512-token block from the loss, and re-run the selector
comparison. If late-position zero-shot BPB reaches adapted BPB's selection accuracy, adaptation is
unnecessary and the finding is "do not block-average cold starts". If it does not, the mechanism
claim survives a test that could have killed it.

One forward pass per model accumulates the loss summed at every position independently, so every
threshold K is derivable afterwards for free.

Writes one JSON per model. Run under slurm/late_position.sbatch; never on a login node.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lm_adapt_bench.config import DataConfig  # noqa: E402
from lm_adapt_bench.data import DataModule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--dataset", required=True, help="the news corpus jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--output-dir", required=True,
                    help="where tokenizer caches live; must NOT be under $HOME (quota)")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-blocks", type=int, default=2500,
                    help="test blocks to score; the published cells scored 2500")
    ap.add_argument("--device-map", default="cuda:0",
                    help="'auto' shards across visible GPUs; needed for the 35B MoE, which does "
                         "not fit one 64 GB A100 in bf16")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("refusing to run on CPU; this belongs in an sbatch job")

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    cfg = DataConfig(dataset_path=args.dataset, max_seq_len=args.max_seq_len)
    dm = DataModule(cfg)
    _, _, test = dm.tokenize(args.model_id, tok, args.max_seq_len, args.output_dir)
    # Same denominator the published cells used, computed the same way.
    bytes_per_token = dm.calculate_avg_bytes_per_token(tok)

    # Ministral-3 ships a Mistral3Config, which AutoModelForCausalLM refuses; the repo already
    # solves this in lm_adapt_bench.utils, so load through the same path the published cells used
    # rather than duplicating the special case here.
    if "mistral3" in args.model_id.lower() or "ministral" in args.model_id.lower():
        import importlib
        module = importlib.import_module("transformers.models.mistral3")
        model_class = getattr(module, "Mistral3" + "ForConditional" + "Generation")
        model = model_class.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, device_map=args.device_map,
            local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, device_map=args.device_map,
            trust_remote_code=True)
    model.eval()

    n_pos = args.max_seq_len - 1  # causal shift drops one
    acc_dev = next(model.parameters()).device
    loss_sum = torch.zeros(n_pos, dtype=torch.float64, device=acc_dev)
    tok_count = torch.zeros(n_pos, dtype=torch.float64, device=acc_dev)

    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= args.max_blocks:
                break
            in_dev = next(model.parameters()).device
            input_ids = batch["input_ids"].to(in_dev)
            labels = batch["labels"].to(in_dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids).logits
            # Per-token loss, no reduction, so it can be bucketed by position.
            shift_logits = logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            per_tok = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1), reduction="none", ignore_index=-100
            ).view(shift_labels.shape)
            valid = (shift_labels != -100).double()
            loss_sum += (per_tok.double() * valid).sum(dim=0).to(acc_dev)
            tok_count += valid.sum(dim=0).to(acc_dev)
            seen += input_ids.size(0)

    ls = loss_sum.cpu().tolist()
    tc = tok_count.cpu().tolist()

    def bpb_from(k: int) -> float | None:
        """Mean nats/token over positions >= k, converted to bits per byte."""
        s = sum(ls[k:])
        n = sum(tc[k:])
        if n == 0 or not bytes_per_token:
            return None
        return (s / n / math.log(2)) / bytes_per_token

    thresholds = [0, 8, 16, 32, 64, 128, 256]
    out = {
        "schema_version": 1,
        "model_id": args.model_id,
        "dataset": args.dataset,
        "max_seq_len": args.max_seq_len,
        "blocks_scored": seen,
        "avg_bytes_per_token": bytes_per_token,
        "loss_sum_by_position": ls,
        "token_count_by_position": tc,
        "bpb_from_position": {str(k): bpb_from(k) for k in thresholds},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.out, "| blocks", seen,
          "| bpb@0", out["bpb_from_position"]["0"], "| bpb@128", out["bpb_from_position"]["128"])


if __name__ == "__main__":
    main()

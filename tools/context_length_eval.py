"""Measure whether a 512-token LoRA adapter transfers to longer inference contexts.

For every requested context length, score the same fixed target windows from one contiguous
held-out token stream.  Only target tokens contribute to loss; preceding tokens provide
context.  The output reports exact decoded-byte BPB for the base and adapted model.

This is GPU inference.  Run through ``slurm/context_length_eval.sbatch``.
"""
import argparse
import json
import math
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--contexts", default="16,32,64,128,256,512,1024,2048")
    p.add_argument("--target-tokens", type=int, default=64)
    p.add_argument("--windows", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def load_stream(filename, split):
    data = torch.load(filename, weights_only=False)
    if "input_ids" in data and torch.is_tensor(data["input_ids"]):
        return data["input_ids"].to(torch.long).reshape(-1)
    return torch.cat([row["input_ids"].to(torch.long) for row in data[split]])


def load_base(model_id):
    from transformers import AutoModelForCausalLM
    common = dict(device_map="auto", trust_remote_code=True)
    for dtype_kw in ("dtype", "torch_dtype"):
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id, **{dtype_kw: torch.bfloat16}, **common)
        except TypeError:
            if dtype_kw == "torch_dtype":
                raise


@torch.no_grad()
def score(model, stream, anchors, contexts, target_tokens, batch_size, boundary_id, tok):
    model.eval()
    device = torch.device("cuda")
    results = {}
    for context in contexts:
        total_nats = 0.0
        total_tokens = 0
        total_bytes = 0
        seqs = [stream[a - context:a + target_tokens] for a in anchors]
        for start in range(0, len(seqs), batch_size):
            ids = torch.stack(seqs[start:start + batch_size]).to(device)
            labels = ids.clone()
            labels[:, :context] = -100
            if boundary_id is not None:
                labels[labels == boundary_id] = -100
            outputs = model(input_ids=ids, labels=labels)
            shifted = labels[:, 1:]
            n = int((shifted != -100).sum())
            total_nats += float(outputs.loss) * n
            total_tokens += n
        for a in anchors:
            target = stream[a:a + target_tokens]
            total_bytes += len(tok.decode(target.tolist(), skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False).encode("utf-8"))
        results[str(context)] = {
            "nats_per_token": total_nats / total_tokens,
            "bits_per_byte": total_nats / math.log(2) / total_bytes,
            "scored_tokens": total_tokens,
            "decoded_bytes": total_bytes,
        }
        print(f"  context={context:4d}  BPB={results[str(context)]['bits_per_byte']:.4f}",
              flush=True)
    return results


def main():
    args = parse_args()
    from peft import PeftModel
    from transformers import AutoTokenizer

    contexts = sorted({int(x) for x in args.contexts.split(",")})
    stream = load_stream(args.cache, args.split)
    max_context = max(contexts)
    last_anchor = len(stream) - args.target_tokens
    if last_anchor <= max_context:
        raise SystemExit("token stream is too short for requested contexts")
    anchors = torch.linspace(max_context, last_anchor, steps=args.windows).round().long().unique().tolist()

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    with_special = tok("hello world", add_special_tokens=True)["input_ids"]
    without_special = tok("hello world", add_special_tokens=False)["input_ids"]
    boundary_id = with_special[0] if len(with_special) > len(without_special) else None
    print(f"[info] {args.model_id}: {len(anchors)} fixed target windows; boundary={boundary_id}",
          flush=True)

    print("[info] base model", flush=True)
    base = load_base(args.model_id)
    base_results = score(base, stream, anchors, contexts, args.target_tokens,
                         args.batch_size, boundary_id, tok)
    del base
    torch.cuda.empty_cache()

    print("[info] adapted model", flush=True)
    base = load_base(args.model_id)
    adapted = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    adapted_results = score(adapted, stream, anchors, contexts, args.target_tokens,
                            args.batch_size, boundary_id, tok)
    del adapted, base
    torch.cuda.empty_cache()

    rows = {}
    for context in contexts:
        key = str(context)
        b, a = base_results[key], adapted_results[key]
        rows[key] = {
            "base_bpb": b["bits_per_byte"],
            "adapted_bpb": a["bits_per_byte"],
            "gain_bpb": b["bits_per_byte"] - a["bits_per_byte"],
            "relative_reduction_pct": 100 * (b["bits_per_byte"] - a["bits_per_byte"]) / b["bits_per_byte"],
            "base_nats_per_token": b["nats_per_token"],
            "adapted_nats_per_token": a["nats_per_token"],
            "scored_tokens": a["scored_tokens"],
            "decoded_bytes": a["decoded_bytes"],
        }
    result = {
        "model_id": args.model_id,
        "adapter": args.adapter,
        "training_context_tokens": 512,
        "target_tokens_per_window": args.target_tokens,
        "n_windows": len(anchors),
        "boundary_token_masked": boundary_id is not None,
        "by_context_length": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

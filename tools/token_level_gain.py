"""Where does the zero-shot -> adapted BPB gain actually come from?

Answers the question the team raised after the talk: what drives the large BPB
improvement on Gemma (and Liquid) after adaptation, and which tokens gain the most
probability?

The pipeline packs the corpus with `add_special_tokens=True` per document
(lm_adapt_bench/data.py:475) and then cuts fixed 512-token blocks. For tokenizers that
inject a BOS-like token, that token therefore lands *inside* the scored stream at every
document boundary -- a token a base model has essentially never seen mid-context. This
script measures how much of the gain sits on those boundary positions versus on ordinary
content tokens.

Outputs a JSON with three decompositions:
  1. gain_by_class  -- share of the total per-token gain on the boundary token, in the
                       k-token post-boundary window, and on everything else.
  2. gain_by_offset -- mean per-token gain as a function of distance from a boundary.
  3. top_tokens     -- tokens ranked by total log-prob mass gained (and lost).

Run via slurm/token_gain.sbatch -- never on the login node.
"""
import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--adapter", required=True, help="path to the PEFT checkpoint dir")
    p.add_argument("--cache", required=True, help="tokenized-split cache .pt from results/cache")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-blocks", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tail", type=int, default=16,
                   help="post-boundary window length in tokens (legacy option name)")
    p.add_argument("--top-k", type=int, default=60)
    return p.parse_args()


@torch.no_grad()
def token_logprobs(model, blocks, batch_size, device):
    """Per-position log-prob of the realised next token. Returns a list of 1-D tensors
    of length (L-1), one per block, aligned so element j scores the token at index j+1."""
    out = []
    model.eval()
    for i in range(0, len(blocks), batch_size):
        chunk = blocks[i:i + batch_size]
        ids = torch.stack([b["input_ids"] for b in chunk]).to(device)
        logits = model(input_ids=ids).logits.float()
        logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        got = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        out.extend(got[j].cpu() for j in range(got.shape[0]))
        del logits, logprobs, got
    return out


def aggregate(blocks, lp_base, lp_adapt, boundary_id, tail, vocab):
    """Split the per-token gain by distance from a document boundary and by token id.

    `lp_*[i][j]` is the log-prob assigned to the token at `blocks[i]["input_ids"][j+1]`,
    so position j in the gain vector is *scored on* target token j+1. Offset 0 therefore
    means "the boundary token itself was the prediction target".
    """
    BIG = 10 ** 6
    gain_cls = defaultdict(float)
    count_cls = defaultdict(int)
    gain_off = torch.zeros(tail + 1, dtype=torch.float64)
    count_off = torch.zeros(tail + 1, dtype=torch.int64)
    # Blocks are cut every max_seq_len tokens irrespective of document boundaries, so most
    # blocks start mid-document with no BOS prefix. If a model relies on BOS for
    # calibration, its loss should be inflated at the START of every block and adaptation
    # should recover it -- a far larger surface than the boundary tokens themselves.
    nb = len(blocks[0]["input_ids"]) - 1
    gain_pos = torch.zeros(nb, dtype=torch.float64)
    base_pos = torch.zeros(nb, dtype=torch.float64)
    count_pos = torch.zeros(nb, dtype=torch.int64)
    tok_gain_t = torch.zeros(vocab, dtype=torch.float64)
    tok_count_t = torch.zeros(vocab, dtype=torch.int64)
    tot_base = tot_adapt = 0.0
    n_pos = 0

    for blk, lb, la in zip(blocks, lp_base, lp_adapt):
        tgt = blk["input_ids"][1:].to(torch.long)   # token predicted at each position
        d = (la - lb).to(torch.float64)             # positive = adapted assigns more prob
        tot_base += float(-lb.sum())
        tot_adapt += float(-la.sum())
        n_pos += d.numel()

        dist = torch.full((d.numel(),), BIG, dtype=torch.long)
        if boundary_id is not None:
            for bi in torch.nonzero(tgt == boundary_id).flatten().tolist():
                hi = min(bi + tail + 1, d.numel())
                rng = torch.arange(bi, hi)
                dist[rng] = torch.minimum(dist[rng], rng - bi)

        for name, m in (("boundary_token", dist == 0),
                        ("recovery_tail", (dist >= 1) & (dist <= tail)),
                        ("content", dist > tail)):
            k = int(m.sum())
            if k:
                gain_cls[name] += float(d[m].sum())
                count_cls[name] += k

        near = dist <= tail
        if bool(near.any()):
            gain_off.scatter_add_(0, dist[near], d[near])
            count_off.scatter_add_(0, dist[near],
                                   torch.ones(int(near.sum()), dtype=torch.int64))

        tok_gain_t.scatter_add_(0, tgt, d)
        tok_count_t.scatter_add_(0, tgt, torch.ones(tgt.numel(), dtype=torch.int64))

        if d.numel() == nb:
            gain_pos += d
            base_pos += (-lb).to(torch.float64)
            count_pos += 1

    # bucket absolute position geometrically -- the effect, if any, is concentrated early
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, nb]
    by_block_pos = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo >= nb:
            break
        hi = min(hi, nb)
        n = int(count_pos[lo:hi].sum())
        if n:
            by_block_pos["%d-%d" % (lo, hi - 1)] = {
                "mean_gain": float(gain_pos[lo:hi].sum()) / n,
                "mean_base_loss": float(base_pos[lo:hi].sum()) / n,
                "positions": n,
            }

    seen = torch.nonzero(tok_count_t).flatten()
    return {
        "by_block_pos": by_block_pos,
        "gain_cls": dict(gain_cls), "count_cls": dict(count_cls),
        "gain_off": {k: float(gain_off[k]) for k in range(tail + 1) if int(count_off[k]) > 0},
        "count_off": {k: int(count_off[k]) for k in range(tail + 1) if int(count_off[k]) > 0},
        "tok_gain": {int(t): float(tok_gain_t[t]) for t in seen},
        "tok_count": {int(t): int(tok_count_t[t]) for t in seen},
        "tot_base": tot_base, "tot_adapt": tot_adapt, "n_pos": n_pos,
    }


def load_base(model_id, device_map):
    """transformers renamed torch_dtype -> dtype in v5; accept either."""
    from transformers import AutoModelForCausalLM
    common = dict(device_map=device_map, trust_remote_code=True)
    for dtype_kw in ("dtype", "torch_dtype"):
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id, **{dtype_kw: torch.bfloat16}, **common)
        except TypeError as e:
            if dtype_kw == "torch_dtype":
                raise
            print(f"[warn] {dtype_kw}= rejected ({e}); retrying", flush=True)


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Which id, if any, does this tokenizer inject at a document start?
    probe_with = tok("hello world", add_special_tokens=True)["input_ids"]
    probe_without = tok("hello world", add_special_tokens=False)["input_ids"]
    n_injected = len(probe_with) - len(probe_without)
    boundary_id = probe_with[0] if n_injected > 0 else None
    print(f"[info] {args.model_id}: injects {n_injected} special token(s); "
          f"boundary_id={boundary_id} "
          f"({repr(tok.decode([boundary_id])) if boundary_id is not None else '--'})",
          flush=True)

    data = torch.load(args.cache, weights_only=False)
    if "input_ids" in data and torch.is_tensor(data["input_ids"]):
        # compact form written by tools/extract_test_blocks.py
        ids = data["input_ids"][:args.max_blocks].to(torch.long)
        blocks = [{"input_ids": ids[i]} for i in range(ids.shape[0])]
    else:
        # raw lm_adapt_bench cache: {"train"|"val"|"test": [{"input_ids", "labels"}, ...]}
        blocks = data[args.split][:args.max_blocks]
    print(f"[info] scoring {len(blocks)} blocks x {len(blocks[0]['input_ids'])} tokens", flush=True)

    device_map = "auto"
    device = torch.device("cuda")

    print("[info] pass 1/2: base model", flush=True)
    base = load_base(args.model_id, device_map)
    lp_base = token_logprobs(base, blocks, args.batch_size, device)
    del base
    torch.cuda.empty_cache()

    print("[info] pass 2/2: adapted model", flush=True)
    base = load_base(args.model_id, device_map)
    adapted = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    lp_adapt = token_logprobs(adapted, blocks, args.batch_size, device)
    del adapted, base
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- aggregate
    tail = args.tail
    vocab = max(int(tok.vocab_size), len(tok)) + 8
    agg = aggregate(blocks, lp_base, lp_adapt, boundary_id, tail, vocab)
    gain_cls, count_cls = agg["gain_cls"], agg["count_cls"]
    gain_off, count_off = agg["gain_off"], agg["count_off"]
    tok_gain, tok_count = agg["tok_gain"], agg["tok_count"]
    tot_base, tot_adapt, n_pos = agg["tot_base"], agg["tot_adapt"], agg["n_pos"]
    total_gain = sum(gain_cls.values())
    nats_base = tot_base / n_pos
    nats_adapt = tot_adapt / n_pos
    # avg_bytes_per_token implied by the reported zero-shot BPB is applied by the caller;
    # here we report the tokenizer-independent quantities plus the ratio.
    result = {
        "model_id": args.model_id,
        "adapter": args.adapter,
        "split": args.split,
        "n_blocks": len(blocks),
        "n_scored_positions": n_pos,
        "injects_special_tokens": n_injected,
        "boundary_id": boundary_id,
        "boundary_str": tok.decode([boundary_id]) if boundary_id is not None else None,
        "mean_nats_per_token_base": nats_base,
        "mean_nats_per_token_adapted": nats_adapt,
        "relative_nats_reduction_pct": 100.0 * (nats_base - nats_adapt) / nats_base,
        "bits_per_token_base": nats_base / math.log(2),
        "bits_per_token_adapted": nats_adapt / math.log(2),
        "total_gain_nats": total_gain,
        "gain_by_class": {
            k: {"nats": gain_cls[k], "positions": count_cls[k],
                "share_of_total_gain_pct": (100.0 * gain_cls[k] / total_gain) if total_gain else None,
                "share_of_positions_pct": 100.0 * count_cls[k] / n_pos,
                "mean_gain_per_position": gain_cls[k] / max(count_cls[k], 1)}
            for k in gain_cls},
        "gain_by_offset": {
            str(k): {"mean_gain": gain_off[k] / max(count_off[k], 1), "positions": count_off[k]}
            for k in sorted(gain_off)},
        "gain_by_block_position": agg["by_block_pos"],
    }

    ranked = sorted(tok_gain.items(), key=lambda kv: -kv[1])
    def describe(items):
        return [{"token_id": t, "token": tok.decode([t]), "total_nats_gained": g,
                 "occurrences": tok_count[t], "mean_gain": g / max(tok_count[t], 1),
                 "share_of_total_gain_pct": (100.0 * g / total_gain) if total_gain else None}
                for t, g in items]
    result["top_tokens_gained"] = describe(ranked[:args.top_k])
    result["top_tokens_lost"] = describe(ranked[-args.top_k:][::-1])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    # ---------------------------------------------------------------- console summary
    print("\n=== %s ===" % args.model_id)
    print("mean nats/token  base=%.4f  adapted=%.4f  reduction=%.2f%%"
          % (nats_base, nats_adapt, result["relative_nats_reduction_pct"]))
    print("\nshare of the total gain, by position class:")
    for k in ("boundary_token", "recovery_tail", "content"):
        if k in result["gain_by_class"]:
            c = result["gain_by_class"][k]
            print("  %-15s %6.2f%% of gain   from %6.3f%% of positions   (mean %+.4f nats)"
                  % (k, c["share_of_total_gain_pct"], c["share_of_positions_pct"],
                     c["mean_gain_per_position"]))
    print("\nmean gain by ABSOLUTE position in the 512-token block:")
    for k, v in result["gain_by_block_position"].items():
        print("  pos %-9s base_loss %6.3f -> gain %+.4f nats  (n=%d)"
              % (k, v["mean_base_loss"], v["mean_gain"], v["positions"]))
    print("\nmean gain by distance from a document boundary:")
    for k in sorted(gain_off)[:min(9, len(gain_off))]:
        print("  offset %-3d %+.4f nats  (n=%d)" % (k, gain_off[k] / max(count_off[k], 1), count_off[k]))
    print("\ntop 15 tokens by probability mass gained:")
    for r in result["top_tokens_gained"][:15]:
        print("  %-22s %8.1f nats  n=%-7d mean %+.4f  (%.2f%% of gain)"
              % (repr(r["token"])[:22], r["total_nats_gained"], r["occurrences"],
                 r["mean_gain"], r["share_of_total_gain_pct"]))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()

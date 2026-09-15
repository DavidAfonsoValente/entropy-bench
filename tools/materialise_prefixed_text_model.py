#!/usr/bin/env python
"""Write a plain causal-LM copy of a checkpoint whose text weights sit under a prefix.

Why this exists
---------------
Ministral-3-14B-Base-2512 is a vision-language checkpoint. Its tensors are named
``language_model.model.*`` and ``language_model.lm_head.weight``, while transformers 5.11's
``Mistral3ForConditionalGeneration`` expects ``model.language_model.*`` plus an ``lm_head``
it constructs itself. The key spaces are disjoint, so loading it through the multimodal
class silently produces a RANDOMLY INITIALISED model: it scored 0.0000 on gsm8k and 0.2848
on hellaswag (chance 0.25) while passing every gate in the E9 pipeline.

Stripping the prefix yields exactly the layout of the class the nested ``text_config``
designates (``Ministral3ForCausalLM``). This writes that corrected checkpoint once, so the
evaluation itself runs the same unmodified code path as the other sixteen models rather
than carrying a bespoke loader.

The transformation is verified, not assumed: the stripped key set must match the target
architecture's own ``state_dict`` keys **exactly** -- same names, same shapes, nothing
missing and nothing extra. That check fails before any GPU time is spent if the layout
hypothesis is wrong.

    python tools/materialise_prefixed_text_model.py <src-model-id-or-path> <dest-dir> \
        [--prefix language_model]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Copied verbatim so the derived checkpoint is self-contained and tokenises identically.
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json",
    "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
)
SHARD_BYTES = 4 * 1024**3


def resolve(src):
    p = Path(src)
    if p.is_dir():
        return p
    import glob, os
    hits = glob.glob(f"{os.environ['HF_HOME']}/hub/models--{src.replace('/', '--')}/snapshots/*")
    if not hits:
        sys.exit(f"FATAL: {src} is neither a directory nor a cached model")
    return Path(sorted(hits)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dest")
    ap.add_argument("--prefix", default="language_model")
    a = ap.parse_args()

    src, dest = resolve(a.src), Path(a.dest)
    prefix = a.prefix.rstrip(".") + "."
    print(f"source : {src}\ndest   : {dest}\nprefix : {prefix}")

    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is None:
        sys.exit("FATAL: source config has no text_config -- nothing to extract")

    # Build the target architecture on the meta device: we only need its key set and shapes,
    # so this allocates no real memory.
    with torch.device("meta"):
        target = AutoModelForCausalLM.from_config(text_cfg)
    want = {k: tuple(v.shape) for k, v in target.state_dict().items()}
    print(f"target : {type(target).__name__}, {len(want)} tensors expected")

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        shards = sorted({src / v for v in json.loads(index_path.read_text())["weight_map"].values()})
    else:
        shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit("FATAL: no safetensors found")

    got, dropped = {}, 0
    for sh in shards:
        for k, v in load_file(str(sh)).items():
            if k.startswith(prefix):
                got[k[len(prefix):]] = v
            else:
                dropped += 1
    print(f"loaded : {len(got)} tensors under {prefix!r}, dropped {dropped} outside it")

    # The verification that makes this safe. Exact key AND shape agreement with the target
    # architecture -- the failure mode being guarded against is a silent partial match.
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    mismatched = sorted(k for k in set(want) & set(got) if tuple(got[k].shape) != want[k])
    if missing or extra or mismatched:
        print(f"FATAL: the stripped checkpoint does not match {type(target).__name__}", file=sys.stderr)
        for name, ks in (("missing", missing), ("unexpected", extra), ("wrong shape", mismatched)):
            if ks:
                print(f"  {name} ({len(ks)}): {ks[:6]}{' ...' if len(ks) > 6 else ''}", file=sys.stderr)
        # tie_word_embeddings would legitimately explain a missing lm_head; say so explicitly.
        if missing == ["lm_head.weight"] and getattr(text_cfg, "tie_word_embeddings", False):
            print("  note: lm_head is tied to the embeddings in this config", file=sys.stderr)
        return 1
    print(f"VERIFIED: stripped keys match {type(target).__name__} exactly "
          f"({len(want)} tensors, names and shapes)")

    dest.mkdir(parents=True, exist_ok=True)
    shard, size, n, weight_map, total = {}, 0, 0, {}, 0
    def flush():
        nonlocal shard, size, n
        if not shard:
            return
        n += 1
        name = f"model-{n:05d}.safetensors"
        save_file(shard, str(dest / name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        print(f"  wrote {name} ({size/1024**3:.1f} GiB, {len(shard)} tensors)")
        shard, size = {}, 0
    for k in sorted(got):
        v = got[k]
        if size and size + v.numel() * v.element_size() > SHARD_BYTES:
            flush()
        shard[k] = v
        size += v.numel() * v.element_size()
        total += v.numel() * v.element_size()
    flush()

    (dest / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    text_cfg.architectures = [type(target).__name__]
    text_cfg.save_pretrained(dest)
    try:
        AutoTokenizer.from_pretrained(src, trust_remote_code=True).save_pretrained(dest)
    except Exception:
        for f in TOKENIZER_FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dest / f)
    (dest / "DERIVED_FROM.json").write_text(json.dumps({
        "source": str(src), "prefix_stripped": prefix,
        "architecture": type(target).__name__, "tensors": len(want),
        "note": "Text tower extracted from a vision-language checkpoint. Key names and shapes "
                "were verified against the architecture's own state_dict before writing.",
    }, indent=2))
    print(f"\nwrote {n} shards, {total/1024**3:.1f} GiB -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Slice a small, portable test-split sample out of a big tokenized-split cache.

The caches in results/cache/*.pt are ~1.3GB each (train+val+test for one tokenizer).
The token-level gain analysis only needs a few thousand test blocks, which is ~16MB --
small enough to ship to a GPU elsewhere. Run this on CPU only.
"""
import argparse
import torch
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n-blocks", type=int, default=4000)
    a = p.parse_args()

    data = torch.load(a.cache, weights_only=False)
    blocks = data[a.split]
    n_total = len(blocks)
    take = blocks[:a.n_blocks]
    # re-pack as two dense tensors: far smaller and faster to load than a list of dicts
    ids = torch.stack([b["input_ids"] for b in take]).to(torch.int32)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": ids, "split": a.split,
                "n_blocks_taken": ids.shape[0], "n_blocks_available": n_total}, a.out)
    print("%s: %s split has %d blocks; wrote %d x %d -> %s (%.1f MB)"
          % (Path(a.cache).name, a.split, n_total, ids.shape[0], ids.shape[1], a.out,
             Path(a.out).stat().st_size / 1e6))


if __name__ == "__main__":
    main()

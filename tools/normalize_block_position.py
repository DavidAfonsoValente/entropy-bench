"""Add byte-normalized block-position metrics to token-gain JSON output.

The expensive model passes already store total nats and counts per position bucket.  This
small post-processing step decodes the matching held-out token blocks, measures UTF-8 bytes
in each bucket, and converts nats/token to bits/byte without rerunning a model.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    p.add_argument("--blocks", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    with open(a.result) as f:
        result = json.load(f)
    packed = torch.load(a.blocks, weights_only=False)
    if "input_ids" not in packed or not torch.is_tensor(packed["input_ids"]):
        raise SystemExit("expected compact block cache with an input_ids tensor")
    ids = packed["input_ids"][:result["n_blocks"], 1:].to(torch.long)
    tok = AutoTokenizer.from_pretrained(a.model_id, trust_remote_code=True)

    for label, row in result["gain_by_block_position"].items():
        lo, hi = (int(x) for x in label.split("-"))
        hi += 1
        total_bytes = sum(
            len(tok.decode(block[lo:hi].tolist(), skip_special_tokens=True,
                           clean_up_tokenization_spaces=False).encode("utf-8"))
            for block in ids
        )
        if total_bytes <= 0:
            raise SystemExit(f"no decoded bytes for bucket {label}")
        bytes_per_token = total_bytes / row["positions"]
        row["mean_bytes_per_token"] = bytes_per_token
        row["mean_base_bits_per_byte"] = row["mean_base_loss"] / math.log(2) / bytes_per_token
        row["mean_gain_bits_per_byte"] = row["mean_gain"] / math.log(2) / bytes_per_token

    result["byte_normalization"] = {
        "method": "decoded UTF-8 bytes within each absolute-position bucket",
        "special_tokens": "excluded from byte counts",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

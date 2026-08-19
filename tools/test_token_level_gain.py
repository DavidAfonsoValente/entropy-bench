"""Unit tests for the boundary/offset decomposition in token_level_gain.aggregate.

Run: venv_lm_adapt/bin/python -m pytest tools/test_token_level_gain.py -q
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_level_gain import aggregate  # noqa: E402

BOS = 2
TAIL = 3
VOCAB = 16


def _block(ids):
    t = torch.tensor(ids, dtype=torch.long)
    return {"input_ids": t, "labels": t}


def test_offsets_are_measured_from_the_boundary_target():
    # targets are ids[1:] = [5, BOS, 7, 8, 9, 10]
    ids = [4, 5, BOS, 7, 8, 9, 10]
    blk = _block(ids)
    n = len(ids) - 1
    lp_base = [torch.zeros(n)]
    # adapted gains exactly 1 nat at every position
    lp_adapt = [torch.ones(n)]

    a = aggregate([blk], lp_base, lp_adapt, BOS, TAIL, VOCAB)

    # target index 1 is the BOS -> offset 0; indices 2,3,4 are offsets 1,2,3; index 5 is content
    assert a["count_cls"]["boundary_token"] == 1
    assert a["count_cls"]["recovery_tail"] == TAIL
    assert a["count_cls"]["content"] == n - 1 - TAIL
    assert a["count_off"] == {0: 1, 1: 1, 2: 1, 3: 1}


def test_gain_sums_and_shares():
    ids = [4, 5, BOS, 7, 8, 9, 10]
    blk = _block(ids)
    n = len(ids) - 1
    lp_base = [torch.full((n,), -3.0)]
    la = torch.full((n,), -3.0)
    la[1] = -0.5          # big gain on the boundary token
    lp_adapt = [la]

    a = aggregate([blk], lp_base, lp_adapt, BOS, TAIL, VOCAB)
    total = sum(a["gain_cls"].values())
    assert abs(total - 2.5) < 1e-9
    assert abs(a["gain_cls"]["boundary_token"] - 2.5) < 1e-9
    assert abs(a["gain_cls"].get("recovery_tail", 0.0)) < 1e-9
    # nats bookkeeping: base is -sum(lp) over all positions
    assert abs(a["tot_base"] - 3.0 * n) < 1e-9
    assert a["n_pos"] == n


def test_no_boundary_token_puts_everything_in_content():
    ids = [4, 5, 6, 7, 8]
    blk = _block(ids)
    n = len(ids) - 1
    a = aggregate([blk], [torch.zeros(n)], [torch.ones(n)], None, TAIL, VOCAB)
    assert a["count_cls"]["content"] == n
    assert "boundary_token" not in a["count_cls"]
    assert a["count_off"] == {}


def test_token_level_attribution():
    # target token 7 appears twice and should accumulate both gains
    ids = [1, 7, 5, 7, 5]
    blk = _block(ids)
    n = len(ids) - 1                       # targets: 7, 5, 7, 5
    la = torch.tensor([1.0, 0.0, 2.0, 0.0])
    a = aggregate([blk], [torch.zeros(n)], [la], None, TAIL, VOCAB)
    assert a["tok_count"][7] == 2
    assert abs(a["tok_gain"][7] - 3.0) < 1e-9
    assert a["tok_count"][5] == 2
    assert abs(a["tok_gain"][5] - 0.0) < 1e-9


def test_overlapping_boundaries_take_the_nearest():
    # two BOS targets close together; offsets must be measured from the nearer one
    ids = [4, BOS, 6, BOS, 8, 9, 10, 11]
    blk = _block(ids)                      # targets: BOS,6,BOS,8,9,10,11
    n = len(ids) - 1
    a = aggregate([blk], [torch.zeros(n)], [torch.zeros(n)], BOS, TAIL, VOCAB)
    assert a["count_cls"]["boundary_token"] == 2
    # target idx1(BOS)=0, idx2=1, idx3(BOS)=0, idx4=1, idx5=2, idx6=3
    assert a["count_off"][0] == 2
    assert a["count_off"][1] == 2
    assert a["count_off"][2] == 1
    assert a["count_off"][3] == 1

"""compute_doc_bpb: every document token is scored exactly once, BOS is never scored, and the
denominator is the byte count of the text, whatever the tokenizer."""
import math

import pytest
import torch

from lm_adapt_bench.evaluate import compute_doc_bpb, doc_windows

VOCAB = 50


class CharTokenizer:
    """One token per character; optionally prepends a BOS like Gemma or Llama do."""
    pad_token_id = 0

    def __init__(self, bos):
        self.bos = bos

    def __call__(self, text, add_special_tokens=True):
        ids = [2 + (ord(c) % (VOCAB - 2)) for c in text]
        return {"input_ids": ([1] if self.bos and add_special_tokens else []) + ids}


class UniformModel(torch.nn.Module):
    """Equal logits everywhere, so each scored token costs exactly ln(VOCAB) nats."""

    def forward(self, input_ids, attention_mask=None):
        return type("Out", (), {"logits": torch.zeros(*input_ids.shape, VOCAB)})()


@pytest.mark.parametrize("context,stride", [(8, 4), (8, 8), (64, 32)])
def test_windows_score_each_token_once(context, stride):
    docs = [(list(range(n)), 1) for n in (1, 2, 7, 8, 9, 30)]
    scored = []
    for d, (ids, _) in enumerate(docs):
        for w, score_from in doc_windows([(ids, 1)], context, stride):
            assert len(w) <= context
            scored += [(d, t) for t in w[score_from:]]
    expected = [(d, t) for d, (ids, _) in enumerate(docs) for t in ids[1:]]
    assert sorted(scored) == sorted(expected)


@pytest.mark.parametrize("bos", [True, False])
def test_doc_bpb_uniform_model(bos):
    texts = ["short", "a much longer document than the context window " * 3, "é ü"]
    r = compute_doc_bpb(UniformModel(), CharTokenizer(bos), texts, torch.device("cpu"),
                        context=16, stride=8, token_budget=64)
    n_chars = sum(len(t) for t in texts)
    # With a BOS every character is a target; without one the first character of each
    # document has nothing to condition on and is not scored.
    assert r["tokens"] == (n_chars if bos else n_chars - len(texts))
    assert r["bytes"] == sum(len(t.encode("utf-8")) for t in texts)
    assert r["bpb"] == pytest.approx(r["tokens"] * math.log(VOCAB) / math.log(2) / r["bytes"])

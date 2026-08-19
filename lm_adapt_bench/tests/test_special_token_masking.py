"""The injected per-document BOS must stay in input_ids but be masked out of the labels.

Regression guard for the packing artefact documented in docs/TOKEN_GAIN_FINDINGS.md.
"""
import torch

from lm_adapt_bench.config import DataConfig
from lm_adapt_bench.data import DataModule

BOS = 7


class FakeTokenizer:
    """Prepends BOS when add_special_tokens=True, like Gemma/Mistral/Llama-3/LFM."""

    def __init__(self, inject=True):
        self.inject = inject

    def __call__(self, text, add_special_tokens=True):
        body = [100 + (ord(c) % 5) for c in text]
        if add_special_tokens and self.inject:
            return {"input_ids": [BOS] + body}
        return {"input_ids": body}


def _manager(tmp_path, mask):
    ds = tmp_path / "x.jsonl"
    if not ds.exists():
        ds.write_text('{"text": "aaaaaaaaaaaa"}\n{"text": "bbbbbbbbbbbb"}\n')
    cfg = DataConfig(dataset_path=str(ds), mask_injected_special_tokens=mask)
    dm = DataModule(cfg)
    dm.train_texts = ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    dm.val_texts = ["cccccccccccc"]
    dm.test_texts = ["dddddddddddd"]
    return dm


def test_injected_token_is_context_but_not_a_target(tmp_path):
    dm = _manager(tmp_path, mask=True)
    train, _, _ = dm.tokenize("fake/model", FakeTokenizer(), 8, str(tmp_path))
    ids = torch.cat([b["input_ids"] for b in train])
    labels = torch.cat([b["labels"] for b in train])
    assert (ids == BOS).sum() > 0, "BOS should remain in the input stream"
    # every BOS position must be masked in the labels
    assert torch.all(labels[ids == BOS] == -100)
    # non-BOS positions are untouched
    assert torch.all(labels[ids != BOS] == ids[ids != BOS])


def test_flag_off_reproduces_old_behaviour(tmp_path):
    dm = _manager(tmp_path, mask=False)
    train, _, _ = dm.tokenize("fake/model", FakeTokenizer(), 8, str(tmp_path))
    ids = torch.cat([b["input_ids"] for b in train])
    labels = torch.cat([b["labels"] for b in train])
    assert torch.equal(ids, labels), "unmasked packing should score every token"


def test_tokenizer_without_special_tokens_is_unaffected(tmp_path):
    dm = _manager(tmp_path, mask=True)
    train, _, _ = dm.tokenize("fake/model-noinject", FakeTokenizer(inject=False), 8,
                              str(tmp_path))
    ids = torch.cat([b["input_ids"] for b in train])
    labels = torch.cat([b["labels"] for b in train])
    assert torch.equal(ids, labels)
    assert (labels == -100).sum() == 0


def test_cache_key_separates_masked_and_unmasked(tmp_path):
    a = _manager(tmp_path, mask=True)._cache_path("m", 8, str(tmp_path))
    b = _manager(tmp_path, mask=False)._cache_path("m", 8, str(tmp_path))
    assert a != b, "a stale cache must not leak the other behaviour"

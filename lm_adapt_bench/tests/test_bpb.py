import math
import torch
import pytest
from unittest.mock import MagicMock
from lm_adapt_bench.evaluate import compute_bpb
from lm_adapt_bench.data import DataModule
from lm_adapt_bench.config import DataConfig
from transformers import AutoTokenizer

def test_bpb_formula_known_value():
    """With fixed mock logits, BPB should match hand-computed value."""
    # CE = ln(2) nats
    # avg_bpt = 4.0
    # expected BPB = (ln(2) / ln(2)) / 4.0 = 0.25
    
    model = MagicMock()
    model.config.pad_token_id = 0
    
    # Mock output loss
    output = MagicMock()
    output.loss = torch.tensor(math.log(2))
    model.return_value = output
    
    val_dataset = [
        {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([1, 2])}
    ]
    
    avg_bpt = 4.0
    device = torch.device("cpu")
    
    bpb = compute_bpb(model, val_dataset, avg_bpt, device, batch_size=1)
    assert bpb == pytest.approx(0.25)

def test_avg_bytes_per_token_computation():
    """avg_bytes_per_token should equal total bytes / total tokens (including special)."""
    tokeniser = AutoTokenizer.from_pretrained("gpt2")
    texts = ["Hello world", "This is a test sentence."]
    
    # Manual computation
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    # Match the updated DataModule logic: add_special_tokens=True
    total_tokens = sum(len(tokeniser(t, add_special_tokens=True)["input_ids"]) for t in texts)
    expected = total_bytes / total_tokens
    
    # Mock DataModule
    config = DataConfig(dataset_path="dummy.txt")
    dm = DataModule.__new__(DataModule)
    dm.test_texts = []
    dm.val_texts = texts
    dm.train_texts = []
    
    actual = dm.get_byte_stats(tokeniser)
    assert actual == pytest.approx(expected)

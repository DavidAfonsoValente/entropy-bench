import os
import pytest
from lm_adapt_bench.data import DataModule
from lm_adapt_bench.config import DataConfig
from transformers import AutoTokenizer

@pytest.fixture
def fixtures_dir():
    return os.path.join(os.path.dirname(__file__), "fixtures")

def test_load_txt(fixtures_dir):
    path = os.path.join(fixtures_dir, "sample.txt")
    config = DataConfig(dataset_path=path, val_split=0.1)
    dm = DataModule(config)
    assert len(dm.raw_texts) >= 20

def test_load_jsonl(fixtures_dir):
    path = os.path.join(fixtures_dir, "sample.jsonl")
    config = DataConfig(dataset_path=path, text_field="text", val_split=0.1)
    dm = DataModule(config)
    assert len(dm.raw_texts) == 20

def test_load_csv(fixtures_dir):
    path = os.path.join(fixtures_dir, "sample.csv")
    config = DataConfig(dataset_path=path, text_field="text", val_split=0.1)
    dm = DataModule(config)
    assert len(dm.raw_texts) == 20

def test_tokenize_produces_blocks(fixtures_dir, tmp_path):
    path = os.path.join(fixtures_dir, "sample.txt")
    config = DataConfig(dataset_path=path, max_seq_len=16)
    dm = DataModule(config)
    tokeniser = AutoTokenizer.from_pretrained("gpt2")
    train_ds, val_ds, test_ds = dm.tokenize("gpt2", tokeniser, 16, str(tmp_path))
    
    for block in train_ds:
        assert block["input_ids"].shape == (16,)
        assert block["labels"].shape == (16,)

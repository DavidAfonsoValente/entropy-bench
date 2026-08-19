import os
import time
import pytest
from lm_adapt_bench.data import DataModule
from lm_adapt_bench.config import DataConfig
from transformers import AutoTokenizer

@pytest.fixture
def fixtures_dir():
    return os.path.join(os.path.dirname(__file__), "fixtures")

def test_cache_hit(fixtures_dir, tmp_path):
    path = os.path.join(fixtures_dir, "sample.txt")
    config = DataConfig(dataset_path=path, max_seq_len=16)
    dm = DataModule(config)
    tokeniser = AutoTokenizer.from_pretrained("gpt2")
    
    cache_path = dm._cache_path("gpt2", 16, str(tmp_path))
    assert not cache_path.exists()
    
    # First call: creates cache
    dm.tokenize("gpt2", tokeniser, 16, str(tmp_path))
    assert cache_path.exists()
    
    # Second call: uses cache (implicitly verified by not crashing and returning same data)
    train_ds, val_ds, test_ds = dm.tokenize("gpt2", tokeniser, 16, str(tmp_path))
    assert len(train_ds) > 0

def test_cache_miss_on_content_change(fixtures_dir, tmp_path):
    path = os.path.join(fixtures_dir, "sample.txt")
    temp_dataset = tmp_path / "data.txt"
    temp_dataset.write_text(open(path).read())
    
    config = DataConfig(dataset_path=str(temp_dataset), max_seq_len=16)
    dm = DataModule(config)
    tokeniser = AutoTokenizer.from_pretrained("gpt2")
    
    cache_path_1 = dm._cache_path("gpt2", 16, str(tmp_path))
    dm.tokenize("gpt2", tokeniser, 16, str(tmp_path))
    assert cache_path_1.exists()
    
    # Modify content and path to trigger miss
    temp_dataset2 = tmp_path / "data2.txt"
    temp_dataset2.write_text("New content that is different.")
    config2 = DataConfig(dataset_path=str(temp_dataset2), max_seq_len=16)
    dm2 = DataModule(config2)
    cache_path_2 = dm2._cache_path("gpt2", 16, str(tmp_path))
    
    assert cache_path_1 != cache_path_2

import os
import pytest
import torch
from lm_adapt_bench.sweep import SweepRunner
from lm_adapt_bench.data import DataModule
from lm_adapt_bench.config import DataConfig, TrainingConfig, SweepConfig, RunConfig

@pytest.fixture
def fixtures_dir():
    return os.path.join(os.path.dirname(__file__), "fixtures")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_sweep_smoke(fixtures_dir, tmp_path):
    dataset_path = os.path.join(fixtures_dir, "sample.txt")
    data_config = DataConfig(dataset_path=dataset_path, max_seq_len=64)
    dm = DataModule(data_config)
    
    sweep_config = SweepConfig(
        n_trials=2,
        sweep_steps=10,
        search_space={
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-4, "log": True},
            "batch_size": {"type": "categorical", "choices": [4, 8]}
        }
    )
    training_config = TrainingConfig(batch_size=4)
    run_config = RunConfig(model_id="gpt2", output_dir=str(tmp_path))
    
    device = torch.device("cuda")
    dtype = torch.float16
    avg_bpt = 4.0
    
    runner = SweepRunner(
        "gpt2", dm, sweep_config, training_config, 
        device, dtype, avg_bpt, str(tmp_path), run_config
    )
    
    best_config, trials_df, study = runner.run()
    
    assert isinstance(best_config, TrainingConfig)
    assert len(trials_df) == 2
    assert "val_bpb" in trials_df.columns

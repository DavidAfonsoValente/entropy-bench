"""--hparams manual: the packaged recipe loads as the paper's configuration, and bad files fail loud."""
import subprocess
import sys

import pytest

from lm_adapt_bench.config import TrainingConfig, load_manual_training_config


def test_packaged_recipe_is_the_paper_configuration():
    cfg = load_manual_training_config(final_epochs=5)
    assert isinstance(cfg, TrainingConfig)
    assert (cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.learning_rate) == (16, 32, 0.05, 1e-4)
    assert cfg.batch_size * cfg.gradient_accumulation_steps == 32
    assert cfg.final_epochs == 5  # caller defaults survive when the file does not set them


def test_file_overrides_defaults_and_accepts_json(tmp_path):
    p = tmp_path / "h.json"
    p.write_text('{"learning_rate": 3e-4, "final_epochs": 2}')
    cfg = load_manual_training_config(str(p), final_epochs=5)
    assert cfg.learning_rate == 3e-4 and cfg.final_epochs == 2


def test_yaml_exponent_is_a_float(tmp_path):
    p = tmp_path / "h.yaml"
    p.write_text("learning_rate: 3e-4\nlora_r: '32'\n")
    cfg = load_manual_training_config(str(p))
    assert cfg.learning_rate == 3e-4 and isinstance(cfg.learning_rate, float) and cfg.lora_r == 32


def test_unknown_field_raises(tmp_path):
    p = tmp_path / "h.yaml"
    p.write_text("learning_rat: 1.0e-4\n")
    with pytest.raises(ValueError, match="learning_rat"):
        load_manual_training_config(str(p))


@pytest.mark.parametrize("extra", [["--hparams", "manual", "--phase", "sweep"],
                                   ["--hparams-file", "x.yaml"]])
def test_cli_rejects_incoherent_flags(extra):
    r = subprocess.run([sys.executable, "-m", "lm_adapt_bench.cli", "--dataset", "unused", *extra],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "--hparams" in r.stderr

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from .contamination.config import ContaminationConfig

@dataclass
class DataConfig:
    dataset_path: str
    text_field: str = "text"
    val_split: float = 0.1
    test_split: float = 0.1
    max_seq_len: int = 512
    max_samples: Optional[int] = None
    seed: int = 42
    hf_dataset_name: Optional[str] = None
    hf_dataset_config: Optional[str] = None
    hf_dataset_split: Optional[str] = "train"
    # Documents are packed with add_special_tokens=True and then cut into fixed blocks, so
    # tokenizers that inject a BOS put it *inside* the scored stream at every document
    # boundary. Base models assign such a token ~1e-8 probability mid-text (measured:
    # +18.9 nats for Gemma-4-12B, +11.5 for LFM2.5), and adaptation recovers it trivially,
    # inflating the reduction for BOS-injecting models only -- 3-9% of their gain on the
    # news corpus, and worse on short-document corpora. Masking it keeps the token as
    # context but stops scoring the model on predicting it, which also makes the scored
    # token population match the add_special_tokens=False denominator used for BPB.
    # Set False to reproduce the pre-fix numbers in the paper. See
    # docs/TOKEN_GAIN_FINDINGS.md.
    mask_injected_special_tokens: bool = True
    contamination: ContaminationConfig = field(default_factory=ContaminationConfig)

@dataclass
class TrainingConfig:
    # ... rest of TrainingConfig remains same
    learning_rate: float = 2e-4
    batch_size: int = 8
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    final_epochs: int = 3
    eval_batch_size: int = 16
    gradient_clip: float = 1.0
    # Wall-time budget (seconds) for the final adaptation loop. None = unlimited.
    # When reached, training stops gracefully and keeps the best checkpoint so far.
    max_train_seconds: Optional[float] = None
    # Resumable (chained) training LR schedule: cosine decay from peak LR down to
    # lr_floor_ratio*peak over `target_train_steps` CUMULATIVE grad-steps, then hold the floor.
    # Decaying by cumulative step (not per-link) anneals smoothly within a link and continues
    # across resumes without re-warmup spikes. Avoids the constant-LR oscillation near the optimum.
    target_train_steps: int = 8000
    lr_floor_ratio: float = 0.1

@dataclass
class SweepConfig:
    n_trials: int = 20
    sweep_steps: int = 200
    mode: str = "lora"           # "lora" or "full"
    lora_targets: Optional[List[str]] = None   # None = auto-detect
    search_space: Dict[str, Any] = field(default_factory=dict)
    # Wall-time budget (seconds) for the whole Optuna sweep. None = unlimited (n_trials only).
    # Optuna stops launching new trials once this is hit; the binding constraint for large
    # models, which can take >30 min/trial and otherwise blow the Slurm walltime in the sweep.
    max_sweep_seconds: Optional[float] = None

@dataclass
class RunConfig:
    model_id: str
    output_dir: str
    device: str = "auto"
    dtype: str = "auto"           # "auto","float32","float16","bfloat16"
    flash_attention: bool = False
    report_title: str = "LM Adapt Bench"
    hf_token: Optional[str] = None
    force_rerun: bool = False
    baseline_only: bool = False
    no_pdf: bool = False
    parallel_rank: int = 0
    total_ranks: int = 1

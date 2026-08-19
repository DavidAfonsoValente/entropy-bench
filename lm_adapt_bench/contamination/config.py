from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ContaminationConfig:
    check_contamination: bool = False
    check_level: str = "off"  # off, light, standard, strict, forensic
    forbidden_corpus: Optional[str] = None
    clean_reference_corpus: Optional[str] = None
    
    # Lexical parameters
    ngram_n: int = 13
    minhash_threshold: float = 0.8
    num_perm: int = 128
    normalize_lowercase: bool = True
    fail_on_split_leakage: bool = True
    
    # Min-K parameters
    sample_size: int = 1000
    test_full_max: int = 5000
    min_k_variant: str = "minkpp" # minkpp, mink
    min_k_percent: float = 20.0
    min_k_min_tokens: int = 50
    min_k_std_epsilon: float = 1e-6
    min_k_outlier_percentile: float = 95.0
    min_k_high_percentile: float = 99.0
    min_k_robust_z_suspicious: float = 2.5
    min_k_robust_z_high: float = 3.5
    min_k_score_direction: str = "higher_is_more_suspicious"
    min_k_severe_guardrail: float = -1.0
    min_k_enable_severe_guardrail: bool = True
    
    # CoDeC parameters
    codec_context_size: int = 4
    codec_samples: int = 500
    codec_delta_threshold: float = 0.0
    codec_robust_z_threshold: float = -2.0
    codec_suspicious_fraction_medium: float = 0.10
    codec_suspicious_fraction_high: float = 0.25
    codec_suspicious_fraction_severe: float = 0.40
    
    # DCQ parameters
    enable_dcq: bool = False
    dc_samples: int = 100
    dcq_num_distractors: int = 2
    dcq_min_margin: float = 0.0
    dcq_candidate_source: str = "min_k" # min_k, codec, random, combined
    
    # Additional
    enable_gds: bool = False
    enable_embedding_overlap: bool = False
    
    # Cleaning/Policy
    cleaning_policy: str = "report_only" # report_only, quarantine, drop
    cleaning_mode: str = "model_specific" # model_specific, global_unified
    export_cleaned_dataset: bool = True # Export Train/Val/Test splits to disk
    use_global_quarantine: bool = False # Import previous quarantine log if it exists
    output_dir: Optional[str] = None
    strict_eval_cleaning: bool = True
    include_in_report: bool = True
    seed: int = 42
    
    # Soft evidence quarantine caps
    max_soft_eval_quarantine_fraction: float = 0.10
    max_soft_val_quarantine_fraction: float = 0.15
    max_soft_train_quarantine_fraction: float = 0.30
    min_active_test_examples: int = 100
    min_active_test_fraction: float = 0.70
    
    # Risk thresholds (Legacy/Global)
    risk_high_threshold: float = 0.7
    risk_medium_threshold: float = 0.3
    codec_suspicious_threshold: float = 0.1
    dcq_accuracy_threshold: float = 0.5

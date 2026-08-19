from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class ContaminationFinding:
    finding_id: str
    record_id: str
    split: str
    detector: str
    severity: str  # low, medium, high, critical
    score: float
    matched_record_id: Optional[str] = None
    matched_source: Optional[str] = None
    reason: str = ""
    recommended_action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MinKSummary:
    variant: str = "minkpp"
    normalized_scoring: bool = True
    fallback_used: bool = False
    valid_n: int = 0
    skipped_too_short: int = 0
    scoring_failed: int = 0
    p95: float = 0.0
    p99: float = 0.0
    median: float = 0.0
    mad: float = 0.0
    calibration_source: str = "in_run"
    calibration_confidence: str = "INSUFFICIENT"
    outlier_fraction: float = 0.0
    suspicious_fraction: float = 0.0
    high_suspicious_fraction: float = 0.0
    severe_guardrail_fraction: float = 0.0
    score_direction: str = "higher_is_more_suspicious"
    notes: List[str] = field(default_factory=list)

@dataclass
class CodecSummary:
    num_samples: int = 0
    context_size: int = 0
    codec_delta_mean: float = 0.0
    codec_delta_median: float = 0.0
    codec_delta_std: float = 0.0
    suspicious_fraction: float = 0.0
    threshold_used: float = 0.0
    confidence_interval: Optional[List[float]] = None
    calibration_confidence: str = "INSUFFICIENT"
    risk_label: str = "LOW"
    notes: List[str] = field(default_factory=list)

@dataclass
class DCQSummary:
    num_candidates: int = 0
    num_scored: int = 0
    original_selected_count: int = 0
    dcq_accuracy: float = 0.0
    mean_margin: float = 0.0
    positive_fraction: float = 0.0
    candidate_source: str = ""
    risk_label: str = "LOW"
    notes: List[str] = field(default_factory=list)

@dataclass
class ModelContaminationScore:
    model_id: str
    split: str
    min_k_summary: Optional[MinKSummary] = None
    codec_summary: Optional[CodecSummary] = None
    dcq_summary: Optional[DCQSummary] = None
    detector_scores: Dict[str, Any] = field(default_factory=dict) # Keep for raw scores
    suspicious_fraction: float = 0.0
    risk_score: float = 0.0
    risk_label: str = "LOW" # LOW, MEDIUM, HIGH, CRITICAL, COMPROMISED
    sample_size: int = 0
    suspicious_indices: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

@dataclass
class DatasetContaminationSummary:
    original_rows: int = 0
    cleaned_rows: int = 0
    duplicates_removed: int = 0
    quarantined_rows: int = 0
    test_rows_dropped: int = 0
    split_leakage_count: int = 0
    forbidden_overlap_count: int = 0
    active_test_examples: int = 0
    active_test_fraction: float = 1.0
    test_split_compromised: bool = False
    risk_level: str = "LOW"
    risk_score: float = 0.0
    findings_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ContaminationAudit:
    run_id: str
    dataset_summary: DatasetContaminationSummary
    model_scores: Dict[str, List[ModelContaminationScore]] = field(default_factory=dict)
    findings: List[ContaminationFinding] = field(default_factory=list)

from typing import List, Dict, Any, Tuple
from .schema import ContaminationFinding, DatasetContaminationSummary
from .config import ContaminationConfig

class RiskAggregator:
    def __init__(self, config: ContaminationConfig):
        self.config = config

    def aggregate_dataset_risk(self, findings: List[ContaminationFinding], original_rows: int) -> Tuple[str, float]:
        """
        Aggregates findings into a global dataset risk level and score.
        """
        if not findings:
            return "LOW", 0.0
            
        risk_score = 0.0
        max_severity = "low"
        
        # Heuristics
        for f in findings:
            if f.severity == "critical":
                risk_score += 0.5
                max_severity = "critical"
            elif f.severity == "high":
                risk_score += 0.2
            elif f.severity == "medium":
                risk_score += 0.05
        
        # Normalized by fraction of findings (this is just one heuristic)
        # findings_fraction = len(findings) / max(1, original_rows)
        # risk_score += findings_fraction * 10.0
        
        risk_score = min(1.0, risk_score)
        
        if max_severity == "critical" or risk_score > self.config.risk_high_threshold:
            return "CRITICAL" if max_severity == "critical" else "HIGH", risk_score
        if risk_score > self.config.risk_medium_threshold:
            return "MEDIUM", risk_score
            
        return "LOW", risk_score

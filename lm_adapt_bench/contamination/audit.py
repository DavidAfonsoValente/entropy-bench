import json
import os
import dataclasses
from typing import Any, List, Dict
from .schema import ContaminationAudit

class AuditWriter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_audit(self, audit: ContaminationAudit):
        # Write summary
        summary_path = os.path.join(self.output_dir, "contamination_summary.json")
        with open(summary_path, "w") as f:
            json.dump(dataclasses.asdict(audit.dataset_summary), f, indent=2)

        # Write findings (audit log)
        audit_path = os.path.join(self.output_dir, "contamination_audit.jsonl")
        with open(audit_path, "w") as f:
            for finding in audit.findings:
                f.write(json.dumps(dataclasses.asdict(finding)) + "\n")

        # Write model scores
        scores_path = os.path.join(self.output_dir, "per_model_contamination_scores.json")
        with open(scores_path, "w") as f:
            # We need to handle non-serializable objects in detector_scores if any
            # (though we mostly used floats and lists)
            serializable_scores = {}
            for model_id, scores in audit.model_scores.items():
                serializable_scores[model_id] = [dataclasses.asdict(s) for s in scores]
            json.dump(serializable_scores, f, indent=2)

    def write_quarantine(self, quarantine: List[Dict[str, Any]]):
        if not quarantine: return
        path = os.path.join(self.output_dir, "quarantine.jsonl")
        with open(path, "w") as f:
            for item in quarantine:
                f.write(json.dumps(item) + "\n")

import logging
import json
from typing import List, Dict, Any, Set, Tuple
from .schema import ContaminationFinding
from .config import ContaminationConfig

class DatasetCleaner:
    def __init__(self, config: ContaminationConfig):
        self.config = config
        self.logger = logging.getLogger("lm_adapt_bench.contamination")

    def apply_cleaning(self, splits: Dict[str, List[str]], split_ids: Dict[str, List[str]], findings: List[ContaminationFinding]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[Dict[str, Any]]]:
        """
        Applies cleaning policy to splits based on findings.
        Returns cleaned splits, cleaned split_ids, and quarantine list.
        """
        quarantine = []
        cleaned_splits = {k: list(v) for k, v in splits.items()}
        cleaned_ids = {k: list(v) for k, v in split_ids.items()}
        
        # Group findings by record_id
        to_remove = {k: set() for k in splits.keys()}
        for f in findings:
            if f.split in to_remove:
                to_remove[f.split].add(f.record_id)
                quarantine.append({
                    "record_id": f.record_id,
                    "split": f.split,
                    "detector": f.detector,
                    "reason": f.reason,
                    "action": self.config.cleaning_policy
                })

        if self.config.cleaning_policy in ["drop", "quarantine"]:
            for split_name, ids_to_drop in to_remove.items():
                if not ids_to_drop: continue
                
                new_texts = []
                new_ids = []
                for text, rid in zip(splits[split_name], split_ids[split_name]):
                    if rid not in ids_to_drop:
                        new_texts.append(text)
                        new_ids.append(rid)
                
                cleaned_splits[split_name] = new_texts
                cleaned_ids[split_name] = new_ids
                self.logger.info(f"Cleaned {split_name}: removed {len(ids_to_drop)} contaminated samples.")
        else:
            self.logger.info("Cleaning policy is 'report_only', no samples removed.")

        return cleaned_splits, cleaned_ids, quarantine

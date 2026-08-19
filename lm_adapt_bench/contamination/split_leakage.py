import logging
from typing import List, Dict, Tuple, Any
from .deduper import Deduper

class SplitLeakageChecker:
    def __init__(self, deduper: Deduper):
        self.deduper = deduper
        self.logger = logging.getLogger("lm_adapt_bench.contamination")

    def check_leakage(self, splits: Dict[str, List[str]], split_ids: Dict[str, List[str]], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Checks for leakage between splits.
        splits: {"train": [...], "val": [...], "test": [...]}
        split_ids: {"train": [...], "val": [...], "test": [...]}
        """
        findings = []
        names = list(splits.keys())
        
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s1, s2 = names[i], names[j]
                self.logger.info(f"Checking leakage between {s1} and {s2}")
                
                # Combined dedupe to find cross-split matches
                combined_texts = splits[s1] + splits[s2]
                combined_ids = [f"{s1}:{tid}" for tid in split_ids[s1]] + [f"{s2}:{tid}" for tid in split_ids[s2]]
                
                # Exact leakage
                exact_dupes = self.deduper.find_exact_duplicates(combined_texts, combined_ids)
                for id1, id2 in exact_dupes:
                    split1, tid1 = id1.split(":", 1)
                    split2, tid2 = id2.split(":", 1)
                    if split1 != split2:
                        findings.append({
                            "record_id": tid2,
                            "matched_record_id": tid1,
                            "split": split2,
                            "detector": "exact_split_leakage",
                            "severity": "critical",
                            "score": 1.0,
                            "reason": f"Exact duplicate found between {split1} and {split2}"
                        })

                # Near-duplicate leakage
                near_dupes = self.deduper.find_near_duplicates(combined_texts, combined_ids, threshold=threshold)
                for id1, id2, score in near_dupes:
                    split1, tid1 = id1.split(":", 1)
                    split2, tid2 = id2.split(":", 1)
                    if split1 != split2:
                        findings.append({
                            "record_id": tid2,
                            "matched_record_id": tid1,
                            "split": split2,
                            "detector": "near_split_leakage",
                            "severity": "high" if score > 0.9 else "medium",
                            "score": score,
                            "reason": f"Near-duplicate (score={score:.2f}) found between {split1} and {split2}"
                        })
                        
        return findings

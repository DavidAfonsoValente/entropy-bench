import os
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from .text_normalize import normalize_text
from .fingerprints import Fingerprinter

class ForbiddenChecker:
    def __init__(self, fingerprinter: Fingerprinter, threshold: float = 0.8):
        self.fingerprinter = fingerprinter
        self.threshold = threshold
        self.forbidden_minhashes = {}
        self.lsh = self.fingerprinter.get_lsh_index(threshold=threshold)
        self.logger = logging.getLogger("lm_adapt_bench.contamination")

    def add_forbidden_corpus(self, texts: List[str], source_name: str = "forbidden"):
        """Indexes a forbidden corpus for overlap checking."""
        self.logger.info(f"Indexing forbidden corpus: {source_name} ({len(texts)} samples)")
        for i, text in enumerate(texts):
            fid = f"{source_name}_{i}"
            m = self.fingerprinter.create_minhash(text)
            if m:
                if self.lsh:
                    self.lsh.insert(fid, m)
                self.forbidden_minhashes[fid] = (m, source_name)

    def check_overlap(self, text: str, record_id: str) -> List[Dict[str, Any]]:
        """Checks a single text against the forbidden index."""
        findings = []
        m = self.fingerprinter.create_minhash(text)
        if not m or not self.lsh:
            return findings

        candidates = self.lsh.query(m)
        for fid in candidates:
            forbidden_m, source = self.forbidden_minhashes[fid]
            score = m.jaccard(forbidden_m)
            if score >= self.threshold:
                findings.append({
                    "record_id": record_id,
                    "matched_source": source,
                    "score": score,
                    "detector": "minhash_forbidden",
                    "severity": "critical" if score > 0.95 else "high"
                })
        
        # Exact n-gram check (simple fallback/supplement)
        # In a real enterprise system, we might use a suffix array or more robust n-gram index.
        # For now, MinHash with high threshold covers this.
        
        return findings

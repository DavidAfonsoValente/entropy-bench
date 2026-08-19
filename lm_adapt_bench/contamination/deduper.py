import hashlib
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from .text_normalize import normalize_text
from .fingerprints import Fingerprinter

class Deduper:
    def __init__(self, fingerprinter: Fingerprinter):
        self.fingerprinter = fingerprinter
        self.logger = logging.getLogger("lm_adapt_bench.contamination")

    def find_exact_duplicates(self, texts: List[str], ids: List[str]) -> List[Tuple[str, str]]:
        """Returns list of (canonical_id, duplicate_id) pairs."""
        hashes = {}
        duplicates = []
        for text, tid in zip(texts, ids):
            norm = normalize_text(text)
            h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            if h in hashes:
                duplicates.append((hashes[h], tid))
            else:
                hashes[h] = tid
        return duplicates

    def find_near_duplicates(self, texts: List[str], ids: List[str], threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Returns list of (id1, id2, score) using MinHashLSH."""
        lsh = self.fingerprinter.get_lsh_index(threshold=threshold)
        if lsh is None:
            self.logger.warning("datasketch not installed, skipping near-duplicate detection.")
            return []

        minhashes = {}
        duplicates = []
        
        for text, tid in zip(texts, ids):
            m = self.fingerprinter.create_minhash(text)
            if m:
                # Find candidates
                candidates = lsh.query(m)
                for cand_id in candidates:
                    # Double check similarity
                    score = m.jaccard(minhashes[cand_id])
                    if score >= threshold:
                        duplicates.append((cand_id, tid, score))
                
                # Insert current
                lsh.insert(tid, m)
                minhashes[tid] = m
                
        return duplicates

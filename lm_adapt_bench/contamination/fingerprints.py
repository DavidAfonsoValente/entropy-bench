import hashlib
from typing import List, Set, Optional, Any
try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    # Fallback or handled at runtime if datasketch is not installed
    MinHash = None
    MinHashLSH = None

from .text_normalize import get_word_shingles

class Fingerprinter:
    def __init__(self, num_perm: int = 128, ngram_n: int = 13):
        self.num_perm = num_perm
        self.ngram_n = ngram_n

    def create_minhash(self, text: str) -> Optional[Any]:
        if MinHash is None:
            return None
        
        m = MinHash(num_perm=self.num_perm)
        shingles = get_word_shingles(text, self.ngram_n)
        if not shingles:
            return None
            
        for s in shingles:
            m.update(s.encode("utf-8"))
        return m

    def get_lsh_index(self, threshold: float = 0.8):
        if MinHashLSH is None:
            return None
        return MinHashLSH(threshold=threshold, num_perm=self.num_perm)

import unicodedata
import re

def normalize_text(text: str, lowercase: bool = False) -> str:
    """
    Standard normalization for contamination detection.
    - NFKC Unicode normalization
    - Normalize whitespace (replace any whitespace sequence with a single space)
    - Strip leading/trailing whitespace
    - Remove control characters
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    
    # Remove control characters (keep tab, newline, carriage return for now, 
    # but they'll be space-normalized anyway)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\t\n\r")
    
    if lowercase:
        text = text.lower()
        
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def get_word_shingles(text: str, n: int) -> list:
    """Produces word-level shingles from normalized text."""
    words = text.split()
    if len(words) < n:
        return []
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]


_GOPHER_STOPWORDS = {"the", "be", "to", "of", "and", "that", "have", "with", "this", "from",
                      "not", "are", "was", "for", "it", "as", "on", "in", "is", "you"}
_GOPHER_BULLETS = ("*", "-", "•", "‣", "·", "+")


def gopher_ok(text: str, min_words: int, max_words: int) -> bool:
    """Standard Gopher-style quality filters (Rae et al. 2021, Appendix A.1.1).

    Shared by every corpus-prep script (tools/prepare_scraped_corpus.py,
    tools/prepare_math_corpus.py) so the recipe stays identical across domains.
    """
    words = text.split()
    n = len(words)
    if n < min_words or n > max_words:
        return False
    mean_len = sum(len(w) for w in words) / n
    if mean_len < 3 or mean_len > 10:
        return False
    if (text.count("#") + text.count("...") + text.count("…")) / n > 0.1:
        return False
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        if sum(ln.lstrip().startswith(_GOPHER_BULLETS) for ln in lines) / len(lines) > 0.90:
            return False
        if sum(ln.rstrip().endswith(("...", "…")) for ln in lines) / len(lines) > 0.30:
            return False
    if sum(any(c.isalpha() for c in w) for w in words) / n < 0.80:
        return False
    lowered = {w.strip(".,!?;:\"'()").lower() for w in words}
    if len(lowered & _GOPHER_STOPWORDS) < 2:
        return False
    return True

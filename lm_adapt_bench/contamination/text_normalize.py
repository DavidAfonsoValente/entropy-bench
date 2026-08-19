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

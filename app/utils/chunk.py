from typing import Iterable, List, Dict
import re

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)

def _by_headings(text: str) -> List[Dict[str, str]]:
    # split into sections; keep title for downstream rerank
    parts = []
    last = 0
    current = {"section": "Document", "text": ""}
    for m in _HEADING.finditer(text):
        # flush previous
        current["text"] += text[last:m.start()]
        if current["text"].strip():
            parts.append({"section": current["section"], "text": current["text"].strip()})
        # start new
        current = {"section": m.group(2).strip(), "text": ""}
        last = m.end()
    current["text"] += text[last:]
    if current["text"].strip():
        parts.append({"section": current["section"], "text": current["text"].strip()})
    return parts

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> Iterable[dict]:
    """
    Markdown-aware chunker.
    Yields dicts: {"text": ..., "section": ...}
    """
    out: List[Dict[str, str]] = []
    for sec in _by_headings(text):
        words = sec["text"].split()
        i, n = 0, len(words)
        if n == 0:
            continue
        while i < n:
            j = min(n, i + chunk_size)
            out.append({"section": sec["section"], "text": " ".join(words[i:j])})
            if j == n:
                break
            i = max(0, j - overlap)
    return out

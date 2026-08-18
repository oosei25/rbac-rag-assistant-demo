import os
import re
from collections import Counter
import math
import uuid
from pathlib import PurePosixPath
from typing import Iterable, List, Optional

from app.schemas import Citation

COLLECTION_NAME = "company_docs"
DENY_MESSAGE = "I don't have enough information to answer that with your current access."

# ---- Search backends ------

def qdrant_search(vec, k: int, filt: Optional[dict]):
    from qdrant_client import QdrantClient
    Q = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    response = Q.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=k,
        query_filter=_qdrant_filter_from_dict(filt),
        score_threshold=float(os.getenv("QDRANT_SCORE_MIN", "0.25")),
        with_payload=True,
        with_vectors=False,
    )
    return response.points

def chroma_search(vec, k: int, where: Optional[dict]):
    import chromadb
    INDEX_PATH = os.getenv("INDEX_PATH", ".local_index/chroma")
    CH = chromadb.PersistentClient(path=INDEX_PATH)
    COL = CH.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    query_kwargs = {
        "query_embeddings": [vec],
        "n_results": k,
        "include": ["metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where
    res = COL.query(**query_kwargs)
    metas = (res.get("metadatas", [[]]) or [[]])[0]
    dists = (res.get("distances", [[]]) or [[]])[0]
    max_distance = float(os.getenv("CHROMA_DIST_MAX", "0.75"))
    # Chroma cosine distance is lower-is-better. Filter here so thresholding is
    # performed by the backend adapter, before lexical relevance checks.
    return [
        (meta, distance)
        for meta, distance in zip(metas, dists)
        if distance is not None and float(distance) <= max_distance
    ]

def _qdrant_filter_from_dict(d: Optional[dict]):
    """Convert {'department': {'$in': [...]}} into Qdrant Filter."""
    if not d:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    depts = d.get("department", {}).get("$in", [])
    if not depts:
        raise ValueError("A department filter must contain at least one value")
    match = MatchValue(value=depts[0]) if len(depts) == 1 else MatchAny(any=depts)
    return Filter(must=[FieldCondition(key="department", match=match)])

# ------- Post-filters --------

def normalize_hits(hits) -> List[dict]:
    """Turn backend hits into scored payloads while preserving backend order."""
    out: List[dict] = []
    for h in hits or []:
        payload = getattr(h, "payload", None)
        if payload is not None:
            if payload:
                item = dict(payload)
                item["score"] = float(getattr(h, "score", 0.0) or 0.0)
                out.append(item)
            continue
        if isinstance(h, dict):
            out.append(dict(h))
            continue
        if isinstance(h, (list, tuple)) and len(h) >= 1:
            meta = h[0]
            if meta:
                item = dict(meta)
                distance = float(h[1]) if len(h) > 1 and h[1] is not None else 1.0
                item["distance"] = distance
                item["score"] = 1.0 - distance
                out.append(item)
    return out

def diversify_by_path(
    items: List[dict], limit: int, max_per_document: int | None = None
) -> List[dict]:
    if max_per_document is None:
        max_per_document = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "3"))
    max_per_document = min(3, max(2, max_per_document))
    counts: Counter[str] = Counter()
    seen_chunks: set[str] = set()
    out: List[dict] = []
    for d in items:
        document_key = d.get("document_id") or d.get("path", "")
        chunk_key = d.get("chunk_id") or f"{document_key}:{d.get('text', '')}"
        if chunk_key in seen_chunks or counts[document_key] >= max_per_document:
            continue
        seen_chunks.add(chunk_key)
        counts[document_key] += 1
        out.append(d)
        if len(out) >= limit:
            break
    return out

# -------- Optional reranker -------

_ce = None
def _ensure_ce():
    global _ce
    if _ce is not None:
        return _ce
    if os.getenv("RERANK_CE", "0") != "1":
        _ce = False
        return _ce
    try:
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        _ce = False
    return _ce

def cross_encoder_rerank(query_text: str, items: List[dict]) -> List[dict]:
    ce = _ensure_ce()
    if not ce or len(items) <= 1:
        return items
    pairs = [(query_text, d.get("text", "")) for d in items]
    try:
        scores = ce.predict(pairs)
        return [d for _, d in sorted(zip(scores, items), key=lambda x: x[0], reverse=True)]
    except Exception:
        return items  # fail open

# --- simple lexical rerank (BM25) -------

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_STOP_WORDS = {
    "a", "an", "and", "are", "about", "does", "for", "from", "in", "is",
    "it", "latest", "me", "of", "on", "say", "show", "summarize", "the",
    "to", "what", "with",
}

def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]


def lexical_relevance(query: str, text: str) -> float:
    query_tokens = {token for token in _tokens(query) if token not in _STOP_WORDS}
    if not query_tokens:
        return 1.0
    text_tokens = set(_tokens(text))
    return len(query_tokens & text_tokens) / len(query_tokens)


def lexical_filter(
    query: str, items: List[dict], minimum: float | None = None
) -> List[dict]:
    if minimum is None:
        minimum = float(os.getenv("LEXICAL_MIN", "0.04"))
    relevant: List[dict] = []
    for item in items:
        score = lexical_relevance(query, item.get("text", ""))
        if score >= minimum:
            enriched = dict(item)
            enriched["lexical_score"] = score
            relevant.append(enriched)
    return relevant


def vector_relevance_filter(items: List[dict], backend: str) -> List[dict]:
    """Apply the configured backend threshold to normalized deterministic hits."""
    if backend == "qdrant":
        minimum = float(os.getenv("QDRANT_SCORE_MIN", "0.25"))
        return [item for item in items if float(item.get("score", 0.0)) >= minimum]
    maximum = float(os.getenv("CHROMA_DIST_MAX", "0.75"))
    return [
        item
        for item in items
        if float(item.get("distance", 1.0)) <= maximum
    ]

def lexical_rerank(query: str, items: List[dict], boost: float = 0.25) -> List[dict]:
    """Combine normalized lexical relevance with vector similarity."""
    if len(items) <= 1:
        return items
    q_toks = _tokens(query)
    if not q_toks:
        return items

    # idf from this candidate set
    df = Counter()
    docs_toks: List[List[str]] = []
    for it in items:
        toks = list(set(_tokens(it.get("text", ""))))
        docs_toks.append(toks)
        for t in toks:
            df[t] += 1
    N = len(items)

    def score_doc(toks: List[str]) -> float:
        s = 0.0
        for t in q_toks:
            if t in toks:
                idf = math.log((N + 1) / (1 + df.get(t, 0))) + 1.0
                s += idf
        return s

    scored = [(score_doc(t), it) for t, it in zip(docs_toks, items)]
    max_s = max((s for s, _ in scored), default=1.0)
    if max_s == 0:
        return items
    combined = [
        (float(item.get("score", 0.0)) + (score / max_s) * boost, item)
        for score, item in scored
    ]
    # Equal scores retain vector-store order because Python sorting is stable.
    return [item for _, item in sorted(combined, key=lambda pair: pair[0], reverse=True)]

def _wants_bullets(q: str) -> bool:
    q = " ".join((q or "").lower().split())
    # Do not force bullets for definitional / descriptive asks.
    if any(phrase in q for phrase in ("company overview", "what is finsolve", "about finsolve")):
        return False
    # explicit requests
    if any(kw in q for kw in (
        "bullets", "bullet", "outline", "list", "key points", "takeaways",
        "pros and cons", "pros", "cons", "steps", "checklist"
    )):
        return True
    # Structural "what are the ..." style asks.
    structural_terms = {
        "strategy", "components", "architecture", "phases", "types",
        "pillars", "objectives", "goals", "features", "services", "modules"
    }
    if any(q.startswith(pfx) for pfx in ("what are", "which are", "name the", "list the")) \
       and any(t in q for t in structural_terms):
        return True

    # "What is the ..." strategy/architecture/etc. -> bullets.
    if q.startswith("what is the ") and any(t in q for t in structural_terms):
        return True
    return False

def _is_summary_query(q: str) -> bool:
    q = " ".join((q or "").lower().split())

    # Strong summary cues -> always summary mode
    if any(k in q for k in (
        "summarize", "summary", "recap", "briefly", "key points", "tl;dr", "tldr"
    )):
        return True

    # "overview" only counts if they also hint at a high-level summary
    if "overview" in q and any(k in q for k in (
        "high level", "high-level", "summary", "bullets", "key points"
    )):
        return True

    return False

# ------- Prompt & validation ------------------

def build_prompt(query_text: str, citations: List[Citation]):
    summary_mode = _is_summary_query(query_text)
    num_predict = int(os.getenv(
        "LLM_NUM_PREDICT_SUMMARY" if summary_mode else "LLM_NUM_PREDICT",
        "240" if summary_mode else "200"
    ))
    num_ctx     = int(os.getenv("LLM_NUM_CTX", "2048"))

    snippets = [
        (
            f"[{citation.citation_id}] "
            f"({citation.department}; {citation.title}; {citation.section}) "
            f"{citation.snippet}"
        )
        for citation in citations
    ]
    context = "\n\n".join(snippets)

    if summary_mode:
        system = (
            "You are an internal assistant.\n"
            "Write a concise, neutral summary in 5-8 PLAIN bullets (no headings, no tables, no code).\n"
            "Only use the provided context. Add bracket citations like [1], [2] for concrete facts.\n"
            "Never paste large blocks from the context; rephrase key points in short sentences.\n"
            "If the context does not contain the answer, reply exactly: "
            f"\"{DENY_MESSAGE}\""
        )
        user = (
            f"Question: {query_text}\n\n"
            f"Context (numbered snippets):\n{context}\n\n"
            "Output format:\n"
            "- 5-8 bullets, one sentence each.\n"
            "- No Markdown headings (#), tables, or bold/italic.\n"
            "- Include [n] citations when you state numbers, dates, or named items."
        )
    else:
        wants_bullets = _wants_bullets(query_text)

        base = [
            "You are an internal assistant.",
            "Answer ONLY using the provided context. Do NOT invent dates, amounts, or names.",
            "Include bracket citations like [1], [2] when using specific facts.",
            f'If the context does not contain the answer, reply exactly: "{DENY_MESSAGE}"',
        ]

        if wants_bullets:
            fmt = [
                'Use 4-8 Markdown bullets ("- item"), one short sentence each.',
                "No Markdown headings (#), tables, or bold/italic.",
            ]
        else:
            fmt = [
                "Answer in 2-5 short sentences.",
                "No Markdown headings (#), tables, or bold/italic.",
            ]

        system = "\n".join(base + fmt)


        user = (
            f"Question: {query_text}\n\n"
            f"Context (numbered snippets):\n{context}\n\n"
            "Instructions:\n"
            "- Use only the snippets above.\n"
            "- Cite with [1], [2]... matching the snippet numbers when you use specific facts.\n"
            f"- If unsupported, say: \"{DENY_MESSAGE}\""
        )

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "options": {"temperature": 0.1, "num_predict": num_predict, "num_ctx": num_ctx},
    }

_CITATION_MARKER = re.compile(r"\[(\d+)\]")


def _safe_citation_path(raw_path: str) -> str:
    normalized = (raw_path or "").replace("\\", "/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in {"", "/"}]
    if "data" in parts:
        parts = parts[parts.index("data") + 1 :]
    if not parts or ".." in parts:
        return "document"
    if PurePosixPath(normalized).is_absolute() and "data" not in PurePosixPath(normalized).parts:
        parts = parts[-1:]
    return PurePosixPath(*parts).as_posix()


def build_citations(docs: Iterable[dict]) -> List[Citation]:
    """Assign prompt IDs once, in final retrieval order."""
    max_chars = int(os.getenv("CITATION_SNIPPET_CHARS", "1200"))
    citations: List[Citation] = []
    for citation_id, doc in enumerate(docs, 1):
        path = _safe_citation_path(str(doc.get("path", "")))
        document_id = str(
            doc.get("document_id")
            or uuid.uuid5(uuid.NAMESPACE_URL, f"rbac-rag:{path}")
        )
        snippet = " ".join(str(doc.get("text", "")).split())[:max_chars]
        citations.append(
            Citation(
                citation_id=citation_id,
                document_id=document_id,
                path=path,
                title=str(doc.get("title") or PurePosixPath(path).name),
                department=str(doc.get("department", "")),
                section=str(doc.get("section") or doc.get("title") or "Document"),
                score=float(doc.get("score", 0.0) or 0.0),
                snippet=snippet,
            )
        )
    return citations


def validate_answer(answer: str, citations: List[Citation], **_ignored) -> bool:
    if not answer:
        return False
    if looks_like_deny(answer):
        return True
    cited_ids = [int(match) for match in _CITATION_MARKER.findall(answer)]
    valid_ids = {citation.citation_id for citation in citations}
    return bool(cited_ids) and all(citation_id in valid_ids for citation_id in cited_ids)


def citations_for_answer(answer: str, citations: List[Citation]) -> List[Citation]:
    cited_ids = {int(match) for match in _CITATION_MARKER.findall(answer)}
    return [
        citation for citation in citations if citation.citation_id in cited_ids
    ]


# ------------ spacing / markdown fixes ----------------------

_NUM_WORD = r"(million|billion|thousand|m|mm|bn|k)"
_PCT_WORD = r"(percent)"

def _strip_markdown(s: str) -> str:
    # strip headings
    s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)
    # strip **bold** and *italic*/_italic_
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\\)\*(.*?)\*(?!\*)", r"\1", s)
    s = re.sub(r"_(.*?)_", r"\1", s)
    return s

def _inline_to_list(txt: str) -> str:
    new_lines = []
    for line in txt.splitlines():
        # Split any line that contains inline bullet separators, even if it starts with a bullet
        if re.search(r"\s[\*\u2022-]\s", line):
            parts = [p.strip(" -*\u2022") for p in re.split(r"\s[\*\u2022-]\s", line) if p.strip()]
            if len(parts) >= 2:
                new_lines.extend(f"- {p}" for p in parts)
                continue
        new_lines.append(line)
    return "\n".join(new_lines)

def _bulletify_inline_lists(s: str) -> str:
    if not s:
        return s
    # star-separated lists on one line
    if " * " in s and "\n" not in s:
        parts = [p.strip(" -*") for p in s.split(" * ") if p.strip()]
        if 2 < len(parts) <= 12:
            return "\n".join(f"- {p}" for p in parts)
    # semicolon-separated lists on one line
    if s.count(";") >= 2 and "\n" not in s:
        parts = [p.strip(" ;") for p in re.split(r";\s*", s) if p.strip()]
        if 2 < len(parts) <= 12:
            return "\n".join(f"- {p}" for p in parts)
    return s

def _force_split_inline_bullets(s: str) -> str:
    if "\n" in s or not s:
        return s
    s = re.sub(r":\s*[\*\u2022-]\s+", ":\n- ", s)  
    s = re.sub(r"\s[\*\u2022-]\s+", "\n- ", s)      
    return s

def sanitize_answer(s: str) -> str:
    if not s:
        return s

    # Layout fixes first
    s = re.sub(r'(?m)^\s{0,3}#{1,6}\s*', '', s)
    s = re.sub(r'(?m)^\s*(?:-{3,}|_{3,}|\*{3,})\s*$', '', s)

    s = _force_split_inline_bullets(s)

    # Spacing/numbering
    s = re.sub(r"(?<=\d)(?=[a-z])", " ", s)
    s = re.sub(fr"(\$?\d+(?:\.\d+)?)\s*{_NUM_WORD}\b", r"\1 \2", s, flags=re.I)
    s = re.sub(fr"(\d+(?:\.\d+)?)\s*{_PCT_WORD}\b", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\s+([.,;:])", r"\1", s)
    s = re.sub(r"([.,;:])(?=\S)", r"\1 ", s)

    # Emphasis removal (not list markers)
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\S)\*(?!\s)([^*]+)\*(?!\S)", r"\1", s)
    s = re.sub(r"_(.*?)_", r"\1", s)

    # Collapse spaces, escape money
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = s.replace("$", r"\$")
    s = re.sub(r":\s*[\*\u2022-]\s+", ":\n- ", s)
    s = re.sub(r"(?<!\n)\s[\*\u2022-]\s+", "\n- ", s)

    # add helpers
    s = _inline_to_list(s)                           
    s = re.sub(r"(?m)^\s*[\*\u2022]\s+", "- ", s)    
    s = s.replace("_", r"\_")
    s = re.sub(r"(?m)(?<!^)\*(?!\s)", r"\*", s)
    s = _bulletify_inline_lists(s)                    

    if os.getenv("FORCE_PLAIN", "0") == "1":
        s = _strip_markdown(s)
    return s


# ------ Query rewrite / fallback -----

def rewrite_query(q: str) -> str:
    """
    Very light rewrite to help retrieval:
    - lowercase, strip weird whitespace
    - add synonyms for frequent policy intents (dress code, PTO, WFH, etc.)
    """
    base = " ".join(q.lower().split())
    synonyms = [
        (r"\bdress\s*code\b", "dress code attire clothing policy"),
        (r"\bpto\b|\bpaid time off\b", "paid time off PTO leave policy"),
        (r"\bwork\s*from\s*home\b|\bwfh\b", "remote work work from home WFH policy"),
        (r"\breimbursement\b", "expense reimbursement policy"),
    ]
    for pat, exp in synonyms:
        if re.search(pat, base):
            base = f"{base} {exp}"
    return base

_DENY_PAT = re.compile(
    r"\b(i\s+don[\u2019']?t|cannot|can[ ]?not|unable|no\s+enough|insufficient)\b.*\b(information|access|data)\b"
    r"|^i\s+don[\u2019']?t\s+have\s+enough|\bnot\s+authorized\b",
    re.I,
)

def looks_like_deny(ans: str) -> bool:
    return bool(_DENY_PAT.search(ans or ""))

def _best_lines_for_query(query: str, text: str) -> List[str]:
    """Pick lines/sentences matching query tokens."""
    if not text:
        return []
    q = set(_tokens(query))
    # split on newlines or sentence punctuation
    cand = re.split(r"(?<=[.!?])\s+|\n+", text)
    scored = []
    for line in cand:
        toks = set(_tokens(line))
        if not toks:
            continue
        overlap = len(q & toks)
        if overlap == 0:
            continue
        # prefer compact, policy-like sentences
        score = overlap / math.log2(len(toks) + 2)
        scored.append((score, line.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ln for _, ln in scored[:3]]

def keyword_slice_answer(query: str, docs: List[dict], max_chars: int = 400) -> str:
    """
    Deterministic fallback: extract 1-3 relevant policy sentences from retrieved docs.
    Keep each excerpt tied to its original prompt citation ID.
    """
    cited_lines: List[str] = []
    for citation_id, document in enumerate(docs, 1):
        lines = _best_lines_for_query(query, document.get("text", ""))
        if lines:
            marker = f" [{citation_id}]"
            used = sum(len(line) + 1 for line in cited_lines)
            remaining = max_chars - used - len(marker)
            if remaining <= 0:
                break
            cited_lines.append(f"{' '.join(lines)[:remaining].rstrip()}{marker}")
        if sum(len(line) for line in cited_lines) >= max_chars:
            break
    if not cited_lines:
        return ""
    snippet = " ".join(cited_lines)
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet

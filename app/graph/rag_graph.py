import re
from typing import TypedDict, List, Optional, Literal, Dict, Any, Iterable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.policy import allowed_departments, infer_requested_departments
from app.services.indexer import VECTOR_DB, embed_one
from ollama import Client
import os

from app.services.rag_helpers import (
    qdrant_search, chroma_search,            
    postfilter_strict, postfilter_relaxed,   
    diversify_by_path, cross_encoder_rerank, 
    build_prompt, validate_answer,           
    sanitize_answer,                         
    rewrite_query, looks_like_deny,          
    keyword_slice_answer, lexical_rerank     
)


OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
TOP_K        = int(os.getenv("TOP_K", "4"))
RBAC_INTENT       = os.getenv("RBAC_INTENT", "0") == "1"
RBAC_INTENT_SOFT  = os.getenv("RBAC_INTENT_SOFT", "0") == "1"
RERANK_CE         = os.getenv("RERANK_CE", "0") == "1"
USE_KEYWORD_FALLBACK = os.getenv("USE_KEYWORD_FALLBACK", "0") == "1"
PASSAGE_SELECTION    = os.getenv("PASSAGE_SELECTION", "0") == "1"
PASSAGE_SELECTION_K  = int(os.getenv("PASSAGE_SELECTION_K", "5"))

ollama = Client(host=OLLAMA_HOST)

class RAGState(TypedDict, total=False):
    query: str
    role: str
    requested_depts: set[str]
    allowed_depts: set[str]
    docs: List[dict]
    sources: List[str]
    answer: str
    error: Optional[str]
    stage: Literal[
        "guarded","retrieved_strict","retrieved_relaxed","deflected_general",
        "reranked","generated","validated","failed"
    ]

def _role_filter(depts: Iterable[str]) -> Dict[str, Any]:
    # Unified filter description. qdrant_search/chroma_search adapt this dict.
    return {"department": {"$in": list(depts)}}

def _backend_search(vec, k, filt):
    return qdrant_search(vec, k, filt) if VECTOR_DB == "qdrant" else chroma_search(vec, k, filt)

def _unique_sources(docs: List[dict]) -> List[str]:
    seen: set[str] = set()
    srcs: List[str] = []
    for d in docs:
        path = d.get("path", "")
        if path and path not in seen:
            seen.add(path)
            srcs.append(path)
    return srcs

DEBUG_GRAPH = os.getenv("DEBUG_GRAPH", "0") == "1"
def dbg(stage, **kw):
    if DEBUG_GRAPH:
        print(f"[graph:{stage}]", {k: (len(v) if isinstance(v, list) else v) for k, v in kw.items()})

# ---Nodes -----
def n_intent_guard(state: RAGState) -> RAGState:
    state["requested_depts"] = infer_requested_departments(state["query"])
    state["allowed_depts"]   = set(allowed_departments(state["role"]))
    # Hard-deny only when SOFT=0; otherwise we let it flow (deflection/safety net will help)
    if RBAC_INTENT and not RBAC_INTENT_SOFT:
        if state["requested_depts"] - state["allowed_depts"]:
            state["error"] = "access_denied"
    state["stage"] = "guarded"
    return state

def n_retrieve_strict(state: RAGState) -> RAGState:
    q    = rewrite_query(state["query"])
    vec  = embed_one(q)
    filt = _role_filter(state["allowed_depts"])
    hits = _backend_search(vec, max(TOP_K, 8), filt)
    state["docs"]  = postfilter_strict(state["query"], hits)[:TOP_K]
    dbg("retrieve_strict", n_docs=len(state["docs"]))
    state["stage"] = "retrieved_strict"
    return state

def n_retrieve_relaxed(state: RAGState) -> RAGState:
    q    = rewrite_query(state["query"])
    vec  = embed_one(q)
    filt = _role_filter(state["allowed_depts"])
    hits = _backend_search(vec, max(TOP_K, 12), filt)
    more = postfilter_relaxed(state["query"], hits)
    state["docs"]  = (state.get("docs") or []) + more
    state["docs"]  = state["docs"][:max(TOP_K * 2, TOP_K)]
    state["stage"] = "retrieved_relaxed"
    return state

def n_deflect_general(state: RAGState) -> RAGState:
    # Safety net: if we STILL have no docs, try 'general' unconditionally.
    if state.get("docs"):
        return state
    q    = rewrite_query(state["query"])
    vec  = embed_one(q)
    filt = _role_filter({"general"})
    hits = _backend_search(vec, max(TOP_K * 3, 12), filt)
    add  = postfilter_relaxed(state["query"], hits)
    state["docs"]  = (state.get("docs") or []) + add
    state["stage"] = "deflected_general"
    return state

def n_rerank(state: RAGState) -> RAGState:
    docs = state.get("docs") or []
    if not docs:
        return state
    docs = diversify_by_path(docs, limit=max(TOP_K * 2, TOP_K))
    docs = lexical_rerank(state["query"], docs, boost=0.25)
    if RERANK_CE:
        docs = cross_encoder_rerank(state["query"], docs)
    state["docs"]  = docs
    state["stage"] = "reranked"
    return state

def n_generate(state: RAGState) -> RAGState:
    docs = state.get("docs") or []
    if not docs:
        return state
    if PASSAGE_SELECTION:
        docs = _select_passages_llm(state["query"], docs, OLLAMA_MODEL)
        state["docs"] = docs
    prompt = build_prompt(state["query"], docs, model=OLLAMA_MODEL)
    resp   = ollama.chat(model=OLLAMA_MODEL, messages=prompt["messages"], options=prompt["options"])
    ans    = (resp.get("message", {}) or {}).get("content", "").strip()
    ans    = sanitize_answer(ans)

    if looks_like_deny(ans) and USE_KEYWORD_FALLBACK:
        fallback = keyword_slice_answer(state["query"], docs)
        if fallback:
            ans = sanitize_answer(fallback)
        
        state["answer"]  = ans
        state["sources"] = []
        state["stage"]   = "generated"
        return state

def n_validate(state: RAGState) -> RAGState:
    docs = state.get("docs") or []
    ans  = state.get("answer","")

    # Denials are already source-free; accept as terminal
    ok = True if looks_like_deny(ans) else validate_answer(ans, docs)

    if not ok:
        state["error"] = "validate_failed"
        state["stage"] = "failed"
    else:
        state["stage"] = "validated"
    return state


def n_fallback(state: RAGState) -> RAGState:
    state["answer"]  = "I don’t have enough information to answer that with your current access."
    state["sources"] = []
    state["stage"]   = "failed"
    return state

# --- Wire the Graph -----
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("intent_guard",     n_intent_guard)
    g.add_node("retrieve_strict",  n_retrieve_strict)
    g.add_node("retrieve_relaxed", n_retrieve_relaxed)
    g.add_node("deflect_general",  n_deflect_general)
    g.add_node("rerank",           n_rerank)
    g.add_node("generate",         n_generate)
    g.add_node("validate",         n_validate)
    g.add_node("fallback",         n_fallback)

    g.set_entry_point("intent_guard")

    g.add_conditional_edges(
        "intent_guard",
        lambda s: "deny" if s.get("error") == "access_denied" else "ok",
        {"deny": "fallback", "ok": "retrieve_strict"},
    )
    g.add_conditional_edges(
        "retrieve_strict",
        lambda s: "have_docs" if s.get("docs") else "need_more",
        {"have_docs": "rerank", "need_more": "retrieve_relaxed"},
    )
    g.add_conditional_edges(
        "retrieve_relaxed",
        lambda s: "have_docs" if s.get("docs") else "maybe_general",
        {"have_docs": "rerank", "maybe_general": "deflect_general"},
    )
    g.add_conditional_edges(
        "deflect_general",
        lambda s: "have_docs" if s.get("docs") else "fail",
        {"have_docs": "rerank", "fail": "fallback"},
    )
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges(
        "validate",
        lambda s: "ok" if s.get("error") is None else "bad",
        {"ok": END, "bad": "fallback"},
    )

    return g.compile(checkpointer=MemorySaver())
PASSAGE_SELECT_INSTR = (
    "Select the passages that best answer the user's question. Respond with a JSON "
    "array of snippet numbers like [1,3,4]. The numbers must match the IDs shown "
    "in the context list."
)

def _select_passages_llm(query_text: str, docs: List[dict], model: str) -> List[dict]:
    """
    Mirror RagService passage selection so the LangGraph path behaves identically.
    """
    if not PASSAGE_SELECTION or len(docs) <= PASSAGE_SELECTION_K:
        return docs

    snippets = []
    for i, d in enumerate(docs, 1):
        text = (d.get("text") or "").replace("\n", " ")
        snippets.append(f"[{i}] {text[:500]}")
    context = "\n\n".join(snippets)

    messages = [
        {"role": "system", "content": PASSAGE_SELECT_INSTR},
        {
            "role": "user",
            "content": (
                f"Question: {query_text}\n\n"
                f"Passages:\n{context}\n\n"
                f"Return up to {PASSAGE_SELECTION_K} snippet numbers."
            ),
        },
    ]

    try:
        resp = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0, "num_predict": 60},
        )
        raw = (resp.get("message", {}) or {}).get("content", "")
    except Exception as e:
        if DEBUG_GRAPH:
            print("[graph] passage_select error:", e)
        return docs[:PASSAGE_SELECTION_K]

    nums = []
    for match in re.findall(r"\d+", raw):
        try:
            idx = int(match)
        except ValueError:
            continue
        if 1 <= idx <= len(docs):
            nums.append(idx)

    seen = set()
    selected: List[dict] = []
    for idx in nums:
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(docs[idx - 1])
        if len(selected) >= PASSAGE_SELECTION_K:
            break

    if not selected and DEBUG_GRAPH:
        print("[graph] passage_select fallback, raw response:", raw[:200])

    return selected or docs[:PASSAGE_SELECTION_K]

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.policy import allowed_departments, infer_requested_departments
from app.schemas import AccessDecisionTrace, Citation
from app.services.rag import rag_service
from app.services.rag_helpers import DENY_MESSAGE


class RAGState(TypedDict, total=False):
    query: str
    role: str
    requested_depts: set[str]
    allowed_depts: set[str]
    docs: List[dict]
    citations: List[Citation]
    retrieval_diagnostics: dict
    access_trace: AccessDecisionTrace
    answer: str
    error: Optional[str]
    stage: Literal["guarded", "retrieved", "generated", "failed"]


def n_intent_guard(state: RAGState) -> RAGState:
    """Record intent as retrieval context; never use it as a denial boundary."""
    state["requested_depts"] = infer_requested_departments(state["query"])
    state["allowed_depts"] = set(allowed_departments(state["role"]))
    state["docs"] = []
    state["citations"] = []
    state.pop("retrieval_diagnostics", None)
    state.pop("access_trace", None)
    state["answer"] = ""
    state["error"] = None
    state["stage"] = "guarded"
    return state


def n_retrieve(state: RAGState) -> RAGState:
    documents, diagnostics = rag_service.retrieve_with_trace(
        state["query"], state["role"]
    )
    state["docs"] = documents
    state["retrieval_diagnostics"] = diagnostics
    state["stage"] = "retrieved"
    return state


def n_generate(state: RAGState) -> RAGState:
    answer, citations, reason = rag_service.generate_from_documents_with_reason(
        state["query"], state.get("docs") or []
    )
    state["answer"] = answer
    state["citations"] = citations
    state["access_trace"] = rag_service.build_access_trace(
        state["role"], state["retrieval_diagnostics"], reason, citations
    )
    state["stage"] = "generated" if citations else "failed"
    return state


def n_fallback(state: RAGState) -> RAGState:
    state["answer"] = DENY_MESSAGE
    state["citations"] = []
    diagnostics = state.get("retrieval_diagnostics") or {
        "allowed_departments": sorted(allowed_departments(state["role"])),
        "requested_departments": sorted(
            infer_requested_departments(state["query"])
        ),
        "initial_filter_departments": sorted(allowed_departments(state["role"])),
        "fallback_filter_departments": [],
        "authorized_after_policy": 0,
        "authorized_after_relevance": 0,
        "selected_for_generation": 0,
    }
    state["access_trace"] = rag_service.build_access_trace(
        state["role"], diagnostics, "no_authorized_relevant_context", []
    )
    state["stage"] = "failed"
    return state


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("intent_guard", n_intent_guard)
    graph.add_node("retrieve", n_retrieve)
    graph.add_node("generate", n_generate)
    graph.add_node("fallback", n_fallback)
    graph.set_entry_point("intent_guard")
    graph.add_edge("intent_guard", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        lambda state: "generate" if state.get("docs") else "fallback",
        {"generate": "generate", "fallback": "fallback"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile(checkpointer=MemorySaver())


__all__ = [
    "RAGState",
    "build_graph",
    "n_fallback",
    "n_generate",
    "n_intent_guard",
    "n_retrieve",
]

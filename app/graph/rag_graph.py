from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.policy import allowed_departments, infer_requested_departments
from app.schemas import Citation
from app.services.rag import rag_service
from app.services.rag_helpers import DENY_MESSAGE


class RAGState(TypedDict, total=False):
    query: str
    role: str
    requested_depts: set[str]
    allowed_depts: set[str]
    docs: List[dict]
    citations: List[Citation]
    answer: str
    error: Optional[str]
    stage: Literal["guarded", "retrieved", "generated", "failed"]


def n_intent_guard(state: RAGState) -> RAGState:
    """Record intent as retrieval context; never use it as a denial boundary."""
    state["requested_depts"] = infer_requested_departments(state["query"])
    state["allowed_depts"] = set(allowed_departments(state["role"]))
    state["docs"] = []
    state["citations"] = []
    state["answer"] = ""
    state["error"] = None
    state["stage"] = "guarded"
    return state


def n_retrieve(state: RAGState) -> RAGState:
    state["docs"] = rag_service.retrieve(state["query"], state["role"])
    state["stage"] = "retrieved"
    return state


def n_generate(state: RAGState) -> RAGState:
    answer, citations = rag_service.generate_from_documents(
        state["query"], state.get("docs") or []
    )
    state["answer"] = answer
    state["citations"] = citations
    state["stage"] = "generated" if citations else "failed"
    return state


def n_fallback(state: RAGState) -> RAGState:
    state["answer"] = DENY_MESSAGE
    state["citations"] = []
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

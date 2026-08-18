from __future__ import annotations

import pytest

import app.graph.rag_graph as graph_module
from app.graph.rag_graph import n_generate, n_intent_guard, n_retrieve
from app.services.rag_helpers import DENY_MESSAGE
from evals.harness import build_harness


pytestmark = pytest.mark.integration


def test_unknown_role_stops_before_query_embedding_or_generation():
    harness = build_harness()
    initial_embedding_calls = len(harness.embedder.calls)

    answer, citations = harness.service.generate("general handbook policy", "guest")

    assert answer == DENY_MESSAGE
    assert citations == []
    assert len(harness.embedder.calls) == initial_embedding_calls
    assert harness.vector_store.calls == []
    assert harness.llm.calls == []


def test_backend_cannot_inject_unauthorized_context_when_it_ignores_filters():
    harness = build_harness(ignore_store_filters=True)

    answer, citations = harness.service.generate(
        "finance budget revenue reimbursement", "employee"
    )

    assert answer == DENY_MESSAGE
    assert citations == []
    assert harness.llm.calls == []


def test_authorized_mixed_intent_returns_only_allowed_context():
    harness = build_harness(ignore_store_filters=True)

    answer, citations = harness.service.generate(
        "Explain engineering API architecture and finance reimbursement", "engineering"
    )

    assert "ENGINEERINGEVIDENCE9074" in answer
    assert "FINANCEEVIDENCE4821" not in answer
    assert {citation.department for citation in citations} == {"engineering"}
    prompt = harness.llm.calls[0]["messages"][1]["content"]
    assert "FINANCEEVIDENCE4821" not in prompt


def test_graph_and_direct_rag_use_the_same_deterministic_pipeline(monkeypatch):
    harness = build_harness()
    monkeypatch.setattr(graph_module, "rag_service", harness.service)
    direct_answer, direct_citations = harness.service.generate(
        "general handbook holiday", "employee"
    )

    state = n_intent_guard({"query": "general handbook holiday", "role": "employee"})
    state = n_retrieve(state)
    state = n_generate(state)

    assert state["answer"] == direct_answer
    assert state["citations"] == direct_citations
    assert state["access_trace"].decision == "answered"
    assert state["access_trace"].authenticated_role == "employee"
    assert state["access_trace"].authorized_source_count == len(direct_citations)

from __future__ import annotations

import json

import pytest

from app.services.rag_helpers import DENY_MESSAGE
from evals.harness import build_harness


pytestmark = pytest.mark.unit


def test_answered_trace_contains_only_authorized_candidate_metrics():
    harness = build_harness(ignore_store_filters=True)

    answer, citations, trace = harness.service.generate_with_trace(
        "Compare the general handbook holiday policy with finance revenue.",
        "employee",
    )
    serialized = json.dumps(trace.model_dump())

    assert "GENERALEVIDENCE1042" in answer
    assert {citation.department for citation in citations} == {"general"}
    assert trace.authenticated_role == "employee"
    assert trace.allowed_departments == ["general"]
    assert trace.requested_departments == ["finance", "general"]
    assert trace.applied_filter.initial_departments == ["general"]
    assert trace.decision == "answered"
    assert trace.reason == "grounded_authorized_answer"
    assert trace.authorized_source_count == 1
    assert trace.candidate_counts.authorized_after_policy >= 1
    assert "FINANCEEVIDENCE4821" not in serialized
    assert "quarterly_financial_report" not in serialized
    assert "finance/" not in serialized


def test_denied_trace_does_not_disclose_rejected_candidate_metadata():
    harness = build_harness(ignore_store_filters=True)

    answer, citations, trace = harness.service.generate_with_trace(
        "finance budget revenue reimbursement", "employee"
    )
    serialized = json.dumps(trace.model_dump())

    assert answer == DENY_MESSAGE
    assert citations == []
    assert trace.decision == "denied"
    assert trace.reason == "no_authorized_relevant_context"
    assert trace.authorized_source_count == 0
    assert trace.candidate_counts.authorized_after_relevance == 0
    assert trace.candidate_counts.selected_for_generation == 0
    assert "FINANCEEVIDENCE4821" not in serialized
    assert "Quarterly Financial Report" not in serialized


def test_unknown_role_trace_fails_closed_before_search():
    harness = build_harness()
    initial_embedding_calls = len(harness.embedder.calls)

    answer, citations, trace = harness.service.generate_with_trace(
        "general handbook", "guest"
    )

    assert answer == DENY_MESSAGE
    assert citations == []
    assert trace.allowed_departments == []
    assert trace.applied_filter.initial_departments == []
    assert trace.decision == "denied"
    assert trace.reason == "role_not_authorized"
    assert len(harness.embedder.calls) == initial_embedding_calls
    assert harness.vector_store.calls == []


def test_invalid_model_citation_is_explained_without_sources():
    harness = build_harness(responses=["Unsupported claim [99]."])

    answer, citations, trace = harness.service.generate_with_trace(
        "general handbook holiday", "employee"
    )

    assert answer == DENY_MESSAGE
    assert citations == []
    assert trace.reason == "answer_failed_citation_validation"
    assert trace.authorized_source_count == 0


from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from evals.harness import build_harness


pytestmark = pytest.mark.integration


def test_chat_response_includes_sanitized_access_trace(monkeypatch):
    harness = build_harness(ignore_store_filters=True)
    monkeypatch.setattr(main_module, "rag_service", harness.service)

    response = TestClient(main_module.app).post(
        "/chat/rag",
        json={"message": "finance budget revenue reimbursement"},
        auth=("Emma", "password"),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["citations"] == []
    assert data["access_trace"]["authenticated_role"] == "employee"
    assert data["access_trace"]["allowed_departments"] == ["general"]
    assert data["access_trace"]["requested_departments"] == ["finance"]
    assert data["access_trace"]["decision"] == "denied"
    assert "path" not in data["access_trace"]
    assert "snippet" not in data["access_trace"]


def test_security_lab_rejects_non_clevel_users_before_comparison(monkeypatch):
    harness = build_harness()
    monkeypatch.setattr(main_module, "rag_service", harness.service)

    response = TestClient(main_module.app).post(
        "/security-lab/compare",
        json={
            "question": "What does payroll say about compensation?",
            "left_role": "marketing",
            "right_role": "hr",
        },
        auth=("Emma", "password"),
    )

    assert response.status_code == 403
    assert harness.vector_store.calls == []
    assert harness.llm.calls == []


def test_clevel_security_lab_compares_same_question_under_two_roles(monkeypatch):
    harness = build_harness(ignore_store_filters=True)
    monkeypatch.setattr(main_module, "rag_service", harness.service)
    question = "What does payroll say about compensation?"

    response = TestClient(main_module.app).post(
        "/security-lab/compare",
        json={
            "question": question,
            "left_role": "marketing",
            "right_role": "hr",
        },
        auth=("Cathy", "cathyceo"),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == question
    assert data["left"]["role"] == "marketing"
    assert data["left"]["access_trace"]["decision"] == "denied"
    assert data["left"]["citations"] == []
    assert data["right"]["role"] == "hr"
    assert data["right"]["access_trace"]["decision"] == "answered"
    assert {item["department"] for item in data["right"]["citations"]} == {"hr"}


def test_security_lab_rejects_unknown_comparison_roles():
    response = TestClient(main_module.app).post(
        "/security-lab/compare",
        json={
            "question": "general handbook",
            "left_role": "guest",
            "right_role": "employee",
        },
        auth=("Cathy", "cathyceo"),
    )

    assert response.status_code == 422


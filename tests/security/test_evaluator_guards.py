from __future__ import annotations

from types import SimpleNamespace

import pytest

import evals.runner as runner
from app.schemas import Citation
from evals.harness import build_harness


pytestmark = pytest.mark.security


def test_allowed_case_cannot_pass_via_generic_denial(monkeypatch):
    monkeypatch.setattr(runner, "build_harness", lambda: build_harness(documents=[]))
    case = {
        "name": "allowed_general",
        "role": "employee",
        "target_department": "general",
        "allowed": True,
        "query": "general handbook holiday",
        "evidence": "GENERALEVIDENCE1042",
    }

    result = runner._answer_case("cross_department", case)

    assert not result["passed"]
    assert {item["code"] for item in result["violations"]} >= {
        "unexpected_denial",
        "missing_expected_content",
        "missing_citation",
    }


def test_forbidden_case_fails_when_authorization_leaks(monkeypatch):
    citation = Citation(
        citation_id=1,
        document_id="finance-doc",
        path="finance/report.md",
        title="Finance",
        department="finance",
        section="Budget",
        score=1.0,
        snippet="FINANCEEVIDENCE4821",
    )
    unsafe_service = SimpleNamespace(
        generate=lambda _query, _role: ("FINANCEEVIDENCE4821 [1]", [citation])
    )
    monkeypatch.setattr(
        runner,
        "build_harness",
        lambda: SimpleNamespace(service=unsafe_service),
    )
    case = {
        "name": "forbidden_finance",
        "role": "employee",
        "target_department": "finance",
        "allowed": False,
        "query": "finance budget revenue",
        "evidence": "FINANCEEVIDENCE4821",
    }

    result = runner._answer_case("cross_department", case)

    assert not result["passed"]
    assert {item["code"] for item in result["violations"]} >= {
        "expected_denial",
        "unauthorized_source_leak",
        "unauthorized_content_leak",
    }

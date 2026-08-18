from __future__ import annotations

from collections import Counter

import pytest
from fastapi.testclient import TestClient

import app.graph.rag_graph as graph_module
import app.main as main_module
from app.graph.rag_graph import n_fallback, n_generate, n_intent_guard, n_retrieve
from app.services.documents import DocumentService, canonical_document_id
from app.services.auth import AuthService
from app.services.rag import RagService
from app.services.rag_helpers import (
    DENY_MESSAGE,
    build_citations,
    build_prompt,
    citations_for_answer,
    diversify_by_path,
    validate_answer,
    vector_relevance_filter,
)


class FakeIndexer:
    vector_db = "qdrant"

    def __init__(self):
        self.embed_calls = 0

    def embed_one(self, _text):
        self.embed_calls += 1
        return [0.1, 0.2]


class FakeLlm:
    def __init__(self, answer="Allowed answer [1]"):
        self.answer = answer
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": self.answer}}


class FakeRagService(RagService):
    def __init__(self, hits, answer="Allowed answer [1]"):
        self.fake_indexer = FakeIndexer()
        self.fake_llm = FakeLlm(answer)
        self.hits = hits
        self.filters = []
        super().__init__(indexer=self.fake_indexer, llm_client=self.fake_llm)

    def _search_backend(self, vector, k, filt):
        self.filters.append(filt)
        return self.hits[:k]


def hit(
    department: str,
    text: str,
    *,
    path: str | None = None,
    document_id: str | None = None,
    chunk_id: str = "chunk-1",
    score: float = 0.9,
):
    path = path or f"{department}/document.md"
    return {
        "department": department,
        "document_id": document_id or f"{department}-id",
        "chunk_id": chunk_id,
        "path": path,
        "title": f"{department.title()} Document",
        "section": "Overview",
        "sensitivity": "internal",
        "score": score,
        "text": text,
    }


def test_unknown_role_is_denied_before_embedding_or_generation():
    service = FakeRagService([hit("general", "company policy")])

    answer, citations = service.generate("What is company policy?", "guest")

    assert answer == DENY_MESSAGE
    assert citations == []
    assert service.fake_indexer.embed_calls == 0
    assert service.fake_llm.calls == []


def test_backend_cannot_inject_unauthorized_llm_context():
    service = FakeRagService(
        [
            hit("finance", "secret payroll policy", chunk_id="secret"),
            hit("general", "public company policy", chunk_id="allowed"),
        ]
    )

    answer, citations = service.generate("What is the company policy?", "employee")

    assert answer == "Allowed answer [1]"
    assert [citation.department for citation in citations] == ["general"]
    prompt = service.fake_llm.calls[0]["messages"][1]["content"]
    assert "secret payroll" not in prompt
    assert "public company policy" in prompt


def test_disallowed_intent_does_not_false_deny_allowed_context():
    service = FakeRagService(
        [hit("general", "The general payroll policy describes request routing.")]
    )

    answer, citations = service.generate("Explain the payroll policy", "employee")

    assert answer != DENY_MESSAGE
    assert citations[0].department == "general"
    assert service.filters == [{"department": {"$in": ["general"]}}]


def test_ambiguous_intent_can_narrow_to_an_allowed_department():
    service = FakeRagService(
        [hit("engineering", "The revenue service architecture uses modular design.")]
    )

    answer, citations = service.generate(
        "How does the revenue service architecture work?", "engineering"
    )

    assert answer != DENY_MESSAGE
    assert citations[0].department == "engineering"
    assert service.filters == [
        {"department": {"$in": ["engineering"]}},
        {"department": {"$in": ["engineering", "general"]}},
    ]


def test_intent_narrowing_falls_back_to_all_authorized_departments():
    class FilterAwareService(FakeRagService):
        def _search_backend(self, vector, k, filt):
            self.filters.append(filt)
            if filt["department"]["$in"] == ["engineering"]:
                return []
            return self.hits[:k]

    service = FilterAwareService(
        [hit("general", "The revenue architecture policy is in the handbook.")]
    )

    answer, citations = service.generate(
        "Explain the revenue architecture policy", "engineering"
    )

    assert answer != DENY_MESSAGE
    assert citations[0].department == "general"


def test_vector_thresholds_are_backend_specific(monkeypatch):
    monkeypatch.setenv("QDRANT_SCORE_MIN", "0.5")
    monkeypatch.setenv("CHROMA_DIST_MAX", "0.4")
    items = [
        {"score": 0.49, "distance": 0.39},
        {"score": 0.50, "distance": 0.40},
        {"score": 0.90, "distance": 0.41},
    ]

    assert vector_relevance_filter(items, "qdrant") == items[1:]
    assert vector_relevance_filter(items, "chroma") == items[:2]


def test_diversification_allows_up_to_three_chunks_per_document():
    items = [
        hit("general", f"policy section {index}", chunk_id=f"a-{index}")
        for index in range(4)
    ]
    items.append(
        hit(
            "general",
            "another policy",
            document_id="other-id",
            path="general/other.md",
            chunk_id="b-1",
        )
    )

    diversified = diversify_by_path(items, limit=10, max_per_document=3)
    counts = Counter(item["document_id"] for item in diversified)

    assert counts["general-id"] == 3
    assert counts["other-id"] == 1


def test_citation_ids_match_prompt_order_and_are_not_resorted():
    documents = [
        hit("marketing", "first", path="/app/resources/data/zeta.md", chunk_id="1"),
        hit("marketing", "second", path="/app/resources/data/alpha.md", chunk_id="2"),
    ]

    citations = build_citations(documents)
    returned = citations_for_answer("Second [2], then first [1].", citations)

    assert [citation.citation_id for citation in returned] == [1, 2]
    assert [citation.path for citation in returned] == ["zeta.md", "alpha.md"]
    prompt_context = build_prompt("question", citations)["messages"][1]["content"]
    assert prompt_context.index("[1]") < prompt_context.index("[2]")
    assert validate_answer("Supported [2].", citations) is True
    assert validate_answer("Unsupported [3].", citations) is False


def test_graph_and_rag_share_citation_representation(monkeypatch):
    service = FakeRagService(
        [hit("general", "company policy", path="general/policy.md")]
    )
    monkeypatch.setattr(graph_module, "rag_service", service)
    direct_answer, direct_citations = service.generate("company policy", "employee")

    state = {"query": "company policy", "role": "employee"}
    state = n_intent_guard(state)
    state = n_retrieve(state)
    state = n_generate(state)

    assert state["answer"] == direct_answer
    assert state["citations"] == direct_citations


def test_denial_message_is_identical_in_rag_and_graph():
    service = FakeRagService([])
    rag_answer, _ = service.generate("missing", "employee")
    graph_state = n_fallback({"query": "missing", "role": "employee"})

    assert rag_answer == graph_state["answer"] == DENY_MESSAGE


@pytest.fixture
def document_api(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    (data_dir / "general").mkdir(parents=True)
    (data_dir / "finance").mkdir()
    general_path = data_dir / "general" / "handbook.md"
    finance_path = data_dir / "finance" / "forecast.md"
    general_path.write_text("# Handbook\nVisible policy", encoding="utf-8")
    finance_path.write_text("# Forecast\nRestricted numbers", encoding="utf-8")
    service = DocumentService(data_dir)
    monkeypatch.setattr(main_module, "document_service", service)
    return TestClient(main_module.app), service, finance_path


def test_document_endpoints_require_auth_and_enforce_role(document_api):
    client, service, finance_path = document_api
    unauthenticated = client.get("/documents")
    employee_docs = client.get("/documents", auth=("Emma", "password"))
    finance_id = canonical_document_id(finance_path, service.data_dir)
    forbidden = client.get(f"/documents/{finance_id}", auth=("Emma", "password"))
    authorized = client.get(f"/documents/{finance_id}", auth=("Sam", "financepass"))

    assert unauthenticated.status_code == 401
    assert [doc["department"] for doc in employee_docs.json()] == ["general"]
    assert forbidden.status_code == 404
    assert authorized.status_code == 200
    assert authorized.json()["content"] == "# Forecast\nRestricted numbers"


def test_invalid_basic_auth_is_401(document_api):
    client, _, _ = document_api
    response = client.get("/documents", auth=("Emma", "wrong"))
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Basic"


def test_invalid_configured_user_database_fails_closed(monkeypatch):
    monkeypatch.setenv("BASIC_USERS_JSON", "not-json")
    assert AuthService()._users_db == {}


def test_reindex_is_forbidden_for_non_admin(monkeypatch):
    called = False

    def fake_reindex():
        nonlocal called
        called = True
        return 1

    monkeypatch.setattr(main_module.indexer_service, "reindex", fake_reindex)
    response = TestClient(main_module.app).post(
        "/admin/reindex", auth=("Emma", "password")
    )
    assert response.status_code == 403
    assert called is False


@pytest.mark.parametrize(
    ("status", "expected_ready"),
    [
        ({"ready": False, "indexed_chunks": 0, "vector_db": "chroma", "error": None}, False),
        ({"ready": True, "indexed_chunks": 7, "vector_db": "chroma", "error": None}, True),
    ],
)
def test_health_exposes_empty_and_populated_index(monkeypatch, status, expected_ready):
    monkeypatch.setattr(main_module.indexer_service, "index_status", lambda: status)
    response = TestClient(main_module.app).get("/healthz")
    assert response.status_code == 200
    assert response.json()["index_ready"] is expected_ready
    assert response.json()["indexed_chunks"] == status["indexed_chunks"]

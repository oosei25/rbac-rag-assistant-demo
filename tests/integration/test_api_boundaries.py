from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.services.auth import AuthService
from app.services.documents import DocumentService, canonical_document_id


pytestmark = pytest.mark.integration


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


def test_invalid_basic_auth_is_401_with_challenge(document_api):
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
def test_health_exposes_index_readiness(monkeypatch, status, expected_ready):
    monkeypatch.setattr(main_module.indexer_service, "index_status", lambda: status)

    response = TestClient(main_module.app).get("/healthz")

    assert response.status_code == 200
    assert response.json()["index_ready"] is expected_ready
    assert response.json()["indexed_chunks"] == status["indexed_chunks"]


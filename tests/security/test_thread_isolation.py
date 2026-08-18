from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.services.rag_helpers import DENY_MESSAGE
from evals.runner import evaluate


pytestmark = pytest.mark.security

THREAD_RESULTS = [
    result
    for result in evaluate()["results"]
    if result["suite"] == "thread_isolation"
]


@pytest.mark.parametrize(
    "result",
    THREAD_RESULTS,
    ids=[result["name"] for result in THREAD_RESULTS],
)
def test_thread_key_case(result):
    assert result["passed"], result["violations"]


def test_graph_endpoint_scopes_same_client_thread_to_authenticated_user(monkeypatch):
    class CapturingGraph:
        def __init__(self):
            self.configs = []

        def invoke(self, _state, config):
            self.configs.append(config)
            return {"answer": DENY_MESSAGE, "citations": []}

    graph = CapturingGraph()
    monkeypatch.setattr(main_module, "graph", graph)
    client = TestClient(main_module.app)
    payload = {"message": "general handbook", "thread_id": "shared"}

    employee = client.post("/chat/graph", json=payload, auth=("Emma", "password"))
    finance = client.post("/chat/graph", json=payload, auth=("Sam", "financepass"))

    assert employee.status_code == finance.status_code == 200
    keys = [config["configurable"]["thread_id"] for config in graph.configs]
    assert keys[0] != keys[1]
    assert "Emma" not in keys[0]
    assert "Sam" not in keys[1]


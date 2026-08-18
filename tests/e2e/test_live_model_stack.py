from __future__ import annotations

import base64
import json
import os
import urllib.request

import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.model_dependent]


def _request(url: str, *, payload: dict | None = None) -> dict:
    username = os.getenv("E2E_USERNAME", "Emma")
    password = os.getenv("E2E_PASSWORD", "password")
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def test_live_stack_returns_grounded_general_answer():
    api_url = os.getenv("E2E_API_URL")
    if not api_url:
        pytest.fail(
            "E2E_API_URL is required when explicitly running the model-dependent tier"
        )

    health = _request(f"{api_url.rstrip('/')}/healthz")
    response = _request(
        f"{api_url.rstrip('/')}/chat/rag",
        payload={"message": "What is the employee handbook holiday policy?"},
    )

    assert health["index_ready"] is True
    assert response["answer"]
    assert response["citations"]
    assert {citation["department"] for citation in response["citations"]} == {
        "general"
    }


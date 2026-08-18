from __future__ import annotations

from collections import Counter

import pytest

from app.services.rag_helpers import (
    build_citations,
    build_prompt,
    citations_for_answer,
    diversify_by_path,
    validate_answer,
    vector_relevance_filter,
)


pytestmark = pytest.mark.unit


def _document(index: int, document_id: str = "doc-a") -> dict:
    return {
        "department": "general",
        "document_id": document_id,
        "chunk_id": f"chunk-{index}",
        "path": f"general/{document_id}.md",
        "title": "Policy",
        "section": f"Section {index}",
        "score": 0.9,
        "text": f"Policy text {index}",
    }


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


def test_diversification_caps_chunks_per_document():
    items = [_document(index) for index in range(4)]
    items.append(_document(5, "doc-b"))

    diversified = diversify_by_path(items, limit=10, max_per_document=3)
    counts = Counter(item["document_id"] for item in diversified)

    assert counts == {"doc-a": 3, "doc-b": 1}


def test_citation_ids_match_prompt_order_and_answer_subset():
    documents = [
        {**_document(1, "zeta"), "path": "/app/resources/data/general/zeta.md"},
        {**_document(2, "alpha"), "path": "/app/resources/data/general/alpha.md"},
    ]

    citations = build_citations(documents)
    returned = citations_for_answer("Second [2], then first [1].", citations)
    context = build_prompt("question", citations)["messages"][1]["content"]

    assert [citation.citation_id for citation in returned] == [1, 2]
    assert [citation.path for citation in returned] == ["general/zeta.md", "general/alpha.md"]
    assert context.index("[1]") < context.index("[2]")
    assert validate_answer("Supported [2].", citations)
    assert not validate_answer("Unsupported [3].", citations)
    assert not validate_answer("Uncited claim.", citations)


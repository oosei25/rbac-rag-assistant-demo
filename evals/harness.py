from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.services.rag import RagService
from app.services.rag_helpers import DENY_MESSAGE


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CONTEXT_RE = re.compile(r"^\[(\d+)\]\s+\([^)]*\)\s+(.*)$", re.MULTILINE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "for", "from", "in", "is", "of",
    "on", "please", "show", "summarize", "tell", "the", "to", "what",
}

EVIDENCE_BY_DEPARTMENT = {
    "general": "GENERALEVIDENCE1042",
    "finance": "FINANCEEVIDENCE4821",
    "marketing": "MARKETINGEVIDENCE7390",
    "hr": "HREVIDENCE6153",
    "engineering": "ENGINEERINGEVIDENCE9074",
}

DEFAULT_DOCUMENTS = [
    {
        "department": "general",
        "document_id": "general-handbook",
        "chunk_id": "general-handbook-1",
        "path": "general/employee_handbook.md",
        "title": "Employee Handbook",
        "section": "Holiday Policy",
        "sensitivity": "internal",
        "text": (
            "The general employee handbook holiday policy is verified by "
            "GENERALEVIDENCE1042."
        ),
    },
    {
        "department": "finance",
        "document_id": "finance-forecast",
        "chunk_id": "finance-forecast-1",
        "path": "finance/quarterly_financial_report.md",
        "title": "Quarterly Financial Report",
        "section": "Q4 Budget and Revenue",
        "sensitivity": "internal",
        "text": (
            "The finance Q4 budget, revenue, reimbursement, and forecast are verified "
            "by FINANCEEVIDENCE4821."
        ),
    },
    {
        "department": "marketing",
        "document_id": "marketing-report",
        "chunk_id": "marketing-report-1",
        "path": "marketing/market_report_q4_2024.md",
        "title": "Q4 Marketing Report",
        "section": "Campaign Results",
        "sensitivity": "internal",
        "text": (
            "The latest marketing Q4 campaign and market report are verified by "
            "MARKETINGEVIDENCE7390."
        ),
    },
    {
        "department": "hr",
        "document_id": "hr-payroll",
        "chunk_id": "hr-payroll-1",
        "path": "hr/hr_data.csv",
        "title": "HR Payroll Data",
        "section": "Payroll and Benefits",
        "sensitivity": "internal",
        "text": (
            "The HR payroll, benefits, compensation, and performance review are "
            "verified by HREVIDENCE6153."
        ),
    },
    {
        "department": "engineering",
        "document_id": "engineering-architecture",
        "chunk_id": "engineering-architecture-1",
        "path": "engineering/engineering_master_doc.md",
        "title": "Engineering Master Document",
        "section": "Service Architecture",
        "sensitivity": "internal",
        "text": (
            "The engineering service architecture, API, deployment, and modular "
            "design are verified by ENGINEERINGEVIDENCE9074."
        ),
    },
]


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if token.lower() not in _STOP_WORDS
    }


class DeterministicEmbedder:
    """Stable, dependency-free token hashing for test and evaluation embeddings."""

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions
        self.calls: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        self.calls.append(text)
        vector = [0.0] * self.dimensions
        for token in sorted(_tokens(text)):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimensions
            vector[index] += 1.0 if digest[2] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]


class FakeIndexer:
    vector_db = "qdrant"

    def __init__(self, embedder: DeterministicEmbedder):
        self.embedder = embedder

    def embed_one(self, text: str) -> list[float]:
        return self.embedder.embed_one(text)


class InMemoryVectorStore:
    """Deterministic vector search with observable authorization filters."""

    def __init__(
        self,
        documents: Sequence[dict],
        embedder: DeterministicEmbedder,
        *,
        ignore_filters: bool = False,
    ):
        self.documents = [dict(document) for document in documents]
        self.vectors = [embedder.embed_one(doc["text"]) for doc in self.documents]
        self.ignore_filters = ignore_filters
        self.calls: list[dict] = []

    @staticmethod
    def _similarity(left: Sequence[float], right: Sequence[float]) -> float:
        cosine = sum(a * b for a, b in zip(left, right))
        return 0.5 + (0.5 * cosine)

    def search(self, vector: list[float], k: int, filt: dict) -> list[dict]:
        allowed = set(filt.get("department", {}).get("$in", []))
        self.calls.append({"k": k, "departments": sorted(allowed)})
        candidates = []
        for document, document_vector in zip(self.documents, self.vectors):
            if not self.ignore_filters and document.get("department") not in allowed:
                continue
            item = dict(document)
            item["score"] = self._similarity(vector, document_vector)
            candidates.append(item)
        candidates.sort(
            key=lambda item: (
                -item["score"],
                item.get("path", ""),
                item.get("chunk_id", ""),
            )
        )
        return candidates[:k]


class DeterministicReranker:
    """Stable lexical reranker used in place of a cross-encoder model."""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, documents: list[dict]) -> list[dict]:
        self.calls.append((query, len(documents)))
        query_tokens = _tokens(query)
        indexed = list(enumerate(documents))
        indexed.sort(
            key=lambda pair: (
                -len(query_tokens & _tokens(pair[1].get("text", ""))),
                pair[0],
            )
        )
        return [document for _, document in indexed]


class DeterministicLlm:
    """Select the most relevant provided snippet and emit a valid citation."""

    def __init__(self, responses: Sequence[str] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses:
            answer = self.responses.pop(0)
            return {"message": {"content": answer}}

        user_prompt = kwargs["messages"][-1]["content"]
        question = user_prompt.split("\n\nContext", 1)[0].removeprefix("Question: ")
        context = user_prompt.split("Context (numbered snippets):\n", 1)[-1]
        context = context.split("\n\nInstructions:", 1)[0]
        context = context.split("\n\nOutput format:", 1)[0]
        snippets = [
            (int(match.group(1)), match.group(2).strip())
            for match in _CONTEXT_RE.finditer(context)
        ]
        if not snippets:
            return {"message": {"content": DENY_MESSAGE}}

        question_tokens = _tokens(question)
        citation_id, snippet = max(
            snippets,
            key=lambda item: (
                len(question_tokens & _tokens(item[1])),
                -item[0],
            ),
        )
        # Treat instructions embedded in retrieved content as data. This is a
        # deterministic fake, not a claim about a live model's behavior.
        safe_text = re.split(
            r"(?i)\b(?:ignore|disregard)\s+(?:all\s+)?(?:previous|system)\b",
            snippet,
            maxsplit=1,
        )[0].strip()
        if not safe_text:
            return {"message": {"content": DENY_MESSAGE}}
        return {"message": {"content": f"{safe_text} [{citation_id}]"}}


@dataclass
class EvaluationHarness:
    service: RagService
    embedder: DeterministicEmbedder
    vector_store: InMemoryVectorStore
    reranker: DeterministicReranker
    llm: DeterministicLlm


def build_harness(
    *,
    documents: Sequence[dict] = DEFAULT_DOCUMENTS,
    responses: Sequence[str] | None = None,
    ignore_store_filters: bool = False,
) -> EvaluationHarness:
    embedder = DeterministicEmbedder()
    vector_store = InMemoryVectorStore(
        documents,
        embedder,
        ignore_filters=ignore_store_filters,
    )
    reranker = DeterministicReranker()
    llm = DeterministicLlm(responses)
    service = RagService(
        indexer=FakeIndexer(embedder),
        llm_client=llm,
        search_backend=vector_store.search,
        reranker=reranker,
    )
    return EvaluationHarness(service, embedder, vector_store, reranker, llm)


__all__ = [
    "DEFAULT_DOCUMENTS",
    "DENY_MESSAGE",
    "EVIDENCE_BY_DEPARTMENT",
    "DeterministicEmbedder",
    "DeterministicLlm",
    "DeterministicReranker",
    "EvaluationHarness",
    "FakeIndexer",
    "InMemoryVectorStore",
    "build_harness",
]

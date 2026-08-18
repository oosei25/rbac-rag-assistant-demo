from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from ollama import Client

from app.policy import allowed_departments, infer_requested_departments
from app.schemas import Citation
from app.services.indexer import indexer_service
from app.services.rag_helpers import (
    DENY_MESSAGE,
    build_citations,
    build_prompt,
    chroma_search,
    citations_for_answer,
    cross_encoder_rerank,
    diversify_by_path,
    keyword_slice_answer,
    lexical_filter,
    lexical_rerank,
    looks_like_deny,
    normalize_hits,
    qdrant_search,
    rewrite_query,
    sanitize_answer,
    validate_answer,
    vector_relevance_filter,
)


class RagBackendError(RuntimeError):
    """Raised when an allowed request cannot reach the generation backend."""


class RagService:
    """Coordinates an authorization-first retrieval and generation flow."""

    def __init__(self, indexer=indexer_service, llm_client=None):
        self.indexer = indexer
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
        self.top_k = int(os.getenv("TOP_K", "6"))
        self.intent_narrowing = os.getenv("INTENT_NARROWING", "1") == "1"
        self.debug = os.getenv("DEBUG_RAG", "0") == "1"
        self.use_keyword_fallback = os.getenv("USE_KEYWORD_FALLBACK", "1") == "1"
        self.passage_selection = os.getenv("PASSAGE_SELECTION", "0") == "1"
        self.passage_selection_k = int(os.getenv("PASSAGE_SELECTION_K", "5"))
        self.client = llm_client or Client(host=self.ollama_host)
        self.deny_msg = DENY_MESSAGE
        self.vector_db = self.indexer.vector_db
        self._passage_instr = (
            "You are a retrieval assistant.\n"
            "Given a question and numbered passages, choose the fewest passages "
            "that directly answer the question.\n"
            "Return only a JSON array of passage numbers, such as [1,3]."
        )

    def _select_passages_llm(self, query_text: str, docs: List[dict]) -> List[dict]:
        if len(docs) <= self.passage_selection_k:
            return docs

        context = "\n\n".join(
            f"[{index}] {(doc.get('text') or '').replace(chr(10), ' ')[:500]}"
            for index, doc in enumerate(docs, 1)
        )
        messages = [
            {"role": "system", "content": self._passage_instr},
            {
                "role": "user",
                "content": (
                    f"Question: {query_text}\n\nPassages:\n{context}\n\n"
                    f"Return up to {self.passage_selection_k} passage numbers."
                ),
            },
        ]

        try:
            response = self.client.chat(
                model=self.ollama_model,
                messages=messages,
                options={"temperature": 0, "num_predict": 60},
            )
            raw = (response.get("message", {}) or {}).get("content", "")
        except Exception as exc:
            if self.debug:
                print("[rag] passage selection failed:", exc)
            return docs[: self.passage_selection_k]

        selected_ids = {
            int(match)
            for match in re.findall(r"\d+", raw)
            if 1 <= int(match) <= len(docs)
        }
        # The selector may choose passages, but it cannot reorder retrieval.
        selected = [
            doc for index, doc in enumerate(docs, 1) if index in selected_ids
        ][: self.passage_selection_k]
        return selected or docs[: self.passage_selection_k]

    def _search_backend(self, vector, k: int, filt: dict):
        if self.vector_db == "qdrant":
            return qdrant_search(vector, k, filt)
        return chroma_search(vector, k, filt)

    @staticmethod
    def _dept_filter(departments: set[str]) -> dict:
        return {"department": {"$in": sorted(departments)}}

    def retrieve(
        self, query_text: str, role: str, k: Optional[int] = None
    ) -> List[dict]:
        k = k or self.top_k
        allowed = set(allowed_departments(role))

        # This check precedes embedding and vector search. An unknown role can
        # never turn an empty filter into an unfiltered backend request.
        if not allowed:
            return []

        retrieval_departments = allowed
        if self.intent_narrowing:
            requested = infer_requested_departments(query_text)
            requested_allowed = requested & allowed
            if requested != {"general"} and requested_allowed:
                retrieval_departments = requested_allowed

        vector = self.indexer.embed_one(rewrite_query(query_text))
        fetch_k = max(k * 4, 12)
        hits = list(
            self._search_backend(
                vector,
                fetch_k,
                self._dept_filter(retrieval_departments),
            )
        )
        if retrieval_departments != allowed:
            # Intent changes priority, not eligibility. A broad authorized
            # search prevents an imperfect classifier from causing denial.
            hits.extend(
                self._search_backend(
                    vector,
                    fetch_k,
                    self._dept_filter(allowed),
                )
            )
        candidates = normalize_hits(hits)
        candidates = vector_relevance_filter(candidates, self.vector_db)

        # Defense in depth: reject backend results outside the authenticated
        # policy even if a vector backend ignores or misapplies its filter.
        candidates = [
            item for item in candidates if item.get("department") in allowed
        ]
        candidates = lexical_filter(query_text, candidates)
        candidates = diversify_by_path(candidates, limit=fetch_k)
        candidates = lexical_rerank(query_text, candidates, boost=0.25)
        candidates = cross_encoder_rerank(query_text, candidates)
        return candidates[:k]

    def generate_from_documents(
        self, query_text: str, docs: List[dict]
    ) -> Tuple[str, List[Citation]]:
        if not docs:
            return self.deny_msg, []
        if self.passage_selection:
            docs = self._select_passages_llm(query_text, docs)

        citations = build_citations(docs)
        prompt = build_prompt(query_text, citations)
        try:
            response = self.client.chat(
                model=self.ollama_model,
                messages=prompt["messages"],
                options=prompt["options"],
            )
            raw_answer = (response.get("message", {}) or {}).get("content", "").strip()
        except Exception as exc:
            if self.debug:
                print("[rag] generation failed:", exc)
            raise RagBackendError("Model backend unavailable") from exc

        answer = sanitize_answer(raw_answer)
        if looks_like_deny(answer):
            if self.use_keyword_fallback:
                fallback = keyword_slice_answer(query_text, docs)
                if fallback:
                    answer = sanitize_answer(fallback)
                else:
                    return self.deny_msg, []
            else:
                return self.deny_msg, []

        if not validate_answer(answer, citations):
            return self.deny_msg, []
        return answer, citations_for_answer(answer, citations)

    def generate(self, query_text: str, role: str) -> Tuple[str, List[Citation]]:
        return self.generate_from_documents(
            query_text,
            self.retrieve(query_text, role),
        )


rag_service = RagService()


def retrieve(query_text: str, role: str, k: Optional[int] = None) -> List[dict]:
    return rag_service.retrieve(query_text, role, k)


def generate(query_text: str, role: str) -> Tuple[str, List[Citation]]:
    return rag_service.generate(query_text, role)


__all__ = ["RagBackendError", "RagService", "rag_service", "retrieve", "generate"]

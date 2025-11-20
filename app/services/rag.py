import os
import re
from typing import List, Optional, Tuple

from ollama import Client

from app.policy import allowed_departments, infer_requested_departments
from app.services.indexer import indexer_service
from app.services.rag_helpers import (
    build_prompt,
    chroma_search,
    cross_encoder_rerank,
    diversify_by_path,
    keyword_slice_answer,
    lexical_rerank,
    looks_like_deny,
    postfilter_relaxed,
    postfilter_strict,
    qdrant_search,
    rewrite_query,
    sanitize_answer,
    validate_answer,
)


class RagService:
    """Coordinates retrieval + generation flow for the chatbot."""

    def __init__(self, indexer=indexer_service):
        self.indexer = indexer
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
        self.top_k = int(os.getenv("TOP_K", "6"))
        self.rbac_intent = os.getenv("RBAC_INTENT", "0") == "1"
        self.rbac_intent_soft = os.getenv("RBAC_INTENT_SOFT", "0") == "1"
        self.debug = os.getenv("DEBUG_RAG", "0") == "1"
        self.use_keyword_fallback = os.getenv("USE_KEYWORD_FALLBACK", "0") == "1"
        self.passage_selection = os.getenv("PASSAGE_SELECTION", "0") == "1"
        self.passage_selection_k = int(os.getenv("PASSAGE_SELECTION_K", "5"))
        self.client = Client(host=self.ollama_host)
        self.deny_msg = "I don’t have enough information in your accessible documents."
        self.vector_db = self.indexer.vector_db
        self._passage_instr = (
                "You are a retrieval assistant.\n"
                "Given a question and numbered passages, choose the FEWEST passages that, "
                "together, directly answer the question.\n"
                "Only keep passages that are clearly useful.\n"
                "Return ONLY a JSON array of snippet numbers, like: [1,3,4]\n"
                "Do NOT include any explanation or extra text."
        )

    def _select_passages_llm(self, query_text: str, docs: List[dict]) -> List[dict]:
        """
        Ask the model to trim the retrieved docs down to the most relevant passages.
        Keeps behaviour deterministic by falling back to the top-k slice if parsing fails.
        """
        if len(docs) <= self.passage_selection_k:
            return docs

        snippets = []
        for i, d in enumerate(docs, 1):
            text = (d.get("text") or "").replace("\n", " ")
            snippets.append(f"[{i}] {text[:500]}")
        context = "\n\n".join(snippets)

        messages = [
            {"role": "system", "content": self._passage_instr},
            {
                "role": "user",
                "content": (
                    f"Question: {query_text}\n\n"
                    f"Passages:\n{context}\n\n"
                    f"Return up to {self.passage_selection_k} snippet numbers."
                ),
            },
        ]

        try:
            resp = self.client.chat(
                model=self.ollama_model,
                messages=messages,
                options={"temperature": 0, "num_predict": 60},
            )
            raw = (resp.get("message", {}) or {}).get("content", "")
        except Exception as e:
            if self.debug:
                print("[rag] passage_select error:", e)
            return docs[: self.passage_selection_k]

        nums = []
        for match in re.findall(r"\d+", raw):
            try:
                idx = int(match)
            except ValueError:
                continue
            if 1 <= idx <= len(docs):
                nums.append(idx)
        # Deduplicate while preserving order
        seen = set()
        selected: List[dict] = []
        for idx in nums:
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(docs[idx - 1])
            if len(selected) >= self.passage_selection_k:
                break

        if not selected and self.debug:
            print("[rag] passage_select fallback, raw response:", raw[:200])

        return selected or docs[: self.passage_selection_k]

    def _search_backend(self, vec, k: int, filt: Optional[dict]):
        if self.vector_db == "qdrant":
            return qdrant_search(vec, k, filt)
        return chroma_search(vec, k, filt)

    @staticmethod
    def _dept_filter(depts: set[str] | List[str]) -> Optional[dict]:
        if not depts:
            return None
        return {"department": {"$in": list(depts)}}

    def retrieve(
        self, query_text: str, role: str, k: Optional[int] = None
    ) -> List[dict]:
        k = k or self.top_k
        q_for_embed = rewrite_query(query_text)

        requested = infer_requested_departments(query_text)
        allowed = set(allowed_departments(role))

        if self.rbac_intent and (requested - allowed) and requested != {"general"}:
            if not self.rbac_intent_soft:
                return []
            inter = requested & allowed
            if inter:
                filt_allowed = self._dept_filter(inter)
            else:
                filt_allowed = self._dept_filter({"general"})
        else:
            filt_allowed = self._dept_filter(allowed)

        vec = self.indexer.embed_one(q_for_embed)

        hits = self._search_backend(vec, max(k, 8), filt_allowed)
        cand = postfilter_strict(query_text, hits)

        if len(cand) < k:
            hits2 = self._search_backend(vec, max(k * 3, 12), filt_allowed)
            cand += postfilter_relaxed(query_text, hits2)

        if not cand:
            if not (self.rbac_intent and self.rbac_intent_soft):
                filt_gen = self._dept_filter({"general"})
                hits_fb = self._search_backend(vec, max(k * 3, 12), filt_gen)
                cand = postfilter_relaxed(query_text, hits_fb)

        if not cand:
            return []

        cand = diversify_by_path(cand, limit=max(k * 2, k))
        cand = lexical_rerank(query_text, cand, boost=0.25)
        cand = cross_encoder_rerank(query_text, cand)
        return cand[: max(k * 2, k)]

    def generate(self, query_text: str, role: str) -> Tuple[str, List[str]]:
        docs = self.retrieve(query_text, role)
        if not docs:
            return self.deny_msg, []
        if self.passage_selection:
            docs = self._select_passages_llm(query_text, docs)

        prompt = build_prompt(query_text, docs, model=self.ollama_model)
        try:
            resp = self.client.chat(
                model=self.ollama_model,
                messages=prompt["messages"],
                options=prompt["options"],
            )
            msg_raw = (resp.get("message", {}) or {}).get("content", "").strip()
        except Exception as e:
            return f"I couldn’t reach the model backend: {e}", []

        if self.debug:
            print("[rag] raw:", msg_raw[:200])
        msg = sanitize_answer(msg_raw)
        if self.debug:
            print("[rag] sanitize:", msg[:200])

        if looks_like_deny(msg) and self.use_keyword_fallback:
            alt = keyword_slice_answer(query_text, docs)
            if alt:
                alt_sanitized = sanitize_answer(alt)
                if self.debug:
                    print("[rag] fallback: using keyword_slice_answer")
                    print("[rag] fallback:", alt_sanitized[:200])
                srcs = [d.get("path", "") for d in docs if d.get("path")]
                return alt_sanitized, sorted(set(srcs))
            if self.debug:
                print("[rag] fallback: using default deny")
            return self.deny_msg, []

        if not validate_answer(msg, docs):
            return self.deny_msg, []

        if self.debug:
            print("[rag] validate:", msg[:200])

        sources = [d.get("path", "") for d in docs if d.get("path")]
        return msg, sorted(set(sources))


rag_service = RagService()


def retrieve(query_text: str, role: str, k: Optional[int] = None) -> List[dict]:
    return rag_service.retrieve(query_text, role, k)


def generate(query_text: str, role: str) -> Tuple[str, List[str]]:
    return rag_service.generate(query_text, role)


__all__ = ["RagService", "rag_service", "retrieve", "generate"]

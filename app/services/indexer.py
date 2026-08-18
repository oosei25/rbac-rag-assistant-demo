
import glob
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

import yaml

from app.services.documents import (
    canonical_document_id,
    department_from_path,
    document_title,
    safe_document_path,
)
from app.utils.chunk import chunk_text
from app.utils.io import read_file


class IndexerService:
    """Encapsulates embedding + vector store plumbing for reuse."""

    def __init__(self):
        self.data_dir = Path(os.getenv("DATA_DIR", "resources/data"))
        self.vector_db = os.getenv("VECTOR_DB", "chroma")
        self.embed_backend = os.getenv("EMBED_BACKEND", "local")
        self.st_model = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.index_path = os.getenv("INDEX_PATH", ".local_index/chroma")
        self.collection_name = "company_docs"

        self._openai_client = None
        self._st_model = None
        self.embed_model = None
        self.embed_dim = 0
        self._chroma_client = None
        self._chroma_collection = None
        self._qdrant_client = None

        # Heavy embedding models and backend connections are initialized on
        # first use, keeping imports and deterministic tests infrastructure-free.

    # -------Embeds
    def _init_embedder(self) -> None:
        if self.embed_model is not None:
            return
        if self.embed_backend == "openai":
            from openai import OpenAI

            self._openai_client = OpenAI()
            self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
            self.embed_dim = 1536  # text-embedding-3-small
        else:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.st_model)
            self.embed_model = self.st_model
            self.embed_dim = self._st_model.get_sentence_embedding_dimension()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._init_embedder()
        if self.embed_backend == "openai":
            resp = self._openai_client.embeddings.create(
                model=self.embed_model,
                input=texts,
            )
            return [d.embedding for d in resp.data]

        embs = self._st_model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [e.tolist() for e in embs]

    def embed_one(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    # ------Vector DB
    def _init_vector_backend(self) -> None:
        if self._qdrant_client is not None or self._chroma_client is not None:
            return
        if self.vector_db == "qdrant":
            from qdrant_client import QdrantClient

            self._qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_URL", "http://qdrant:6333")
            )
        else:
            import chromadb

            self._chroma_client = chromadb.PersistentClient(path=self.index_path)
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                self.collection_name, metadata={"hnsw:space": "cosine"}
            )

    def _reset_collection(self) -> None:
        self._init_vector_backend()
        if self.vector_db != "qdrant":
            self._chroma_client.delete_collection(self.collection_name)
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            return
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        names = [c.name for c in self._qdrant_client.get_collections().collections]
        if self.collection_name in names:
            self._qdrant_client.delete_collection(self.collection_name)
        self._qdrant_client.create_collection(
            self.collection_name,
            vectors_config=VectorParams(size=self.embed_dim, distance=Distance.COSINE),
        )
        for field in ("document_id", "department"):
            try:
                self._qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                print(f"[indexer] create_payload_index({field}) skipped: {exc}")

    def _upsert_batch(self, points: List[Dict[str, Any]]) -> None:
        self._init_vector_backend()
        if self.vector_db == "qdrant":
            from uuid import UUID, uuid4
            from qdrant_client.models import PointStruct

            qpoints = []
            for p in points:
                pid = p.get("id")
                if isinstance(pid, int):
                    pid_norm = pid
                else:
                    try:
                        pid_norm = str(UUID(str(pid)))
                    except Exception:
                        pid_norm = str(uuid4())
                qpoints.append(
                    PointStruct(
                        id=pid_norm, vector=p["vector"], payload=p["payload"]
                    )
                )
            self._qdrant_client.upsert(
                collection_name=self.collection_name, points=qpoints, wait=True
            )
            return

        ids = [p["id"] for p in points]
        embs = [p["vector"] for p in points]
        metas = [p["payload"] for p in points]
        self._chroma_collection.upsert(ids=ids, embeddings=embs, metadatas=metas)

    # --------Helpers
    def _sidecar_for(self, p: Path) -> Dict[str, Any]:
        cand = self.data_dir.parent / "metadata" / f"{p.stem}.yml"
        if cand.exists():
            return yaml.safe_load(cand.read_text()) or {}
        return {}

    def _doc_meta(self, p: Path, content: str) -> Dict[str, Any]:
        base = {
            "path": safe_document_path(p, self.data_dir),
            "department": department_from_path(p, self.data_dir),
            "sensitivity": "internal",
            "tenant_id": "default",
            "title": document_title(p, content),
            "document_id": canonical_document_id(p, self.data_dir),
            "source_url": None,
            "version": "v1",
        }
        sidecar = self._sidecar_for(p)
        for field in ("sensitivity", "tenant_id", "title", "source_url", "version"):
            if field in sidecar:
                base[field] = sidecar[field]
        return base

    @staticmethod
    def _stable_chunk_id(doc_id: str, chunk_text: str, idx: int) -> str:
        content_hash = hashlib.sha1(
            f"{doc_id}|{idx}|".encode("utf-8") + chunk_text.encode("utf-8")
        ).hexdigest()
        return str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"rbac-rag-chunk:{content_hash}")
        )

    # ------ Reindex
    def reindex(self, batch_size: int = 64) -> int:
        self._init_embedder()
        files = sorted(
            Path(f) for f in glob.glob(str(self.data_dir / "**/*.*"), recursive=True)
        )
        points: List[Dict[str, Any]] = []

        for fp in files:
            text = read_file(fp)
            if not text:
                continue

            meta = self._doc_meta(fp, text)

            for idx, ch in enumerate(chunk_text(text)):
                ch_text = ch["text"] if isinstance(ch, dict) else str(ch)
                section = (
                    (ch.get("section") if isinstance(ch, dict) else None)
                    or meta.get("title")
                )

                vec = self.embed_one(ch_text)
                sid = self._stable_chunk_id(meta["document_id"], ch_text, idx)

                payload = {
                    key: value
                    for key, value in {
                        **meta,
                        "chunk_id": sid,
                        "text": ch_text,
                        "section": section,
                    }.items()
                    if value is not None
                }

                p = fp.name.lower()
                for q in ("q4", "q3", "q2", "q1"):
                    if q in p:
                        payload["quarter"] = q
                        break
                m = re.search(r"(20\d{2})", p)
                if m:
                    payload["year"] = m.group(1)

                points.append({"id": sid, "vector": vec, "payload": payload})

        self._reset_collection()
        for start in range(0, len(points), batch_size):
            self._upsert_batch(points[start : start + batch_size])
        return len(points)

    def index_status(self) -> Dict[str, Any]:
        """Return a backend-neutral readiness snapshot without mutating state."""
        try:
            self._init_vector_backend()
            if self.vector_db == "qdrant":
                names = {
                    collection.name
                    for collection in self._qdrant_client.get_collections().collections
                }
                if self.collection_name not in names:
                    count = 0
                else:
                    info = self._qdrant_client.get_collection(self.collection_name)
                    count = int(info.points_count or 0)
            else:
                count = int(self._chroma_collection.count())
            return {
                "ready": count > 0,
                "indexed_chunks": count,
                "vector_db": self.vector_db,
                "error": None,
            }
        except Exception as exc:
            return {
                "ready": False,
                "indexed_chunks": 0,
                "vector_db": self.vector_db,
                "error": type(exc).__name__,
            }


indexer_service = IndexerService()


def reindex(batch_size: int = 64) -> int:
    return indexer_service.reindex(batch_size)


def embed_one(text: str) -> List[float]:
    return indexer_service.embed_one(text)


def embed_many(texts: List[str]) -> List[List[float]]:
    return indexer_service.embed_many(texts)


VECTOR_DB = indexer_service.vector_db
INDEX_PATH = indexer_service.index_path

__all__ = [
    "IndexerService",
    "indexer_service",
    "reindex",
    "embed_one",
    "embed_many",
    "VECTOR_DB",
    "INDEX_PATH",
]

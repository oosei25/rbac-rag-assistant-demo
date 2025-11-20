
import os, glob, uuid, yaml, hashlib
import re
from pathlib import Path
from typing import Any, Dict, List

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

        self._init_embedder()
        self._init_vector_backend()

    # -------Embeds
    def _init_embedder(self) -> None:
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

    def _ensure_collection(self) -> None:
        if self.vector_db != "qdrant":
            return
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        try:
            names = [c.name for c in self._qdrant_client.get_collections().collections]
        except Exception:
            names = []
        if self.collection_name not in names:
            self._qdrant_client.recreate_collection(
                self.collection_name,
                vectors_config=VectorParams(size=self.embed_dim, distance=Distance.COSINE),
            )
            for field in ("doc_id", "department"):
                try:
                    self._qdrant_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception as e:
                    print(f"[indexer] create_payload_index({field}) skipped: {e}")

    def _delete_doc(self, doc_id: str) -> None:
        if self.vector_db == "qdrant":
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            try:
                self._qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                    ),
                    wait=True,
                )
            except Exception as e:
                print(f"[indexer] qdrant delete_doc({doc_id}) failed: {e}")
            return

        try:
            self._chroma_collection.delete(where={"doc_id": doc_id})
        except Exception as e:
            print(f"[indexer] chroma delete_doc({doc_id}) failed: {e}")

    def _upsert_batch(self, points: List[Dict[str, Any]]) -> None:
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
    @staticmethod
    def _dept_from_path(p: Path) -> str:
        try:
            i = p.parts.index("data")
            return p.parts[i + 1]
        except ValueError:
            return "general"

    def _sidecar_for(self, p: Path) -> Dict[str, Any]:
        cand = self.data_dir.parent / "metadata" / f"{p.stem}.yml"
        if cand.exists():
            return yaml.safe_load(cand.read_text()) or {}
        return {}

    def _doc_meta(self, p: Path) -> Dict[str, Any]:
        base = {
            "path": str(p),
            "department": self._dept_from_path(p),
            "sensitivity": "internal",
            "tenant_id": "default",
            "title": p.name,
            "doc_id": str(uuid.uuid5(uuid.NAMESPACE_URL, str(p.resolve()))),
            "source_url": None,
            "version": "v1",
        }
        base.update(self._sidecar_for(p))
        return base

    @staticmethod
    def _stable_chunk_id(doc_id: str, chunk_text: str, idx: int) -> str:
        h = hashlib.sha1(
            f"{doc_id}|{idx}|".encode("utf-8") + chunk_text.encode("utf-8")
        ).hexdigest()
        return h

    # ------ Reindex
    def reindex(self, batch_size: int = 64) -> int:
        self._ensure_collection()
        files = [
            Path(f) for f in glob.glob(str(self.data_dir / "**/*.*"), recursive=True)
        ]
        n, batch = 0, []

        for fp in files:
            text = read_file(fp)
            if not text:
                continue

            meta = self._doc_meta(fp)
            try:
                self._delete_doc(meta["doc_id"])
            except Exception as e:
                print(f"[indexer] delete_doc failed for {meta['doc_id']}: {e}")

            for idx, ch in enumerate(chunk_text(text)):
                ch_text = ch["text"] if isinstance(ch, dict) else str(ch)
                section = (
                    (ch.get("section") if isinstance(ch, dict) else None)
                    or meta.get("title")
                )

                vec = self.embed_one(ch_text)
                sid = self._stable_chunk_id(meta["doc_id"], ch_text, idx)

                payload = {**meta, "text": ch_text, "section": section}

                p = fp.name.lower()
                for q in ("q4", "q3", "q2", "q1"):
                    if q in p:
                        payload["quarter"] = q
                        break
                m = re.search(r"(20\d{2})", p)
                if m:
                    payload["year"] = m.group(1)

                batch.append({"id": sid, "vector": vec, "payload": payload})

                if len(batch) >= batch_size:
                    self._upsert_batch(batch)
                    n += len(batch)
                    batch.clear()

        if batch:
            self._upsert_batch(batch)
            n += len(batch)
        return n


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

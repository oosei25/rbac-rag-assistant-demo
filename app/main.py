from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from ollama import Client
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentDetail,
    DocumentSummary,
    HealthResponse,
)
from app.services.auth import auth_service
from app.services.documents import DocumentService
from app.services.indexer import indexer_service
from app.services.rag import rag_service
from app.services.threads import scoped_thread_id
from app.graph.rag_graph import build_graph

graph = build_graph()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.indexing_error = None
    if os.getenv("AUTO_INDEX", "1") == "1":
        for attempt in range(1, 6):
            try:
                count = indexer_service.reindex()
                print(f"[startup] indexed chunks: {count}")
                if count:
                    app.state.indexing_error = None
                    break
                app.state.indexing_error = "No indexable documents found"
            except Exception as exc:
                print(f"[startup] auto-index attempt {attempt} failed:", exc)
                app.state.indexing_error = type(exc).__name__
            if attempt < 5:
                await asyncio.sleep(2)

    # Warm the LLM once so first user call isn't slow
    try:
        c = Client(host=os.getenv("OLLAMA_HOST", "http://ollama:11434"))
        c.chat(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
            messages=[{"role": "user", "content": "."}],
            options={"num_predict": 1},
        )
        print("[startup] LLM warmed")
    except Exception as e:
        print("[startup] LLM warm-up skipped:", e)

    yield

app = FastAPI(title="RAG-RBAC Chatbot", lifespan=lifespan)
app.state.indexing_error = None
document_service = DocumentService(indexer_service.data_dir)

_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Routes ---
@app.get("/login")
def login(user=Depends(auth_service.authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}

@app.get("/test")
def test(user=Depends(auth_service.authenticate)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}

@app.post("/admin/reindex")
def admin_reindex(_user=Depends(auth_service.require_roles("engineering", "clevel"))):
    n = indexer_service.reindex()
    return {"indexed_chunks": n}

@app.get("/healthz", response_model=HealthResponse)
def healthz():
    status = indexer_service.index_status()
    return HealthResponse(
        ok=True,
        index_ready=status["ready"],
        indexed_chunks=status["indexed_chunks"],
        vector_db=status["vector_db"],
        indexing_error=app.state.indexing_error or status["error"],
    )

@app.get("/version")
def version():
    return {
        "ollama_model": rag_service.ollama_model,
        "vector_db": indexer_service.vector_db,
    }


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents(user=Depends(auth_service.authenticate)):
    return document_service.list_for_role(user["role"])


@app.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str, user=Depends(auth_service.authenticate)):
    document = document_service.get_for_role(document_id, user["role"])
    if document is None:
        # Use the same response for missing and unauthorized IDs so callers
        # cannot enumerate the existence of restricted documents.
        raise HTTPException(status_code=404, detail="Document not found")
    return document


# ---RAG endpoint ----
@app.post("/chat/rag", response_model=ChatResponse)
def chat_rag(body: ChatRequest, user=Depends(auth_service.authenticate)):
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")
    try:
        answer, citations = rag_service.generate(body.message, user["role"])
        return ChatResponse(answer=answer, citations=citations)
    except Exception as e:
        # Avoids leaking internals to clients
        raise HTTPException(status_code=500, detail="RAG pipeline error.") from e
 
# ---LangGraph endpoint ----
@app.post("/chat/graph", response_model=ChatResponse)
def chat_graph(body: ChatRequest, user=Depends(auth_service.authenticate)):
    if graph is None:
        return ChatResponse(answer="Graph pipeline is disabled on the server.", citations=[])
    # Client IDs are namespaced by the authenticated principal before they
    # reach the checkpointer, preventing cross-user checkpoint collisions.
    tid = scoped_thread_id(user["username"], body.thread_id)
    cfg = {"configurable": {"thread_id": tid, "checkpoint_ns": "default"}}
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")
    try:
        result = graph.invoke({"query": body.message, "role": user["role"]}, config=cfg)
        return ChatResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Graph pipeline error.") from e

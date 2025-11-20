from __future__ import annotations

import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List
from ollama import Client
from fastapi import FastAPI, Depends
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.indexer import indexer_service
from app.services.rag import rag_service
from app.services.auth import auth_service
from app.graph.rag_graph import build_graph

graph = build_graph()

# --- Lifespan (optional auto-index with marker to avoid repeats) ---
INDEX_MARKER = Path("/tmp/.rag_indexed")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("AUTO_INDEX") == "1" and not INDEX_MARKER.exists():
        try:
            count = indexer_service.reindex()
            print(f"[startup] indexed chunks: {count}")
            INDEX_MARKER.touch()
        except Exception as e:
            print("[startup] auto-index failed:", e)

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

_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Models ---
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

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

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/version")
def version():
    return {
        "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
        "vector_db": os.getenv("VECTOR_DB", "qdrant"),
    }


# ---RAG endpoint ----
@app.post("/chat/rag", response_model=ChatResponse)
def chat_rag(body: ChatRequest, user=Depends(auth_service.authenticate)):
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")
    try:
        answer, sources = rag_service.generate(body.message, user["role"])
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        # Avoids leaking internals to clients
        raise HTTPException(status_code=500, detail="RAG pipeline error.") from e
 
# ---LangGraph endpoint ----
@app.post("/chat/graph", response_model=ChatResponse)
def chat_graph(body: ChatRequest, user=Depends(auth_service.authenticate)):
    if graph is None:
        return ChatResponse(answer="Graph pipeline is disabled on the server.", sources=[])
    # stable per-user default if client didn’t send one
    tid = body.thread_id or f"{user['username']}"
    cfg = {"configurable": {"thread_id": tid, "checkpoint_ns": "default"}}
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")
    try:
        result = graph.invoke({"query": body.message, "role": user["role"]}, config=cfg)
        return ChatResponse(answer=result.get("answer",""), sources=result.get("sources",[]))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Graph pipeline error.") from e

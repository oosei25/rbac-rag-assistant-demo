from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A prompt-aligned, authorized source excerpt returned to clients."""

    citation_id: int = Field(ge=1)
    document_id: str
    path: str
    title: str
    department: str
    section: str
    score: float
    snippet: str


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


class DocumentSummary(BaseModel):
    document_id: str
    path: str
    title: str
    department: str
    preview: str


class DocumentDetail(DocumentSummary):
    content: str


class HealthResponse(BaseModel):
    ok: bool
    index_ready: bool
    indexed_chunks: int
    vector_db: str
    indexing_error: str | None = None

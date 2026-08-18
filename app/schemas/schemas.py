from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DemoRole = Literal[
    "employee",
    "finance",
    "marketing",
    "hr",
    "engineering",
    "clevel",
]


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


class AppliedAccessFilter(BaseModel):
    """Department-only filter summary safe to expose to an authenticated user."""

    initial_departments: list[str]
    fallback_departments: list[str] = Field(default_factory=list)


class CandidateCounts(BaseModel):
    """Counts measured only after the authorization boundary."""

    authorized_after_policy: int = Field(ge=0)
    authorized_after_relevance: int = Field(ge=0)
    selected_for_generation: int = Field(ge=0)


class AccessDecisionTrace(BaseModel):
    authenticated_role: str
    allowed_departments: list[str]
    requested_departments: list[str]
    applied_filter: AppliedAccessFilter
    candidate_counts: CandidateCounts
    decision: Literal["answered", "denied"]
    reason: Literal[
        "grounded_authorized_answer",
        "role_not_authorized",
        "no_authorized_relevant_context",
        "model_returned_no_grounded_answer",
        "answer_failed_citation_validation",
    ]
    authorized_source_count: int = Field(ge=0)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    access_trace: AccessDecisionTrace | None = None


class SecurityLabRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    left_role: DemoRole
    right_role: DemoRole


class SecurityLabResult(BaseModel):
    role: DemoRole
    answer: str
    citations: list[Citation]
    access_trace: AccessDecisionTrace


class SecurityLabResponse(BaseModel):
    question: str
    left: SecurityLabResult
    right: SecurityLabResult


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

"""Pydantic schemas for API request validation and response serialization."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Document Schemas ───────────────────────────────────────────────────────────

class DocumentBase(BaseModel):
    filename: str


class DocumentResponse(DocumentBase):
    id: int
    file_type: str
    status: str
    chunk_count: int | None = None
    error_msg: str | None = None
    upload_date: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


# ── Chat Schemas ───────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    title: str = Field(default="New Chat", max_length=255)


class SessionResponse(BaseModel):
    id: int
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int


class MessageResponse(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    sources: str | None = None
    timestamp: datetime

    model_config = {"from_attributes": True}


class SessionDetailResponse(SessionResponse):
    messages: list[MessageResponse] = []


class QueryRequest(BaseModel):
    session_id: int
    question: str = Field(..., min_length=1, max_length=2000)


class SourceItem(BaseModel):
    """Structured source citation returned with every RAG answer."""

    filename: str
    doc_id: int | None = None
    chunk_index: int | None = None
    title: str | None = None
    authors: str | None = None
    pub_year: int | None = None
    doi: str | None = None
    subjareas: str | None = None
    chunk_text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: int
    message_id: int


# ── Analytics Schemas ─────────────────────────────────────────────────────────

class AnalyticsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    avg_chunks_per_doc: float
    by_status: dict[str, int]
    by_type: dict[str, int]
    top_docs_by_chunks: list[dict[str, Any]]
    total_sessions: int
    total_queries: int


# ── Health / Status Schemas ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_available: bool
    documents_indexed: int

"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    documents_indexed: int


class SessionCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class SessionResponse(BaseModel):
    id: int
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int


class SourceItem(BaseModel):
    filename: str
    doc_id: int | None = None
    chunk_index: int | None = None
    title: str | None = None
    authors: str | None = None
    pub_year: int | None = None
    doi: str | None = None
    subjareas: str | None = None
    chunk_text: str


class MessageResponse(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    sources: str | None = None
    timestamp: datetime

    model_config = {"from_attributes": True}


class SessionDetailResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    messages: list[MessageResponse] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class QueryRequest(BaseModel):
    session_id: int
    question: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: int
    message_id: int


class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    status: str
    chunk_count: int | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class UploadResponse(BaseModel):
    id: int
    filename: str
    status: str
    message: str | None = None


class AnalyticsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_sessions: int
    total_queries: int
    avg_chunks_per_doc: float
    by_status: dict[str, int]
    by_type: dict[str, int]
    top_docs_by_chunks: list[dict[str, Any]]
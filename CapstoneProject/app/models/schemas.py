"""Pydantic schemas for API request validation and response serialization."""

from datetime import datetime

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


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: int
    message_id: int


# ── Health / Status Schemas ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_available: bool
    documents_indexed: int

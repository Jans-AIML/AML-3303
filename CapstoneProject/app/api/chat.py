"""Chat API router — manage sessions and submit queries."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.database import ChatSession, Message
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    SessionCreate,
    SessionDetailResponse,
    SessionListResponse,
    SessionResponse,
)
from app.services.llm_service import generate_answer
from app.services.vector_store import retrieve_chunks

router = APIRouter()


# ── Session management ────────────────────────────────────────────────────────

@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    session = ChatSession(title=payload.title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: int, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: int, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    db.delete(session)
    db.commit()


# ── Query / RAG ───────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, db: Session = Depends(get_db)):
    """Accept a user question, run RAG, return a grounded answer and sources."""
    session = db.get(ChatSession, payload.session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # 1. Save user message
    user_msg = Message(session_id=payload.session_id, role="user", content=payload.question)
    db.add(user_msg)
    db.commit()

    # 2. Retrieve relevant chunks from ChromaDB
    chunks, sources = retrieve_chunks(payload.question)

    # 3. Fetch recent conversation history (last 3 pairs = 6 messages)
    history = (
        db.query(Message)
        .filter(Message.session_id == payload.session_id)
        .order_by(Message.timestamp.desc())
        .limit(6)
        .all()
    )
    history_text = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in reversed(history)
    )

    # 4. Call LLM with context
    answer = generate_answer(question=payload.question, chunks=chunks, history=history_text)

    # 5. Save assistant message
    import json
    assistant_msg = Message(
        session_id=payload.session_id,
        role="assistant",
        content=answer,
        sources=json.dumps(sources),
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return QueryResponse(
        answer=answer,
        sources=sources,
        session_id=payload.session_id,
        message_id=assistant_msg.id,
    )

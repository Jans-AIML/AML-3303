"""Chat API router — manage sessions and submit queries."""
import json
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, selectinload

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


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    session = ChatSession(title=payload.title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).order_by(ChatSession.id.desc()).all()
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(
    session_id: int,
    limit_messages: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    session = (
        db.query(ChatSession)
        .options(selectinload(ChatSession.messages))
        .filter(ChatSession.id == session_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    ordered_messages = sorted(session.messages, key=lambda m: m.id or 0)
    session.messages = ordered_messages[-limit_messages:]
    return session


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: int, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    db.delete(session)
    db.commit()


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, db: Session = Depends(get_db)):
    session = db.get(ChatSession, payload.session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    user_msg = Message(session_id=payload.session_id, role="user", content=payload.question)
    db.add(user_msg)
    db.commit()

    recent_history = (
        db.query(Message)
        .filter(Message.session_id == payload.session_id)
        .order_by(Message.id.desc())
        .limit(6)
        .all()
    )
    recent_history = list(reversed(recent_history))

    chunks, sources = retrieve_chunks(payload.question, top_k=3)
    answer = generate_answer(payload.question, chunks, recent_history)

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
        session_id=payload.session_id,
        message_id=assistant_msg.id,
        answer=answer,
        sources=sources,
    )
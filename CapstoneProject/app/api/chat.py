"""Chat API router — manage sessions and submit queries."""

from __future__ import annotations

import json
import logging
import time

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
from app.services.safety_service import (
    moderate_chunks,
    moderate_input,
    moderate_output,
    sanitize_sources,
)
from app.services.vector_store import retrieve_chunks

logger = logging.getLogger(__name__)
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
    sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: int, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    return session


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: int, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    db.delete(session)
    db.commit()


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, db: Session = Depends(get_db)):
    """Accept a user question, run RAG, return a grounded answer and sources."""
    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    session = db.get(ChatSession, payload.session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    t_session_lookup = time.perf_counter() - t0

    t0 = time.perf_counter()
    user_msg = Message(
        session_id=payload.session_id,
        role="user",
        content=payload.question,
    )
    db.add(user_msg)
    db.commit()
    t_save_user = time.perf_counter() - t0

    t0 = time.perf_counter()
    input_check = moderate_input(payload.question)
    t_input_guard = time.perf_counter() - t0

    if not input_check.allowed:
        blocked_answer = input_check.message

        t1 = time.perf_counter()
        assistant_msg = Message(
            session_id=payload.session_id,
            role="assistant",
            content=blocked_answer,
            sources=json.dumps([]),
        )
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)
        t_save_blocked = time.perf_counter() - t1

        total_time = time.perf_counter() - t_total_start
        logger.info(
            "QUERY TIMING | session=%s | blocked=input | "
            "session_lookup=%.3fs | save_user=%.3fs | input_guard=%.3fs | "
            "save_blocked=%.3fs | total=%.3fs",
            payload.session_id, t_session_lookup, t_save_user,
            t_input_guard, t_save_blocked, total_time,
        )

        return QueryResponse(
            answer=blocked_answer,
            sources=[],
            session_id=payload.session_id,
            message_id=assistant_msg.id,
        )

    t0 = time.perf_counter()
    chunks, sources = retrieve_chunks(payload.question)
    t_retrieve = time.perf_counter() - t0

    t0 = time.perf_counter()
    safe_chunks, retrieval_check = moderate_chunks(chunks)
    safe_sources = sanitize_sources(sources)
    t_retrieval_guard = time.perf_counter() - t0

    if not retrieval_check.allowed:
        blocked_answer = retrieval_check.message

        t1 = time.perf_counter()
        assistant_msg = Message(
            session_id=payload.session_id,
            role="assistant",
            content=blocked_answer,
            sources=json.dumps([]),
        )
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)
        t_save_blocked = time.perf_counter() - t1

        total_time = time.perf_counter() - t_total_start
        logger.info(
            "QUERY TIMING | session=%s | blocked=retrieval | "
            "session_lookup=%.3fs | save_user=%.3fs | input_guard=%.3fs | "
            "retrieve=%.3fs | retrieval_guard=%.3fs | save_blocked=%.3fs | total=%.3fs",
            payload.session_id, t_session_lookup, t_save_user, t_input_guard,
            t_retrieve, t_retrieval_guard, t_save_blocked, total_time,
        )

        return QueryResponse(
            answer=blocked_answer,
            sources=[],
            session_id=payload.session_id,
            message_id=assistant_msg.id,
        )

    t0 = time.perf_counter()
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
    t_history = time.perf_counter() - t0

    t0 = time.perf_counter()
    answer = generate_answer(
        question=payload.question,
        chunks=safe_chunks,
        history=history_text,
    )
    t_generate = time.perf_counter() - t0

    t0 = time.perf_counter()
    output_check = moderate_output(answer)
    if not output_check.allowed:
        answer = output_check.message
        safe_sources = []
    t_output_guard = time.perf_counter() - t0

    t0 = time.perf_counter()
    assistant_msg = Message(
        session_id=payload.session_id,
        role="assistant",
        content=answer,
        sources=json.dumps(safe_sources),
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)
    t_save_assistant = time.perf_counter() - t0

    total_time = time.perf_counter() - t_total_start
    logger.info(
        "QUERY TIMING | session=%s | chunks=%d | safe_chunks=%d | "
        "session_lookup=%.3fs | save_user=%.3fs | input_guard=%.3fs | "
        "retrieve=%.3fs | retrieval_guard=%.3fs | history=%.3fs | "
        "generate=%.3fs | output_guard=%.3fs | save_assistant=%.3fs | total=%.3fs",
        payload.session_id, len(chunks), len(safe_chunks),
        t_session_lookup, t_save_user, t_input_guard,
        t_retrieve, t_retrieval_guard, t_history,
        t_generate, t_output_guard, t_save_assistant, total_time,
    )

    return QueryResponse(
        answer=answer,
        sources=safe_sources,
        session_id=payload.session_id,
        message_id=assistant_msg.id,
    )

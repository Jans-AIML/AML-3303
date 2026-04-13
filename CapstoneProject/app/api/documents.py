"""Documents API router — upload, list, retrieve, delete, and analytics."""

import os
import shutil

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.models.database import ChatSession, Document, Message
from app.models.schemas import AnalyticsResponse, DocumentListResponse, DocumentResponse
from app.tasks.background import process_document_task

router = APIRouter()

ALLOWED = set(settings.allowed_extensions)


def _validate_extension(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '.{ext}' is not allowed. Accepted: {sorted(ALLOWED)}",
        )
    return ext


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Upload a document. Processing (chunking + embedding) runs in the background."""
    ext = _validate_extension(file.filename or "")

    # Persist file to disk
    dest_path = os.path.join(settings.upload_dir, file.filename)
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Create DB record
    doc = Document(filename=file.filename, file_path=dest_path, file_type=ext, status="pending")
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Kick off background processing
    background_tasks.add_task(process_document_task, doc.id, dest_path, ext)

    return doc


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db)):
    """Return all uploaded documents."""
    docs = db.query(Document).order_by(Document.upload_date.desc()).all()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(db: Session = Depends(get_db)):
    """Return key metrics about documents and chat usage."""
    # Documents by status
    by_status = {
        row[0]: row[1]
        for row in db.query(Document.status, func.count(Document.id)).group_by(Document.status).all()
    }
    # Documents by file type
    by_type = {
        row[0]: row[1]
        for row in db.query(Document.file_type, func.count(Document.id)).group_by(Document.file_type).all()
    }
    # Chunk aggregates (processed docs only)
    total_chunks = db.query(func.coalesce(func.sum(Document.chunk_count), 0)).filter(
        Document.status == "processed"
    ).scalar() or 0
    avg_chunks = db.query(func.avg(Document.chunk_count)).filter(
        Document.status == "processed"
    ).scalar() or 0.0
    # Top 10 documents by chunk count
    top_docs = (
        db.query(Document.filename, Document.chunk_count)
        .filter(Document.chunk_count.isnot(None))
        .order_by(Document.chunk_count.desc())
        .limit(10)
        .all()
    )
    # Chat stats
    total_sessions = db.query(func.count(ChatSession.id)).scalar() or 0
    total_queries = (
        db.query(func.count(Message.id)).filter(Message.role == "user").scalar() or 0
    )
    return AnalyticsResponse(
        total_documents=sum(by_status.values()),
        total_chunks=int(total_chunks),
        avg_chunks_per_doc=round(float(avg_chunks), 1),
        by_status=by_status,
        by_type=by_type,
        top_docs_by_chunks=[
            {"filename": row[0], "chunk_count": row[1]} for row in top_docs
        ],
        total_sessions=total_sessions,
        total_queries=total_queries,
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    """Return details for a single document."""
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """Delete a document record and its uploaded file."""
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Remove file from disk if it still exists
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    db.delete(doc)
    db.commit()

"""Background task: parse, chunk, embed, and index an uploaded document."""

import logging

from app.db.session import SessionLocal
from app.models.database import Document
from app.services.document_processor import chunk_text, extract_text
from app.services.vector_store import add_chunks

logger = logging.getLogger(__name__)


def process_document_task(doc_id: int, file_path: str, file_type: str) -> None:
    """Called by FastAPI BackgroundTasks after a document is uploaded.

    Updates the document status in the DB at each stage so the UI can
    poll for progress.
    """
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            logger.error("process_document_task: doc_id=%s not found", doc_id)
            return

        # ── Step 1: Mark as processing ─────────────────────────
        doc.status = "processing"
        db.commit()

        # ── Step 2: Extract text ───────────────────────────────
        logger.info("Extracting text from %s (type=%s)", file_path, file_type)
        text = extract_text(file_path, file_type)

        # ── Step 3: Chunk ──────────────────────────────────────
        chunks = chunk_text(text)
        logger.info("Created %d chunks for doc_id=%s", len(chunks), doc_id)

        # ── Step 4: Embed and store in ChromaDB ────────────────
        add_chunks(doc_id=doc_id, filename=doc.filename, chunks=chunks)

        # ── Step 5: Update DB status ───────────────────────────
        doc.status = "processed"
        doc.chunk_count = len(chunks)
        db.commit()
        logger.info("doc_id=%s processed successfully (%d chunks)", doc_id, len(chunks))

    except Exception as exc:
        logger.exception("Failed to process doc_id=%s: %s", doc_id, exc)
        doc = db.get(Document, doc_id)
        if doc:
            doc.status = "failed"
            doc.error_msg = str(exc)
            db.commit()
    finally:
        db.close()

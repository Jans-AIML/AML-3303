"""ChromaDB vector store service.

Handles adding document chunks and querying for similar content.
"""

from __future__ import annotations

import logging
import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import settings

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _clean_chunk(text: str) -> str:
    """Remove NIH watermark boilerplate from within a chunk."""
    text = re.sub(
        r"(NIH-PA\s*\n?Author\s*\n?Manuscript\s*\n?)+",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(NIH Public Access\s*\n?)+", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"Author manuscript; available in PMC[^\n]*\n?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Published in final edited form[^\n]*\n?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Q Rev Biophys\. Author manuscript[^\n]*\n?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def add_chunks(
    doc_id: int,
    filename: str,
    chunks: list[str],
    doc_metadata: dict | None = None,
) -> None:
    """Embed and store document chunks in ChromaDB.

    If chunks for *doc_id* already exist, delete them first to avoid duplicates.
    """
    collection = _get_collection()

    try:
        existing = collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception as exc:
        logger.warning(
            "Dedup check skipped for doc_id=%s — ChromaDB not ready: %s",
            doc_id,
            exc,
        )

    ids = [f"doc{doc_id}_chunk{i}" for i in range(len(chunks))]

    base: dict = {
        "doc_id": doc_id,
        "filename": filename,
    }

    if doc_metadata:
        for k, v in doc_metadata.items():
            if v is not None and v != "" and isinstance(v, (str, int, float, bool)):
                base[k] = v

    metadatas = [{**base, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)


def retrieve_chunks(query: str, top_k: int | None = None) -> tuple[list[str], list[dict]]:
    """Retrieve the most relevant chunks plus structured source metadata."""
    collection = _get_collection()

    if collection.count() == 0:
        return [], []

    final_k = top_k or settings.retrieval_top_k
    raw_k = min(final_k * 2, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=raw_k,
        include=["documents", "metadatas"],
    )

    docs: list[str] = results["documents"][0] if results.get("documents") else []
    metas: list[dict] = results["metadatas"][0] if results.get("metadatas") else []

    candidates: list[tuple[str, dict]] = []

    for doc, meta in zip(docs, metas):
        cleaned = _clean_chunk(doc)
        if len(cleaned.split()) >= 20:
            candidates.append((cleaned, meta))
        elif doc and len(doc.split()) >= 20:
            candidates.append((doc.strip(), meta))

    if not candidates:
        for doc, meta in zip(docs, metas):
            if doc:
                candidates.append((doc.strip(), meta))

    candidates = candidates[:final_k]

    final_docs: list[str] = []
    final_sources: list[dict] = []

    for doc_text, meta in candidates:
        final_docs.append(doc_text)
        final_sources.append(
            {
                "filename": meta.get("filename", "unknown"),
                "doc_id": meta.get("doc_id"),
                "chunk_index": meta.get("chunk_index"),
                "title": meta.get("title"),
                "authors": meta.get("authors"),
                "pub_year": meta.get("pub_year"),
                "doi": meta.get("doi"),
                "subjareas": meta.get("subjareas"),
                "chunk_text": doc_text,
            }
        )

    return final_docs, final_sources


def delete_document_chunks(doc_id: int) -> None:
    """Remove all chunks belonging to a specific document."""
    collection = _get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
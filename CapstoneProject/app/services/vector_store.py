"""ChromaDB vector store service.

Handles adding document chunks and querying for similar content.
"""

from __future__ import annotations

import re
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import settings

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
    text = re.sub(r'(NIH-PA\s*\n?Author\s*\n?Manuscript\s*\n?)+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'(NIH Public Access\s*\n?)+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Author manuscript; available in PMC[^\n]*\n?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Published in final edited form[^\n]*\n?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Q Rev Biophys\. Author manuscript[^\n]*\n?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def add_chunks(
    doc_id: int,
    filename: str,
    chunks: list[str],
    doc_metadata: dict | None = None,
) -> None:
    """Embed and store document chunks in ChromaDB.

    If chunks for *doc_id* already exist (e.g. a re-upload), they are deleted
    first so we never accumulate duplicate embeddings for the same document.

    *doc_metadata* is an optional dict of document-level fields
    (title, pub_year, authors, doi, subjareas) stored alongside every chunk.
    Values must be str, int, float, or bool to satisfy ChromaDB constraints.
    """
    collection = _get_collection()

    # ── Deduplication: remove any prior chunks for this doc ──────────────────
    try:
        existing = collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Dedup check skipped for doc_id=%s — ChromaDB not ready: %s", doc_id, exc
        )

    ids = [f"doc{doc_id}_chunk{i}" for i in range(len(chunks))]

    base: dict = {"doc_id": doc_id, "filename": filename}
    if doc_metadata:
        for k, v in doc_metadata.items():
            if v is not None and v != "" and isinstance(v, (str, int, float, bool)):
                base[k] = v

    metadatas = [{**base, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)


def retrieve_chunks(query: str, top_k: int | None = None) -> tuple[list[str], list[str]]:
    collection = _get_collection()
    if collection.count() == 0:
        return [], []

    k = min((top_k or settings.retrieval_top_k) * 2, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"],
    )

    docs: list[str] = results["documents"][0] if results["documents"] else []
    metas: list[dict] = results["metadatas"][0] if results["metadatas"] else []

    # Clean watermark from each chunk, skip if nothing meaningful left
    clean_docs = []
    clean_metas = []
    for doc, meta in zip(docs, metas):
        cleaned = _clean_chunk(doc)
        if len(cleaned.split()) >= 20:
            clean_docs.append(cleaned)
            clean_metas.append(meta)

    # fallback: use raw docs if everything got filtered
    final_docs = clean_docs[:top_k or settings.retrieval_top_k] or docs[:top_k or settings.retrieval_top_k]
    final_metas = clean_metas[:top_k or settings.retrieval_top_k] or metas[:top_k or settings.retrieval_top_k]

    # Build citations
    seen: dict[str, str] = {}
    for m in final_metas:
        filename = m.get("filename", "unknown")
        if filename in seen:
            continue
        authors = m.get("authors", "")
        year = m.get("pub_year", "")
        if authors or year:
            parts: list[str] = []
            if authors:
                parts.append(str(authors))
            if year:
                parts.append(f"({year})")
            parts.append(f"— {filename}")
            seen[filename] = " ".join(parts)
        else:
            seen[filename] = filename

    return final_docs, list(seen.values())


def delete_document_chunks(doc_id: int) -> None:
    """Remove all chunks belonging to a specific document."""
    collection = _get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
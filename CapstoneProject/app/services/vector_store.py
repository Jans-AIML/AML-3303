"""ChromaDB vector store service.

Handles adding document chunks and querying for similar content.
"""

from __future__ import annotations

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
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [f"doc{doc_id}_chunk{i}" for i in range(len(chunks))]

    # Build base metadata shared across all chunks
    base: dict = {"doc_id": doc_id, "filename": filename}
    if doc_metadata:
        for k, v in doc_metadata.items():
            # Only store values ChromaDB accepts; skip blanks
            if v is not None and v != "" and isinstance(v, (str, int, float, bool)):
                base[k] = v

    metadatas = [{**base, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)


def retrieve_chunks(query: str) -> tuple[list[str], list[str]]:
    """Return top-k relevant chunks and formatted source citations.

    Citation format (when metadata is available):
        ``Smith J., Jones M. (2019) — S0264410X19301264.json``
    Falls back to just the filename when author/year metadata is absent.
    Sources are de-duplicated by filename while preserving relevance order.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return [], []

    results = collection.query(
        query_texts=[query],
        n_results=min(settings.retrieval_top_k, collection.count()),
        include=["documents", "metadatas"],
    )

    docs: list[str] = results["documents"][0] if results["documents"] else []
    metas: list[dict] = results["metadatas"][0] if results["metadatas"] else []

    # Build rich citations, de-duplicated by filename (relevance order)
    seen: dict[str, str] = {}  # filename → citation string
    for m in metas:
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

    return docs, list(seen.values())


def delete_document_chunks(doc_id: int) -> None:
    """Remove all chunks belonging to a specific document."""
    collection = _get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])

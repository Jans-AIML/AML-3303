"""ChromaDB vector store service.

Handles adding document chunks and querying for similar content.
"""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import settings

_client: chromadb.PersistentClient | None = None
_collection = None
_embedding_fn = None


def _get_collection():
    global _client, _collection, _embedding_fn
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        if _embedding_fn is None:
            _embedding_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=_embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunks(doc_id: int, filename: str, chunks: list[str], doc_metadata: dict | None = None) -> None:
    collection = _get_collection()
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
    count = collection.count()
    if count == 0:
        return [], []

    k = min(top_k or settings.retrieval_top_k, count)
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"],
    )

    docs: list[str] = results["documents"][0] if results["documents"] else []
    metas: list[dict] = results["metadatas"][0] if results["metadatas"] else []

    trimmed_docs = []
    for doc in docs[:k]:
        trimmed_docs.append(" ".join(doc.split())[:1200])

    seen: dict[str, str] = {}
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

    return trimmed_docs, list(seen.values())


def delete_document_chunks(doc_id: int) -> None:
    collection = _get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
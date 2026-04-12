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


def add_chunks(doc_id: int, filename: str, chunks: list[str]) -> None:
    """Embed and store document chunks in ChromaDB."""
    collection = _get_collection()
    ids = [f"doc{doc_id}_chunk{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "filename": filename, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)


def retrieve_chunks(query: str) -> tuple[list[str], list[str]]:
    """Return the top-k most relevant chunks and their source filenames."""
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
    sources = list({m.get("filename", "unknown") for m in metas})
    return docs, sources


def delete_document_chunks(doc_id: int) -> None:
    """Remove all chunks belonging to a specific document."""
    collection = _get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])

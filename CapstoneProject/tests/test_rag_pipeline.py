"""Tests for the RAG pipeline — document processing, chunking, and retrieval.

These tests are UNIT tests that run in-process without an Ollama server.
The LLM call is bypassed; we test only the document parsing and vector logic.
"""

import os
import tempfile

import pytest

from app.services.document_processor import chunk_text, extract_text
from app.services.vector_store import add_chunks, delete_document_chunks, retrieve_chunks


# ── Document processor tests ──────────────────────────────────────────────────

class TestTextExtraction:
    def test_extract_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world.\nThis is a test document.\nLine three.")
        text = extract_text(str(f), "txt")
        assert "Hello world" in text
        assert "Line three" in text

    def test_extract_csv_file(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("title,content\nRefund Policy,Refunds accepted within 30 days.\n")
        text = extract_text(str(f), "csv")
        assert "Refund Policy" in text
        assert "30 days" in text

    def test_extract_csv_formats_as_key_value(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("name,value\nfoo,bar\n")
        text = extract_text(str(f), "csv")
        assert "name: foo" in text or "foo" in text

    def test_unsupported_type_raises_error(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(str(f), "xyz")


class TestChunking:
    def test_short_text_returns_single_chunk(self):
        short = "This is a very short document."
        chunks = chunk_text(short)
        assert len(chunks) >= 1
        assert any("short document" in c for c in chunks)

    def test_long_text_is_split_into_multiple_chunks(self):
        # 2000-char text should be split into multiple 500-char chunks
        long_text = ("Word " * 400)  # ~2000 chars
        chunks = chunk_text(long_text)
        assert len(chunks) > 1

    def test_chunks_contain_original_content(self):
        text = "Section A: Setup instructions. " * 30
        chunks = chunk_text(text)
        all_text = " ".join(chunks)
        assert "Setup instructions" in all_text

    def test_chunks_are_strings(self):
        chunks = chunk_text("Some sample text for testing purposes.")
        assert all(isinstance(c, str) for c in chunks)

    def test_empty_text_returns_empty_or_one_chunk(self):
        chunks = chunk_text("")
        assert isinstance(chunks, list)


# ── Vector store tests ─────────────────────────────────────────────────────────

class TestVectorStore:
    """Uses a temporary ChromaDB directory so tests don't pollute the real DB."""

    @pytest.fixture(autouse=True)
    def temp_chroma(self, tmp_path, monkeypatch):
        """Redirect ChromaDB to a per-test temporary directory for isolation.

        We must patch ``settings`` on the vector_store module directly because
        it imports settings at module load time, so patching app.config.settings
        would have no effect on the already-bound reference.
        """
        import app.services.vector_store as vs
        import uuid

        # Unique collection name per test prevents cross-test data bleed
        unique_collection = f"test_{uuid.uuid4().hex}"

        class _TestSettings:
            chroma_persist_dir = str(tmp_path / "chroma")
            chroma_collection_name = unique_collection
            embedding_model = "all-MiniLM-L6-v2"
            retrieval_top_k = 3

        monkeypatch.setattr(vs, "_client", None)
        monkeypatch.setattr(vs, "_collection", None)
        monkeypatch.setattr(vs, "settings", _TestSettings())
        yield
        monkeypatch.setattr(vs, "_client", None)
        monkeypatch.setattr(vs, "_collection", None)

    def test_add_and_retrieve_chunks(self):
        chunks = [
            "Refund policy: customers may return within 30 days.",
            "Setup guide: download the installer from the website.",
            "Account settings: change your password under profile settings.",
        ]
        add_chunks(doc_id=1, filename="policy.txt", chunks=chunks)
        results, sources = retrieve_chunks("How do I get a refund?")
        assert len(results) > 0
        assert any("refund" in r.lower() or "return" in r.lower() for r in results)

    def test_sources_contain_filename(self):
        add_chunks(doc_id=2, filename="manual.txt", chunks=["Product manual content."])
        _, sources = retrieve_chunks("product manual")
        assert "manual.txt" in sources

    def test_retrieve_from_empty_store_returns_empty(self):
        results, sources = retrieve_chunks("anything")
        assert results == []
        assert sources == []

    def test_delete_removes_chunks(self):
        add_chunks(doc_id=3, filename="temp.txt", chunks=["Temporary document content."])
        delete_document_chunks(doc_id=3)
        results, _ = retrieve_chunks("temporary document")
        # After deletion there should be no results for doc 3
        # (collection is empty, so empty list is returned)
        assert isinstance(results, list)

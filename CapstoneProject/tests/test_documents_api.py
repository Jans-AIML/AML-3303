"""Tests for the documents API endpoints.

Uses FastAPI TestClient (in-process, no running server needed) with an
in-memory SQLite database so tests are fully isolated and repeatable.
"""

import io
import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import get_db
from app.main import app
from app.models.database import Base

# ── In-memory test database ────────────────────────────────────────────────────
# StaticPool ensures all connections share the same in-memory SQLite database.
TEST_DB_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after.

    Also patches ``process_document_task`` so background processing never
    touches the real ``support_assistant.db`` file during tests (which would
    fail with a lock error if the seed script is running concurrently).
    """
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    with patch("app.api.documents.process_document_task"):
        yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a temporary .txt file for upload tests."""
    f = tmp_path / "sample.txt"
    f.write_text(
        "This is a sample support document.\n"
        "It contains information about product setup and troubleshooting.\n"
        "Section 1: Installation\nDownload the installer and follow the steps.\n"
        "Section 2: Refund Policy\nRefunds are accepted within 30 days.\n"
    )
    return f


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_version_not_empty(self, client):
        response = client.get("/health")
        assert response.json()["version"] != ""


class TestDocumentList:
    def test_empty_list_on_fresh_db(self, client):
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total"] == 0

    def test_response_schema_present(self, client):
        response = client.get("/api/documents")
        data = response.json()
        assert "documents" in data
        assert "total" in data


class TestDocumentUpload:
    def test_upload_txt_file_accepted(self, client, sample_txt_file):
        with open(sample_txt_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("sample.txt", f, "text/plain")},
            )
        assert response.status_code == 202
        data = response.json()
        assert data["filename"] == "sample.txt"
        assert data["file_type"] == "txt"
        assert data["status"] in ("pending", "processing", "processed")
        assert "id" in data

    def test_upload_creates_db_record(self, client, sample_txt_file):
        with open(sample_txt_file, "rb") as f:
            client.post(
                "/api/documents/upload",
                files={"file": ("sample.txt", f, "text/plain")},
            )
        list_response = client.get("/api/documents")
        assert list_response.json()["total"] == 1

    def test_upload_rejected_for_bad_extension(self, client):
        fake_file = io.BytesIO(b"not a valid file")
        response = client.post(
            "/api/documents/upload",
            files={"file": ("malicious.exe", fake_file, "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "not allowed" in response.json()["detail"].lower()

    def test_upload_rejected_for_no_extension(self, client):
        fake_file = io.BytesIO(b"no extension file")
        response = client.post(
            "/api/documents/upload",
            files={"file": ("noextension", fake_file, "text/plain")},
        )
        assert response.status_code == 400


class TestDocumentGetAndDelete:
    def test_get_document_by_id(self, client, sample_txt_file):
        with open(sample_txt_file, "rb") as f:
            upload_data = client.post(
                "/api/documents/upload",
                files={"file": ("sample.txt", f, "text/plain")},
            ).json()

        doc_id = upload_data["id"]
        response = client.get(f"/api/documents/{doc_id}")
        assert response.status_code == 200
        assert response.json()["id"] == doc_id

    def test_get_nonexistent_document_returns_404(self, client):
        response = client.get("/api/documents/9999")
        assert response.status_code == 404

    def test_delete_document(self, client, sample_txt_file):
        with open(sample_txt_file, "rb") as f:
            doc_id = client.post(
                "/api/documents/upload",
                files={"file": ("sample.txt", f, "text/plain")},
            ).json()["id"]

        delete_response = client.delete(f"/api/documents/{doc_id}")
        assert delete_response.status_code == 204

        get_response = client.get(f"/api/documents/{doc_id}")
        assert get_response.status_code == 404

    def test_delete_nonexistent_returns_404(self, client):
        response = client.delete("/api/documents/9999")
        assert response.status_code == 404

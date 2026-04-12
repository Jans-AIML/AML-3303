"""Tests for the chat sessions and query API endpoints."""

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
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def session(client):
    """Create a chat session and return its data."""
    response = client.post("/api/chat/sessions", json={"title": "Test Session"})
    return response.json()


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestSessionCreation:
    def test_create_session_returns_201(self, client):
        response = client.post("/api/chat/sessions", json={"title": "My Test Session"})
        assert response.status_code == 201

    def test_create_session_schema(self, client):
        response = client.post("/api/chat/sessions", json={"title": "My Test Session"})
        data = response.json()
        assert "id" in data
        assert data["title"] == "My Test Session"
        assert "created_at" in data

    def test_create_session_default_title(self, client):
        response = client.post("/api/chat/sessions", json={})
        assert response.status_code == 201
        assert response.json()["title"] == "New Chat"

    def test_title_too_long_rejected(self, client):
        response = client.post("/api/chat/sessions", json={"title": "x" * 256})
        assert response.status_code == 422


class TestSessionList:
    def test_empty_list_on_fresh_db(self, client):
        response = client.get("/api/chat/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["total"] == 0

    def test_list_includes_created_sessions(self, client, session):
        response = client.get("/api/chat/sessions")
        assert response.json()["total"] == 1

    def test_multiple_sessions_counted(self, client):
        for i in range(3):
            client.post("/api/chat/sessions", json={"title": f"Session {i}"})
        response = client.get("/api/chat/sessions")
        assert response.json()["total"] == 3


class TestSessionGetAndDelete:
    def test_get_session_by_id(self, client, session):
        response = client.get(f"/api/chat/sessions/{session['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session["id"]
        assert "messages" in data

    def test_get_session_has_empty_messages(self, client, session):
        response = client.get(f"/api/chat/sessions/{session['id']}")
        assert response.json()["messages"] == []

    def test_get_nonexistent_session_returns_404(self, client):
        response = client.get("/api/chat/sessions/9999")
        assert response.status_code == 404

    def test_delete_session(self, client, session):
        delete_response = client.delete(f"/api/chat/sessions/{session['id']}")
        assert delete_response.status_code == 204

        get_response = client.get(f"/api/chat/sessions/{session['id']}")
        assert get_response.status_code == 404

    def test_delete_nonexistent_returns_404(self, client):
        response = client.delete("/api/chat/sessions/9999")
        assert response.status_code == 404


class TestQueryValidation:
    """Tests that the /query endpoint validates input correctly.

    Note: This does NOT test actual RAG/LLM responses (those require Ollama
    running and documents indexed). LLM integration is tested separately via
    test_rag_pipeline.py.
    """

    def test_query_requires_existing_session(self, client):
        response = client.post(
            "/api/chat/query",
            json={"session_id": 9999, "question": "What is the refund policy?"},
        )
        assert response.status_code == 404

    def test_query_rejects_empty_question(self, client, session):
        response = client.post(
            "/api/chat/query",
            json={"session_id": session["id"], "question": ""},
        )
        assert response.status_code == 422

    def test_query_rejects_question_too_long(self, client, session):
        response = client.post(
            "/api/chat/query",
            json={"session_id": session["id"], "question": "x" * 2001},
        )
        assert response.status_code == 422

    def test_query_missing_session_id_rejected(self, client):
        response = client.post(
            "/api/chat/query",
            json={"question": "What is the refund policy?"},
        )
        assert response.status_code == 422

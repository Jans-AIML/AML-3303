"""FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import init_db

# Ensure upload directory exists at startup
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.chroma_persist_dir, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database tables on first run."""
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG-powered AI support assistant backed by Ollama and ChromaDB.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Allow the Streamlit frontend (running on a different port) to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers (imported here to avoid circular imports) ─────────────────────────
from app.api.documents import router as documents_router  # noqa: E402
from app.api.chat import router as chat_router  # noqa: E402
from app.models.schemas import HealthResponse  # noqa: E402

app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """Liveness probe — also reports Ollama availability and indexed document count."""
    import httpx

    from app.services.vector_store import _get_collection

    ollama_available = False
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        ollama_available = resp.status_code == 200
    except Exception:
        pass

    documents_indexed = 0
    try:
        documents_indexed = _get_collection().count()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        ollama_available=ollama_available,
        documents_indexed=documents_indexed,
    )

"""FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 8501
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import init_db

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.chroma_persist_dir, exist_ok=True)

_health_cache = {"timestamp": 0.0, "payload": None}
_HEALTH_TTL = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.documents import router as documents_router  # noqa: E402
from app.api.chat import router as chat_router  # noqa: E402
from app.models.schemas import HealthResponse  # noqa: E402

app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    now = time.time()
    cached = _health_cache.get("payload")
    if cached is not None and now - _health_cache["timestamp"] < _HEALTH_TTL:
        return cached

    import httpx
    from app.services.vector_store import _get_collection

    ollama_available = False
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=1.5)
        ollama_available = resp.status_code == 200
    except Exception:
        pass

    documents_indexed = 0
    try:
        documents_indexed = _get_collection().count()
    except Exception:
        pass

    payload = HealthResponse(
        status="ok",
        version=settings.app_version,
        ollama_available=ollama_available,
        documents_indexed=documents_indexed,
    )
    _health_cache["timestamp"] = now
    _health_cache["payload"] = payload
    return payload
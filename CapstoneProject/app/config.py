"""Application configuration via Pydantic Settings.

All values can be overridden by environment variables or a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── General ────────────────────────────────────────────────
    app_name: str = "AI Research Support Assistant"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Database ───────────────────────────────────────────────
    database_url: str = "sqlite:///./support_assistant.db"

    # ── File Storage ───────────────────────────────────────────
    upload_dir: str = "data/uploads"
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = ["pdf", "txt", "csv", "docx", "json"]

    # ── ChromaDB ───────────────────────────────────────────────
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "support_docs"

    # ── Embeddings ─────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Ollama / LLM ───────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma:latest"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512       # reduced from 1024 for speed

    # ── RAG ────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 6        # increased from 5
    max_context_chars: int = 5000   # increased from 2500 — improves answer quality

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()

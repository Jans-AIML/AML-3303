# AI Research Support Assistant

> AML-3303 Capstone Project — Applied Machine Learning  
> Lambton College | April 2026

An intelligent RAG-based support chatbot that ingests research documents, retrieves relevant information via semantic search, and generates professional answers using a local Ollama LLM.

---

## System Architecture

```
Streamlit UI  ──HTTP──►  FastAPI Backend  ──►  SQLite (metadata + history)
                                        ──►  ChromaDB   (document embeddings)
                                        ──►  Ollama     (gemma — local LLM)
```

## Features

- Upload PDF, TXT, CSV, DOCX, and JSON (Elsevier OA) documents
- Automatic background processing (chunking + embedding)
- Semantic retrieval via ChromaDB + sentence-transformers (`all-MiniLM-L6-v2`)
- Grounded answer generation with Ollama (gemma, local, no API key)
- Author + year rich citations: `Smith J., Jones M. (2021) — article.json`
- Deduplication: re-uploading a document replaces its embeddings, not duplicates
- Chat session management with full conversation history in SQLite
- FastAPI REST API with auto-generated Swagger docs at `/docs`
- Streamlit UI with real-time status badges, session management, and source expanders
- Analytics dashboard: KPI cards, status/type bar charts, top-10 documents

---

## Presentation

Open [`docs/presentation.html`](docs/presentation.html) in any browser for the 9-slide reveal.js deck (requires an internet connection to load the CDN).

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running with `gemma`

```bash
ollama serve        # start Ollama server
ollama pull gemma   # download gemma model (first time only)
```

### Installation

```bash
# Clone and enter project
git clone <repo-url>
cd AML-3303/CapstoneProject

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the application

**Terminal 1 — Backend API:**
```bash
uvicorn app.main:app --reload --port 8501
```
API docs (Swagger UI): http://localhost:8501/docs

**Terminal 2 — Streamlit UI:**
```bash
streamlit run ui/streamlit_app.py --server.port 8001
```
UI available at: http://localhost:8001

### (Optional) Seed the Elsevier Open Access corpus

If you have the [Elsevier OA corpus](https://elsevier.digitalcommonsdata.com/datasets/zm33cdndxs/2) downloaded, seed the knowledge base:

```bash
# Seed 400 articles (stratified across all 27 subject areas, ~4 min)
python scripts/seed_elsevier.py --reset --n 400 --stratified

# Seed 10,000 articles (~3-4 hours on CPU)
python scripts/seed_elsevier.py --reset --n 10000 --stratified
```

> **Important:** Complete the seed *before* starting the backend. Running them
> concurrently will corrupt the ChromaDB HNSW index.

---

## Project Structure

```
CapstoneProject/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # App settings (Pydantic)
│   ├── models/
│   │   ├── database.py            # SQLAlchemy ORM models
│   │   └── schemas.py             # Pydantic schemas
│   ├── api/
│   │   ├── documents.py           # Document upload/management
│   │   └── chat.py                # Chat query/session endpoints
│   ├── services/
│   │   ├── document_processor.py  # Text extraction + chunking
│   │   ├── vector_store.py        # ChromaDB operations
│   │   └── llm_service.py         # Ollama LLM integration
│   ├── tasks/
│   │   └── background.py          # Background processing task
│   └── db/
│       └── session.py             # DB session factory
├── ui/
│   └── streamlit_app.py           # Streamlit frontend
├── scripts/
│   └── seed_elsevier.py           # Bulk-load Elsevier OA JSON articles
├── data/sample_docs/              # Sample test documents (5 Elsevier articles)
├── tests/                         # 43 pytest tests (API + RAG pipeline)
├── docs/
│   └── tech_spec.md               # Full technical specification
├── logs/                          # Backend + seed logs
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Environment Variables

Create a `.env` file in `CapstoneProject/` to override defaults:

```ini
DEBUG=false
OLLAMA_MODEL=gemma:latest
OLLAMA_BASE_URL=http://localhost:11434
DATABASE_URL=sqlite:///./support_assistant.db
RETRIEVAL_TOP_K=5
```

---

## API Reference

| Method | Endpoint                    | Description                   |
|--------|-----------------------------|-------------------------------|
| GET    | `/health`                   | Liveness check                |
| POST   | `/api/documents/upload`     | Upload a document             |
| GET    | `/api/documents`            | List all documents            |
| GET    | `/api/documents/{id}`       | Document status + details     |
| DELETE | `/api/documents/{id}`       | Delete a document             |
| POST   | `/api/chat/sessions`        | Create a chat session         |
| GET    | `/api/chat/sessions`        | List all sessions             |
| GET    | `/api/chat/sessions/{id}`   | Session + message history     |
| POST   | `/api/chat/query`           | Ask a question (RAG answer)   |
| DELETE | `/api/chat/sessions/{id}`   | Delete a session              |

---

## Tech Stack

| Layer      | Technology                 |
|------------|----------------------------|
| Backend    | FastAPI + Uvicorn          |
| Validation | Pydantic v2                |
| Database   | SQLite + SQLAlchemy        |
| Vector DB  | ChromaDB                   |
| Embeddings | sentence-transformers      |
| LLM        | Ollama (gemma — local)     |
| UI         | Streamlit                  |
| Linting    | Ruff                       |
| Testing    | pytest                     |

---

## Team

| Member | Student ID | Contributions |
|--------|-----------|---------------|
| TBD    | TBD       | All modules   |
| TBD    | TBD       | All modules   |
| TBD    | TBD       | All modules   |

---

## Screenshots

*(Add screenshots here after Day 4 UI build)*

---

## License

For academic use only — AML-3303 Capstone Project, Lambton College.

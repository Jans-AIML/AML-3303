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

- Upload PDF, TXT, CSV, DOCX support documents
- Automatic background processing (chunking + embedding)
- Semantic retrieval via ChromaDB + sentence-transformers
- Grounded answer generation with Ollama (gemma, local, free)
- Chat session management with full conversation history in SQLite
- Source citations returned with every answer
- FastAPI REST API with auto-generated Swagger docs

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
uvicorn app.main:app --reload --port 8000
```
API docs available at: http://localhost:8000/docs

**Terminal 2 — Streamlit UI:**
```bash
streamlit run ui/streamlit_app.py
```
UI available at: http://localhost:8501

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
├── data/sample_docs/              # Sample test documents
├── tests/                         # pytest test suite
├── docs/
│   ├── tech_spec.md               # Full technical specification
│   └── architecture.md            # Architecture diagram
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

# Technical Specification — AI Research Support Assistant

**Course:** AML-3303 | Applied Machine Learning  
**Team Size:** 3 students  
**Date:** April 2026  
**Version:** 1.0

---

## 1. Problem Statement

Research teams and students often struggle with scattered academic documentation — published papers, policy PDFs, and help guides — spread across multiple formats. Manually searching for answers is time-consuming and leads to inconsistent support quality.

This project builds an **AI-powered Research Support Assistant** that ingests documents from the Elsevier Open Journals dataset, retrieves relevant information using semantic search, and generates professional answers through a local LLM via Ollama.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    Streamlit Web Application                    │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐   │
│  │ Upload Docs  │   │  Chat Window  │   │  Document List   │   │
│  └──────┬───────┘   └──────┬────────┘   └────────┬─────────┘   │
└─────────┼──────────────────┼─────────────────────┼─────────────┘
          │ HTTP              │ HTTP                 │ HTTP
┌─────────▼──────────────────▼─────────────────────▼─────────────┐
│                       FASTAPI BACKEND                           │
│  ┌──────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │  /api/documents  │  │   /api/chat     │  │ /api/sessions │  │
│  └────────┬─────────┘  └────────┬────────┘  └───────────────┘  │
│           │                     │                               │
│  ┌────────▼─────────────────────▼────────────────────────────┐ │
│  │                    SERVICE LAYER                          │ │
│  │  DocumentProcessor │ VectorStore │ LLMService │ Embeddings│ │
│  └────────┬───────────────────────┬───────────────────────────┘ │
│           │ Background Task       │                             │
└───────────┼───────────────────────┼─────────────────────────────┘
            │                       │
┌───────────▼──────┐    ┌───────────▼─────────────────────────────┐
│   SQLite DB      │    │           VECTOR STORE + LLM             │
│  ┌────────────┐  │    │  ┌──────────────┐   ┌────────────────┐  │
│  │ documents  │  │    │  │   ChromaDB   │   │ Ollama (gemma) │  │
│  │ messages   │  │    │  │ (embeddings) │   │  Local Server  │  │
│  │  sessions  │  │    │  └──────────────┘   └────────────────┘  │
│  └────────────┘  │    └─────────────────────────────────────────┘
└──────────────────┘
```

---

## 3. Data Flow

### 3.1 Document Ingestion Flow
```
User uploads file (PDF/CSV/TXT)
    → FastAPI receives file
    → Store metadata in SQLite (status: "pending")
    → Return upload confirmation immediately
    → Background task starts:
        ├── Extract text (PyPDF2 / pdfplumber / csv reader)
        ├── Split into chunks (500 tokens, 50 overlap)
        ├── Generate embeddings (sentence-transformers)
        ├── Persist chunks in ChromaDB
        └── Update SQLite status to "processed"
```

### 3.2 Query / RAG Flow
```
User types question
    → FastAPI receives query + session_id
    → Embed the query (same embedding model)
    → ChromaDB semantic search → top 5 relevant chunks
    → Build grounded prompt:
        "Answer using ONLY this context: {chunks}
         Question: {question}"
    → Ollama (gemma) generates response
    → Save {user_msg, assistant_msg} to SQLite messages table
    → Return answer + source citations
```

---

## 4. Database Schema

### Table: `documents`
| Column        | Type     | Description                          |
|---------------|----------|--------------------------------------|
| id            | INTEGER  | Primary key (autoincrement)          |
| filename      | TEXT     | Original uploaded filename           |
| file_path     | TEXT     | Local storage path                   |
| file_type     | TEXT     | "pdf", "csv", "txt"                  |
| status        | TEXT     | "pending" / "processing" / "processed" / "failed" |
| chunk_count   | INTEGER  | Number of chunks created             |
| upload_date   | DATETIME | Timestamp of upload                  |
| error_msg     | TEXT     | Error details if failed              |

### Table: `chat_sessions`
| Column        | Type     | Description                          |
|---------------|----------|--------------------------------------|
| id            | INTEGER  | Primary key (autoincrement)          |
| title         | TEXT     | Auto-generated or user-set title     |
| created_at    | DATETIME | Session creation timestamp           |

### Table: `messages`
| Column        | Type     | Description                          |
|---------------|----------|--------------------------------------|
| id            | INTEGER  | Primary key (autoincrement)          |
| session_id    | INTEGER  | Foreign key → chat_sessions.id       |
| role          | TEXT     | "user" / "assistant"                 |
| content       | TEXT     | Message text                         |
| sources       | TEXT     | JSON string of retrieved chunk IDs   |
| timestamp     | DATETIME | Message creation timestamp           |

---

## 5. API Design

### Documents API

| Method | Endpoint                  | Description                   |
|--------|---------------------------|-------------------------------|
| POST   | `/api/documents/upload`   | Upload a document file        |
| GET    | `/api/documents`          | List all documents            |
| GET    | `/api/documents/{id}`     | Get document details + status |
| DELETE | `/api/documents/{id}`     | Delete a document             |

### Chat API

| Method | Endpoint                    | Description                   |
|--------|-----------------------------|-------------------------------|
| POST   | `/api/chat/sessions`        | Create a new chat session     |
| GET    | `/api/chat/sessions`        | List all sessions             |
| GET    | `/api/chat/sessions/{id}`   | Get session + messages        |
| POST   | `/api/chat/query`           | Send a question, get answer   |
| DELETE | `/api/chat/sessions/{id}`   | Delete a session              |

---

## 6. RAG Pipeline Design

### Chunking Strategy
- **Chunk size:** 500 characters (≈125 tokens)
- **Overlap:** 50 characters
- **Splitter:** `RecursiveCharacterTextSplitter` (LangChain)
- **Metadata per chunk:** `{doc_id, filename, chunk_index}`

### Embedding Model
- **Model:** `all-MiniLM-L6-v2` (via sentence-transformers / ChromaDB built-in)
- **Dimension:** 384
- **Why:** Lightweight, fast, high quality for semantic search

### Retrieval
- **Top-k:** 5 chunks per query
- **Distance metric:** Cosine similarity
- **Collection:** One ChromaDB collection per application (`support_docs`)

### Prompt Template
```
You are a helpful research support assistant. Answer the user's question
using ONLY the context provided below. If the answer is not in the context,
say "I don't have enough information in the uploaded documents to answer that."
Be concise and professional.

Context:
{context}

Question: {question}

Answer:
```

---

## 7. Technology Stack

| Layer       | Technology              | Version  | Reason                             |
|-------------|-------------------------|----------|------------------------------------|
| Backend API | FastAPI                 | ≥0.111   | Modern async Python API framework  |
| Validation  | Pydantic                | ≥2.0     | Schema validation, type safety     |
| Database    | SQLite + SQLAlchemy     | ≥2.0     | Zero-config, file-based SQL DB     |
| Vector DB   | ChromaDB                | ≥0.5     | Persistent embedding store         |
| Embeddings  | sentence-transformers   | ≥2.7     | Lightweight local embeddings       |
| LLM         | Ollama (gemma:latest)   | 0.18.0   | Free, local, no API costs          |
| LLM Client  | langchain-ollama        | ≥0.2     | LangChain Ollama integration       |
| Doc Parsing | PyPDF2, pdfplumber      | latest   | PDF text extraction                |
| UI          | Streamlit               | ≥1.35    | Already installed, fast to build   |
| Linting     | Ruff                    | ≥0.4     | Fast Python linter (SDLC rule)     |
| Testing     | pytest                  | ≥8.0     | Unit + integration tests           |

---

## 8. Prompt Engineering Strategy

### Grounding Rules
1. The LLM **must** answer from context only
2. Explicit fallback: "I don't have enough information..."
3. System prompt enforces professional, concise tone

### Context Injection
- Context is limited to top-5 chunks (≤2500 chars) to stay within context window
- Chunks include source filename for citation

### Conversation History
- Last 3 exchanges are prepended to the prompt for continuity
- Full history stored in SQLite for analytics

---

## 9. Background Automation

**Trigger:** Document upload endpoint  
**Handler:** FastAPI `BackgroundTasks`  
**Task:** `process_document_task(doc_id, file_path, file_type)`  
**Steps:**
1. Update document status → "processing"
2. Extract text
3. Chunk text
4. Generate embeddings
5. Store in ChromaDB
6. Update document status → "processed" (or "failed" on error)

---

## 10. Project Structure

```
CapstoneProject/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # App settings (Pydantic Settings)
│   ├── models/
│   │   ├── database.py            # SQLAlchemy ORM models
│   │   └── schemas.py             # Pydantic request/response schemas
│   ├── api/
│   │   ├── documents.py           # Document upload/management router
│   │   └── chat.py                # Chat query/session router
│   ├── services/
│   │   ├── document_processor.py  # Text extraction + chunking
│   │   ├── embeddings.py          # Embedding generation
│   │   ├── vector_store.py        # ChromaDB CRUD
│   │   └── llm_service.py         # Ollama LLM calls
│   ├── tasks/
│   │   └── background.py          # Background document processing task
│   └── db/
│       └── session.py             # SQLAlchemy session factory
├── ui/
│   └── streamlit_app.py           # Streamlit frontend
├── data/
│   └── sample_docs/               # Sample test documents
├── chroma_db/                     # ChromaDB persisted storage
├── tests/
│   ├── test_documents_api.py
│   ├── test_chat_api.py
│   └── test_rag_pipeline.py
├── docs/
│   ├── tech_spec.md               # This document
│   └── architecture.md            # Draw.io exported PNG
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 11. Local Setup Instructions

```bash
# 1. Clone the repository
git clone <repo-url>
cd AML-3303/CapstoneProject

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure Ollama is running with gemma model
ollama serve &
ollama pull gemma

# 5. Start the backend API
uvicorn app.main:app --reload --port 8000

# 6. Start the Streamlit UI (new terminal)
streamlit run ui/streamlit_app.py
```

---

## 12. Team Responsibilities

| Module                        | Primary Owner | Reviewer |
|-------------------------------|---------------|----------|
| FastAPI backend + DB          | All           | All      |
| RAG pipeline (services/)      | All           | All      |
| Streamlit UI                  | All           | All      |
| Testing + documentation       | All           | All      |

*All team members contribute to all modules. Git commits track individual contributions.*

---

## 13. Optional Enhancements (if time allows)
- [ ] Source citations shown in chat UI
- [ ] Document status polling / progress bar
- [ ] Groq API as cloud LLM fallback
- [ ] Analytics dashboard (query counts, response times)
- [ ] Docker Compose for reproducible setup

# Technical Specification — AI Research Support Assistant

**Course:** AML-3303 | Applied Machine Learning  
**Team Size:** 3 students  
**Date:** April 2026  
**Version:** 1.2

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
User uploads file (PDF / CSV / TXT / DOCX / JSON)
    → FastAPI receives file
    → Store metadata in SQLite (status: "pending")
    → Return upload confirmation immediately
    → Background task starts:
        ├── Extract text
        │     PDF  → PyPDF2 + pdfplumber
        │     DOCX → python-docx
        │     CSV  → csv.reader row concatenation
        │     TXT  → plain read
        │     JSON → Elsevier OA parser (see §3.2)
        ├── For JSON: extract_elsevier_metadata()
        │     → {title, pub_year, authors, doi, subjareas}
        ├── Split text into 500-char chunks (50-char overlap)
        ├── Delete existing ChromaDB chunks for this doc_id  ← deduplication
        ├── Embed chunks (all-MiniLM-L6-v2) + store in ChromaDB
        │     with metadata: {doc_id, filename, authors, pub_year, doi}
        └── Update SQLite status to "processed" (or "failed" on error)
```

### 3.2 Elsevier OA JSON Parser

Each Elsevier JSON article is structured as:
```
{
  "doi": "10.1016/...",
  "title": "Article title",
  "authors": [{"given": "...", "surname": "..."}],
  "pub_year": 2021,
  "subjareas": ["MEDI", "COMP"],
  "abstract": "...",
  "highlights": ["..."],
  "body_text": [{"section": "Introduction", "text": "..."}]
}
```

The parser (`document_processor.py:_extract_elsevier_json`) builds a single
text block: metadata header → abstract → highlights → body sections.  
`extract_elsevier_metadata()` returns only ChromaDB-compatible scalar values
(str/int) for storage alongside embeddings.

### 3.3 Bulk Corpus Loading Flow (seed script)
```
python scripts/seed_elsevier.py --reset --n 400 --stratified
    → Scan archive directory, group files by primary subject area
    → Sample ceil(N / num_areas) articles per subject area
    → For each article:
        ├── Call _extract_elsevier_json() + extract_elsevier_metadata()
        ├── Insert document record in SQLite (status: "processed")
        ├── Chunk + embed + store in ChromaDB with rich metadata
        └── Log result  ✓ / ✗
    → Print summary: loaded | skipped | failed
```

> ⚠️ **Important:** Complete the seed before starting the backend.
> Two processes writing to ChromaDB concurrently will corrupt the HNSW index.

### 3.4 Query / RAG Flow
```
User types question
    → FastAPI receives query + session_id
    → Embed the query (same model: all-MiniLM-L6-v2)
    → ChromaDB semantic search → top 5 most similar chunks
    → Build citation strings from chunk metadata:
        └── Has authors/year → "Smith J., Jones M. (2021) — article.json"
        └── No metadata     → "article.json"
    → Build grounded prompt: SYSTEM_PROMPT + history + context + question
    → Ollama (gemma:latest) generates response (~30–90 s on CPU)
    → Save {user_msg, assistant_msg, sources} to SQLite messages table
    → Return {answer, sources[]}
```

---

## 4. Database Schema

### Table: `documents`
| Column        | Type     | Description                          |
|---------------|----------|--------------------------------------|
| id            | INTEGER  | Primary key (autoincrement)          |
| filename      | TEXT     | Original uploaded filename           |
| file_path     | TEXT     | Local storage path                   |
| file_type     | TEXT     | "pdf", "csv", "txt", "docx", "json" |
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
| sources       | TEXT     | JSON array of citation strings       |
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
- **Metadata per chunk:** `{doc_id, filename, chunk_index, authors, pub_year, doi, subjareas}`

### Embedding Model
- **Model:** `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimension:** 384
- **Why:** Lightweight, fast, proven semantic quality for scientific text

### Retrieval
- **Top-k:** 5 chunks per query
- **Distance metric:** Cosine similarity
- **Collection:** Single ChromaDB collection `support_docs`

### Deduplication
Before inserting embeddings for a document, all existing ChromaDB entries with
the same `doc_id` are deleted. This means re-uploading or re-seeding a document
replaces its embeddings rather than creating duplicates.

### Citation Design
Every retrieved chunk carries its source metadata. The `retrieve_chunks()` function
builds a human-readable citation string for each unique source document:

```
# When Elsevier metadata is available:
"Smith J., Jones M., et al. (2021) — S0040402016303829.json"

# For plain uploads without author metadata:
"manual.txt"
```

Citations are returned alongside the answer in the API response and displayed
as collapsible expanders in the Streamlit UI.

### Prompt Template (actual implementation)
```
[SYSTEM]
You are an expert research support assistant helping academics and students
navigate scientific literature from the Elsevier Open Access corpus.

Guidelines:
- Cite sources by mentioning authors and year when available
  (e.g., "According to Smith et al. (2019)...")
- Lead with the direct answer, then supporting evidence.
- Use precise scientific language appropriate to the field.
- If multiple sources address the question, synthesize their key points.
- If context is insufficient, respond with:
  "The uploaded documents do not contain sufficient information..."
- Do NOT fabricate data, citations, or findings.

[Optional] Conversation history: last 3 exchanges

Context from documents:
{top-5 chunks, max 2500 chars}

Question: {user question}

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
| Doc Parsing | python-docx             | latest   | DOCX text extraction               |
| Doc Parsing | built-in json / csv     | —        | JSON (Elsevier OA) and CSV parsing |
| Text Split  | LangChain               | ≥0.2     | `RecursiveCharacterTextSplitter`   |
| UI          | Streamlit               | ≥1.35    | Rapid data-app UI                  |
| Linting     | Ruff                    | ≥0.4     | Fast Python linter (SDLC rule)     |
| Testing     | pytest                  | ≥8.0     | Unit + integration tests (43 total)|

---

## 8. Prompt Engineering Strategy

### Grounding Rules
1. The LLM **must** answer from context only
2. Explicit fallback: "I don't have enough information..."
3. System prompt enforces professional, concise tone

### Context Injection
- Context is limited to top-5 chunks (≤2500 chars) to stay within context window
- Each chunk carries author, year, DOI, and subject area metadata
- Citation strings are built from this metadata and returned with every answer

### Conversation History
- Last 3 exchanges are prepended to the prompt for continuity
- Full history stored in SQLite for analytics

---

## 9. Background Automation

**Trigger:** Document upload endpoint  
**Handler:** FastAPI `BackgroundTasks`  
**Task:** `process_document_task(doc_id, file_path, file_type)`  
**Steps:**
1. Update document status → `"processing"`
2. Extract text (`extract_text(path, type)`)
3. For JSON files: extract Elsevier metadata (`extract_elsevier_metadata(path)`)
4. Chunk text with `RecursiveCharacterTextSplitter` (500 chars / 50 overlap)
5. Delete existing ChromaDB entries for this `doc_id` (deduplication)
6. Generate embeddings and store chunks in ChromaDB with metadata
7. Update SQLite `chunk_count` and status → `"processed"` (or `"failed"` on error)

**Second automation path — bulk seed script:**  
`scripts/seed_elsevier.py` bypasses the HTTP API and writes directly to SQLite
and ChromaDB in a single process. Supports `--stratified` sampling across all
27 Elsevier subject area codes and `--reset` for a clean rebuild.

---

## 10. Project Structure

```
CapstoneProject/
├── app/
│   ├── main.py                    # FastAPI entry point + lifespan
│   ├── config.py                  # App settings (Pydantic Settings)
│   ├── models/
│   │   ├── database.py            # SQLAlchemy ORM models
│   │   └── schemas.py             # Pydantic request/response schemas
│   ├── api/
│   │   ├── documents.py           # Document upload/management router
│   │   └── chat.py                # Chat query/session router
│   ├── services/
│   │   ├── document_processor.py  # Text extraction, chunking, Elsevier parser
│   │   ├── vector_store.py        # ChromaDB CRUD + dedup + citations
│   │   └── llm_service.py         # Ollama LLM integration + system prompt
│   ├── tasks/
│   │   └── background.py          # Background processing task
│   └── db/
│       └── session.py             # SQLAlchemy session factory
├── scripts/
│   └── seed_elsevier.py           # Bulk Elsevier OA corpus loader
├── ui/
│   └── streamlit_app.py           # Streamlit frontend
├── data/
│   └── sample_docs/               # 5 sample Elsevier OA articles
├── chroma_db/                     # ChromaDB persisted storage (gitignored)
├── logs/                          # Backend + seed run logs (gitignored)
├── tests/
│   ├── test_documents_api.py      # 12 document API tests
│   ├── test_chat_api.py           # 16 chat API tests
│   └── test_rag_pipeline.py       # 15 RAG pipeline unit tests
├── docs/
│   └── tech_spec.md               # This document
├── pytest.ini
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

# 5. (Optional) Seed the knowledge base with Elsevier OA articles
#    Requires the corpus downloaded to ~/Downloads/archive/json/json/
#    Run this BEFORE starting the backend.
python scripts/seed_elsevier.py --reset --n 400 --stratified   # ~4 min
# python scripts/seed_elsevier.py --reset --n 10000 --stratified  # ~4 hours

# 6. Start the backend API (port 8001)
uvicorn app.main:app --reload --port 8001

# 7. Start the Streamlit UI (new terminal)
streamlit run ui/streamlit_app.py
```

API docs (Swagger UI): http://localhost:8001/docs  
Streamlit UI: http://localhost:8501

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

## 13. Optional Enhancements
- [x] Source citations shown in chat UI (author + year + filename)
- [x] Document status polling with status icons (✅ ⏳ 🕐 ❌)
- [x] Deduplication — re-upload replaces embeddings, no duplicates
- [x] Research-domain system prompt with scientific citation style
- [x] Stratified bulk-loading across all 27 Elsevier subject area codes
- [ ] Analytics dashboard (query counts, response times, subject area breakdown)
- [ ] Groq API as cloud LLM fallback
- [ ] Docker Compose for reproducible setup

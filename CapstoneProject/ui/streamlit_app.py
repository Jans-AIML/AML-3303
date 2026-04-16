"""Streamlit frontend for the AI Research Support Assistant.

Best-practice version for minimizing GET calls:
- Load sessions/documents/messages into st.session_state
- Render from local state
- Update local state after POST/DELETE actions
- Only re-fetch from backend on initial load, manual refresh, or session switch

Run with:
    streamlit run ui/streamlit_app.py --server.port 8001
"""

from __future__ import annotations

import json

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8501"
LLM_TIMEOUT = 180
UPLOAD_TIMEOUT = 30
SHORT_TIMEOUT = 10

st.set_page_config(
    page_title="AI Research Support Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    .stChatMessage { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Session state initialization
# ---------------------------
DEFAULTS = {
    "page": "💬 Chat",
    "session_id": None,
    "_uploader_key": 0,
    "health_data": None,
    "sessions_data": [],
    "documents_data": [],
    "messages_data": [],
    "current_session_title": "Chat",
    "pending_assistant_message": None,
    "last_loaded_session_id": None,
    "initialized": False,
    "just_uploaded": None,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------
# API helpers
# ---------------------------
def _api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=SHORT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Cannot reach the backend. Make sure `uvicorn app.main:app --reload --port 8501` is running."
        )
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


@st.cache_data(ttl=15, show_spinner=False)
def cached_api_get(path: str) -> dict | None:
    return _api_get(path)


def clear_api_cache() -> None:
    cached_api_get.clear()


def _api_post(
    path: str,
    *,
    payload: dict | None = None,
    files=None,
    timeout: int = SHORT_TIMEOUT,
) -> dict | None:
    try:
        kwargs: dict = {"timeout": timeout}
        if files:
            kwargs["files"] = files
        else:
            kwargs["json"] = payload
        r = requests.post(f"{API_BASE}{path}", **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Cannot reach the backend. Make sure `uvicorn app.main:app --reload --port 8501` is running."
        )
        return None
    except requests.exceptions.Timeout:
        st.error(
            "⏱️ Request timed out. The server is still working — try refreshing in a moment."
        )
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _api_delete(path: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=SHORT_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as exc:
        st.error(f"Delete failed: {exc}")
        return False


# ---------------------------
# Data loading helpers
# ---------------------------
def load_health(force: bool = False) -> None:
    if force:
        clear_api_cache()
    data = cached_api_get("/health")
    if data is not None:
        st.session_state["health_data"] = data


def load_sessions(force: bool = False) -> None:
    if force:
        clear_api_cache()
    data = cached_api_get("/api/chat/sessions")
    if data:
        st.session_state["sessions_data"] = data.get("sessions", [])
        if st.session_state["sessions_data"]:
            valid_ids = {s["id"] for s in st.session_state["sessions_data"]}
            if st.session_state["session_id"] not in valid_ids:
                st.session_state["session_id"] = st.session_state["sessions_data"][0]["id"]
        else:
            st.session_state["session_id"] = None


def load_documents(force: bool = False) -> None:
    if force:
        clear_api_cache()
    data = cached_api_get("/api/documents")
    if data:
        st.session_state["documents_data"] = data.get("documents", [])


def load_session_detail(session_id: int, force: bool = False) -> None:
    if session_id is None:
        st.session_state["messages_data"] = []
        st.session_state["current_session_title"] = "Chat"
        st.session_state["last_loaded_session_id"] = None
        return

    if not force and st.session_state["last_loaded_session_id"] == session_id:
        return

    if force:
        clear_api_cache()

    data = cached_api_get(f"/api/chat/sessions/{session_id}")
    if data:
        st.session_state["messages_data"] = data.get("messages", [])
        st.session_state["current_session_title"] = data.get("title", f"Session {session_id}")
        st.session_state["last_loaded_session_id"] = session_id
        st.session_state["pending_assistant_message"] = None


def initialize_data() -> None:
    if st.session_state["initialized"]:
        return

    load_health()
    load_sessions()
    load_documents()

    if st.session_state["session_id"] is not None:
        load_session_detail(st.session_state["session_id"], force=False)

    st.session_state["initialized"] = True


def refresh_all() -> None:
    clear_api_cache()
    load_health(force=False)
    load_sessions(force=False)
    load_documents(force=False)
    if st.session_state["session_id"] is not None:
        st.session_state["last_loaded_session_id"] = None
        load_session_detail(st.session_state["session_id"], force=False)


# ---------------------------
# Analytics page
# ---------------------------

def render_analytics() -> None:
    st.title("📊 Analytics Dashboard")
    st.caption("Live statistics from the SQLite database.")

    data = cached_api_get("/api/documents/analytics")
    if not data:
        st.warning("Could not load analytics. Make sure the backend is running.")
        return

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Documents", f"{data['total_documents']:,}")
    c2.metric("🔢 Chunks Indexed", f"{data['total_chunks']:,}")
    c3.metric("💬 Chat Sessions", data["total_sessions"])
    c4.metric("❓ Queries Answered", data["total_queries"])

    if data["avg_chunks_per_doc"] > 0:
        st.caption(
            f"Average **{data['avg_chunks_per_doc']}** chunks per processed document · "
            f"{data['total_chunks']:,} total vectors in ChromaDB"
        )

    st.divider()

    # ── Status + Type breakdown ─────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🟡 Documents by Status")
        if data["by_status"]:
            df_status = pd.DataFrame.from_dict(
                data["by_status"], orient="index", columns=["Count"]
            ).sort_values("Count", ascending=False)
            st.bar_chart(df_status, color="#4CAF50")
        else:
            st.caption("No data yet.")

    with col_right:
        st.subheader("📂 Documents by File Type")
        if data["by_type"]:
            df_type = pd.DataFrame.from_dict(
                data["by_type"], orient="index", columns=["Count"]
            ).sort_values("Count", ascending=False)
            st.bar_chart(df_type, color="#2196F3")
        else:
            st.caption("No data yet.")

    st.divider()

    # ── Top documents table ─────────────────────────────────────────────────
    st.subheader("🏆 Top 10 Documents by Chunk Count")
    if data["top_docs_by_chunks"]:
        df_top = pd.DataFrame(data["top_docs_by_chunks"])
        df_top["filename"] = df_top["filename"].str.replace(".json", "", regex=False)
        df_top["filename"] = df_top["filename"].str[:30]
        df_top = df_top.set_index("filename").rename(columns={"chunk_count": "Chunks"})
        st.bar_chart(df_top, color="#FF9800")
    else:
        st.caption("No documents available.")

    st.divider()

    # ── Raw detail table ──────────────────────────────────────────────────
    with st.expander("📝 Full status breakdown", expanded=False):
        st.write("**By Processing Status**")
        st.dataframe(
            pd.DataFrame.from_dict(data["by_status"], orient="index", columns=["Count"]),
            use_container_width=True,
        )
        st.write("**By File Type**")
        st.dataframe(
            pd.DataFrame.from_dict(data["by_type"], orient="index", columns=["Count"]),
            use_container_width=True,
        )


# ---------------------------
# UI helpers
# ---------------------------
def render_sources(sources: list) -> None:
    if not sources:
        return

    with st.expander("📚 Sources", expanded=False):
        for i, src in enumerate(sources, start=1):
            if isinstance(src, dict):
                filename = src.get("filename", "unknown")
                chunk_index = src.get("chunk_index", "?")
                title = src.get("title")
                authors = src.get("authors")
                pub_year = src.get("pub_year")
                doi = src.get("doi")
                chunk_text = src.get("chunk_text", "").strip()

                st.markdown(f"**{i}. {filename} — chunk {chunk_index}**")

                meta_parts = []
                if title:
                    meta_parts.append(f"Title: {title}")
                if authors:
                    meta_parts.append(f"Authors: {authors}")
                if pub_year:
                    meta_parts.append(f"Year: {pub_year}")
                if doi:
                    meta_parts.append(f"DOI: {doi}")

                if meta_parts:
                    st.caption(" | ".join(meta_parts))

                if chunk_text:
                    st.code(chunk_text, language="text")
                else:
                    st.caption("No chunk text available.")
            else:
                st.caption(f"• {src}")


def append_local_user_message(question: str) -> None:
    st.session_state["messages_data"].append(
        {
            "role": "user",
            "content": question,
            "sources": None,
        }
    )


def append_local_assistant_message(answer: str, sources: list, message_id: int | None = None) -> None:
    stored_sources = json.dumps(sources)
    st.session_state["messages_data"].append(
        {
            "id": message_id,
            "role": "assistant",
            "content": answer,
            "sources": stored_sources,
        }
    )


# ---------------------------
# Initial data load
# ---------------------------
initialize_data()


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("AML-3303 Capstone · Lambton College · April 2026")

    st.session_state["page"] = st.radio(
        "Navigate",
        ["💬 Chat", "📊 Analytics"],
        horizontal=True,
        label_visibility="collapsed",
        index=0 if st.session_state["page"] == "💬 Chat" else 1,
    )

    st.divider()

    health = st.session_state["health_data"]
    if health:
        ollama_icon = "🟢" if health.get("ollama_available") else "🟡"
        st.caption(
            f"Backend: **{health.get('status', 'unknown')}** | "
            f"Ollama: {ollama_icon} | "
            f"Docs indexed: **{health.get('documents_indexed', 0)}**"
        )

    if st.button("🔄 Refresh data", use_container_width=True):
        refresh_all()
        st.rerun()

    st.divider()
    st.subheader("💬 Chat Sessions")

    if st.button("➕ New Session", use_container_width=True):
        new_session = _api_post(
            "/api/chat/sessions",
            payload={"title": "New Chat"},
            timeout=SHORT_TIMEOUT,
        )
        if new_session:
            clear_api_cache()
            new_session_item = {
                "id": new_session["id"],
                "title": new_session["title"],
                "created_at": new_session.get("created_at"),
            }
            st.session_state["sessions_data"] = [new_session_item] + st.session_state["sessions_data"]
            st.session_state["session_id"] = new_session["id"]
            st.session_state["messages_data"] = []
            st.session_state["current_session_title"] = new_session["title"]
            st.session_state["last_loaded_session_id"] = new_session["id"]
            st.session_state["pending_assistant_message"] = None
            st.rerun()

    sessions = st.session_state["sessions_data"]

    if sessions:
        session_map = {s["id"]: f"#{s['id']} — {s['title']}" for s in sessions}
        ids = list(session_map.keys())

        selected_id = st.selectbox(
            "Active session",
            options=ids,
            format_func=lambda x: session_map[x],
            index=ids.index(st.session_state["session_id"]) if st.session_state["session_id"] in ids else 0,
            label_visibility="collapsed",
        )

        if selected_id != st.session_state["session_id"]:
            st.session_state["session_id"] = selected_id
            st.session_state["pending_assistant_message"] = None
            st.session_state["last_loaded_session_id"] = None
            load_session_detail(selected_id, force=False)
            st.rerun()

        if st.button("🗑️ Delete Session", use_container_width=True, type="secondary"):
            if _api_delete(f"/api/chat/sessions/{selected_id}"):
                clear_api_cache()
                st.session_state["sessions_data"] = [
                    s for s in st.session_state["sessions_data"] if s["id"] != selected_id
                ]
                if st.session_state["sessions_data"]:
                    next_id = st.session_state["sessions_data"][0]["id"]
                    st.session_state["session_id"] = next_id
                    st.session_state["last_loaded_session_id"] = None
                    load_session_detail(next_id, force=False)
                else:
                    st.session_state["session_id"] = None
                    st.session_state["messages_data"] = []
                    st.session_state["current_session_title"] = "Chat"
                    st.session_state["last_loaded_session_id"] = None
                st.rerun()
    else:
        st.caption("No sessions yet — create one above.")

    st.divider()
    st.subheader("📄 Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "csv", "docx", "json"],
        help="PDF, TXT, CSV, DOCX, or Elsevier JSON",
        label_visibility="collapsed",
        key=f"uploader_{st.session_state['_uploader_key']}",
    )

    if uploaded_file:
        resp = _api_post(
            "/api/documents/upload",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            },
            timeout=UPLOAD_TIMEOUT,
        )
        if resp:
            clear_api_cache()
            st.session_state["just_uploaded"] = resp["filename"]
            st.session_state["_uploader_key"] += 1
            load_documents(force=False)
            load_health(force=False)
            st.rerun()

    if st.session_state["just_uploaded"]:
        st.success(f"✓ Uploaded **{st.session_state['just_uploaded']}** — processing in background.")
        st.session_state["just_uploaded"] = None

    docs = st.session_state["documents_data"]
    STATUS_ICON = {
        "processed": "✅",
        "processing": "⏳",
        "pending": "🕐",
        "failed": "❌",
    }

    if docs:
        for doc in docs:
            icon = STATUS_ICON.get(doc.get("status"), "❓")
            display_name = doc.get("filename", "unknown")
            if len(display_name) > 28:
                display_name = display_name[:25] + "…"

            col_name, col_del = st.columns([5, 1])
            with col_name:
                chunks = f" ({doc['chunk_count']} chunks)" if doc.get("chunk_count") else ""
                st.caption(f"{icon} {display_name}{chunks}")
            with col_del:
                if st.button("✕", key=f"del_doc_{doc['id']}", help="Delete document"):
                    if _api_delete(f"/api/documents/{doc['id']}"):
                        clear_api_cache()
                        st.session_state["documents_data"] = [
                            d for d in st.session_state["documents_data"] if d["id"] != doc["id"]
                        ]
                        load_health(force=False)
                        st.rerun()
    else:
        st.caption("No documents loaded yet.")

    st.info(
        "Tip: run `python scripts/seed_elsevier.py` to bulk-load articles from the Elsevier OA archive.",
        icon="💡",
    )


# ---------------------------
# Page routing
# ---------------------------
if st.session_state["page"] == "📊 Analytics":
    render_analytics()
    st.stop()


# ---------------------------
# Chat page
# ---------------------------
if st.session_state["session_id"] is None:
    st.title("🔬 AI Research Support Assistant")
    st.markdown(
        """
Welcome! This assistant answers questions grounded in your uploaded research documents.

**To get started:**
1. Click **➕ New Session** in the sidebar
2. Upload documents (or run the Elsevier seed script)
3. Ask a question below
"""
    )
    st.stop()

st.title(f"💬 {st.session_state['current_session_title']}")

for msg in st.session_state["messages_data"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            try:
                sources = json.loads(msg["sources"])
                render_sources(sources)
            except (json.JSONDecodeError, TypeError):
                pass

user_input = st.chat_input("Ask a question about the research documents…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    append_local_user_message(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            result = _api_post(
                "/api/chat/query",
                payload={"session_id": st.session_state["session_id"], "question": user_input},
                timeout=LLM_TIMEOUT,
            )

        if result:
            clear_api_cache()
            answer = result["answer"]
            sources = result.get("sources", [])
            append_local_assistant_message(
                answer=answer,
                sources=sources,
                message_id=result.get("message_id"),
            )
            st.markdown(answer)
            render_sources(sources)
        else:
            st.error(
                "Could not get a response. "
                "Check that the FastAPI backend and Ollama are both running."
            )

API_BASE = "http://localhost:8501"
LLM_TIMEOUT = 180     # seconds — gemma on CPU can take 30-90 s
UPLOAD_TIMEOUT = 30   # seconds — upload returns 202 immediately
SHORT_TIMEOUT = 10    # seconds — read-only calls

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Support Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Small CSS tweaks ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .stChatMessage { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API helpers ───────────────────────────────────────────────────────────────

def _api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=SHORT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot reach the backend. Make sure `uvicorn app.main:app --reload --port 8501` is running.")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _api_post(
    path: str,
    *,
    payload: dict | None = None,
    files=None,
    timeout: int = SHORT_TIMEOUT,
) -> dict | None:
    try:
        kwargs: dict = {"timeout": timeout}
        if files:
            kwargs["files"] = files
        else:
            kwargs["json"] = payload
        r = requests.post(f"{API_BASE}{path}", **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot reach the backend. Make sure `uvicorn app.main:app --reload --port 8501` is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The server is still working — try refreshing in a moment.")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _api_delete(path: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=SHORT_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as exc:
        st.error(f"Delete failed: {exc}")
        return False


# ── Analytics page ──────────────────────────────────────────────────────────────

def render_analytics() -> None:
    st.title("📊 Analytics Dashboard")
    st.caption("Live statistics from the SQLite database.")

    data = _api_get("/api/documents/analytics")
    if not data:
        st.warning("Could not load analytics. Make sure the backend is running.")
        return

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Documents", f"{data['total_documents']:,}")
    c2.metric("🔢 Chunks Indexed", f"{data['total_chunks']:,}")
    c3.metric("💬 Chat Sessions", data["total_sessions"])
    c4.metric("❓ Queries Answered", data["total_queries"])

    if data["avg_chunks_per_doc"] > 0:
        st.caption(
            f"Average **{data['avg_chunks_per_doc']}** chunks per processed document · "
            f"{data['total_chunks']:,} total vectors in ChromaDB"
        )

    st.divider()

    # ── Status + Type breakdown ─────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🟡 Documents by Status")
        if data["by_status"]:
            df_status = pd.DataFrame.from_dict(
                data["by_status"], orient="index", columns=["Count"]
            ).sort_values("Count", ascending=False)
            st.bar_chart(df_status, color="#4CAF50")
        else:
            st.caption("No data yet.")

    with col_right:
        st.subheader("📂 Documents by File Type")
        if data["by_type"]:
            df_type = pd.DataFrame.from_dict(
                data["by_type"], orient="index", columns=["Count"]
            ).sort_values("Count", ascending=False)
            st.bar_chart(df_type, color="#2196F3")
        else:
            st.caption("No data yet.")

    st.divider()

    # ── Top documents table ─────────────────────────────────────────────────
    st.subheader("🏆 Top 10 Documents by Chunk Count")
    if data["top_docs_by_chunks"]:
        df_top = pd.DataFrame(data["top_docs_by_chunks"])
        # Truncate long Elsevier filenames for readability
        df_top["filename"] = df_top["filename"].str.replace(".json", "", regex=False)
        df_top["filename"] = df_top["filename"].str[:30]
        df_top = df_top.set_index("filename").rename(columns={"chunk_count": "Chunks"})
        st.bar_chart(df_top, color="#FF9800")
    else:
        st.caption("No documents available.")

    st.divider()

    # ── Raw detail table ──────────────────────────────────────────────────
    with st.expander("📝 Full status breakdown", expanded=False):
        st.write("**By Processing Status**")
        st.dataframe(
            pd.DataFrame.from_dict(data["by_status"], orient="index", columns=["Count"]),
            use_container_width=True,
        )
        st.write("**By File Type**")
        st.dataframe(
            pd.DataFrame.from_dict(data["by_type"], orient="index", columns=["Count"]),
            use_container_width=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("AML-3303 Capstone · Lambton College · April 2026")

    # ── Page selector ─────────────────────────────────────────────
    page = st.radio(
        "Navigate",
        ["💬 Chat", "📊 Analytics"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    # Backend status
    health = _api_get("/health")
    if health:
        ollama_icon = "🟢" if health.get("ollama_available") else "🟡"
        st.caption(
            f"Backend: **{health['status']}** | "
            f"Ollama: {ollama_icon} | "
            f"Docs indexed: **{health.get('documents_indexed', 0)}**"
        )
    st.divider()

    # ── Chat sessions ───────────────────────────────────────────────────────
    st.subheader("💬 Chat Sessions")

    if st.button("➕ New Session", use_container_width=True):
        new_session = _api_post("/api/chat/sessions", payload={"title": "New Chat"}, timeout=SHORT_TIMEOUT)
        if new_session:
            st.session_state["session_id"] = new_session["id"]
            st.rerun()

    sessions_data = _api_get("/api/chat/sessions")
    sessions = sessions_data["sessions"] if sessions_data else []

    if sessions:
        session_map = {s["id"]: f"#{s['id']} — {s['title']}" for s in sessions}

        # Default to the first session if none is selected
        if "session_id" not in st.session_state or st.session_state["session_id"] not in session_map:
            st.session_state["session_id"] = sessions[0]["id"]

        selected_id = st.selectbox(
            "Active session",
            options=list(session_map.keys()),
            format_func=lambda x: session_map[x],
            index=list(session_map.keys()).index(st.session_state["session_id"]),
            label_visibility="collapsed",
        )
        st.session_state["session_id"] = selected_id

        if st.button("🗑️ Delete Session", use_container_width=True, type="secondary"):
            if _api_delete(f"/api/chat/sessions/{selected_id}"):
                st.session_state.pop("session_id", None)
                st.rerun()
    else:
        st.caption("No sessions yet — create one above.")

    st.divider()

    # ── Document management ─────────────────────────────────────────────────
    st.subheader("📄 Documents")

    # Key rotates after each successful upload so the widget resets and
    # does NOT re-trigger on st.rerun() (which would cause an infinite loop).
    if "_uploader_key" not in st.session_state:
        st.session_state["_uploader_key"] = 0

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "csv", "docx", "json"],
        help="PDF, TXT, CSV, DOCX, or Elsevier JSON",
        label_visibility="collapsed",
        key=f"uploader_{st.session_state['_uploader_key']}",
    )
    if uploaded_file:
        resp = _api_post(
            "/api/documents/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            timeout=UPLOAD_TIMEOUT,
        )
        if resp:
            st.success(f"✓ Uploaded **{resp['filename']}** — processing in background.")
            st.session_state["_uploader_key"] += 1  # reset the uploader widget
            st.rerun()

    docs_data = _api_get("/api/documents")
    docs = docs_data["documents"] if docs_data else []

    STATUS_ICON = {"processed": "✅", "processing": "⏳", "pending": "🕐", "failed": "❌"}

    if docs:
        for doc in docs:
            icon = STATUS_ICON.get(doc["status"], "❓")
            display_name = doc["filename"]
            if len(display_name) > 28:
                display_name = display_name[:25] + "…"
            col_name, col_del = st.columns([5, 1])
            with col_name:
                chunks = f" ({doc['chunk_count']} chunks)" if doc.get("chunk_count") else ""
                st.caption(f"{icon} {display_name}{chunks}")
            with col_del:
                if st.button("✕", key=f"del_doc_{doc['id']}", help="Delete document"):
                    if _api_delete(f"/api/documents/{doc['id']}"):
                        st.rerun()
    else:
        st.caption("No documents loaded yet.")
        st.info(
            "Tip: run `python scripts/seed_elsevier.py` to bulk-load "
            "articles from the Elsevier OA archive.",
            icon="💡",
        )


# ── Page routing ──────────────────────────────────────────────────────────────
if page == "📊 Analytics":
    render_analytics()
    st.stop()

# ── Main chat area ────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.title("🔬 AI Research Support Assistant")
    st.markdown(
        """
        Welcome! This assistant answers questions grounded in your uploaded research documents.

        **To get started:**
        1. Click **➕ New Session** in the sidebar
        2. Upload documents (or run the Elsevier seed script)
        3. Ask a question below
        """
    )
    st.stop()

session_id: int = st.session_state["session_id"]

# Fetch session details (title + message history)
session_detail = _api_get(f"/api/chat/sessions/{session_id}")
if not session_detail:
    st.error("Session not found — it may have been deleted.")
    st.stop()

st.title(f"💬 {session_detail['title']}")

# ── Render message history ────────────────────────────────────────────────────

for msg in session_detail.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            try:
                sources = json.loads(msg["sources"])
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        for src in sources:
                            st.caption(f"• {src}")
            except (json.JSONDecodeError, TypeError):
                pass

# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask a question about the research documents…")

if user_input:
    # Show user turn immediately (before API roundtrip)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer… (this can take up to 60 s)"):
            result = _api_post(
                "/api/chat/query",
                payload={"session_id": session_id, "question": user_input},
                timeout=LLM_TIMEOUT,
            )

        if result:
            st.markdown(result["answer"])
            sources = result.get("sources", [])
            if sources:
                with st.expander("📚 Sources", expanded=False):
                    for src in sources:
                        st.caption(f"• {src}")
            st.rerun()  # refresh history only on success
        else:
            st.error(
                "Could not get a response. "
                "Check that the FastAPI backend and Ollama are both running."
            )

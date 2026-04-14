"""Streamlit frontend for the AI Research Support Assistant.

Run with:
    streamlit run ui/streamlit_app.py --server.port 8001
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8501")
LLM_TIMEOUT = 180
UPLOAD_TIMEOUT = 30
SHORT_TIMEOUT = 10
CACHE_TTL = 5

st.set_page_config(
    page_title="AI Research Support Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .stChatMessage { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "_http" not in st.session_state:
    st.session_state["_http"] = requests.Session()


def _http() -> requests.Session:
    return st.session_state["_http"]


def _request(method: str, path: str, *, timeout: int, **kwargs) -> requests.Response | None:
    try:
        response = _http().request(method, f"{API_BASE}{path}", timeout=timeout, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot reach the backend. Make sure `uvicorn app.main:app --reload --port 8501` is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The server is still working — try refreshing in a moment.")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _cached_get(path: str) -> dict[str, Any] | None:
    response = _request("GET", path, timeout=SHORT_TIMEOUT)
    if response is None:
        return None
    return response.json()


def _api_get(path: str, *, use_cache: bool = True) -> dict[str, Any] | None:
    if use_cache:
        return _cached_get(path)
    response = _request("GET", path, timeout=SHORT_TIMEOUT)
    if response is None:
        return None
    return response.json()


def _api_post(path: str, *, payload: dict | None = None, files=None, timeout: int = SHORT_TIMEOUT) -> dict | None:
    kwargs: dict[str, Any] = {}
    if files:
        kwargs["files"] = files
    else:
        kwargs["json"] = payload
    response = _request("POST", path, timeout=timeout, **kwargs)
    if response is None:
        return None
    _cached_get.clear()
    return response.json()


def _api_delete(path: str) -> bool:
    response = _request("DELETE", path, timeout=SHORT_TIMEOUT)
    if response is None:
        return False
    _cached_get.clear()
    return True


def render_analytics() -> None:
    st.title("📊 Analytics Dashboard")
    st.caption("Live statistics from the SQLite database.")

    data = _api_get("/api/documents/analytics", use_cache=True)
    if not data:
        st.warning("Could not load analytics. Make sure the backend is running.")
        return

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
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🟡 Documents by Status")
        if data["by_status"]:
            df_status = pd.DataFrame.from_dict(data["by_status"], orient="index", columns=["Count"]).sort_values("Count", ascending=False)
            st.bar_chart(df_status, color="#4CAF50")
        else:
            st.caption("No data yet.")

    with col_right:
        st.subheader("📂 Documents by File Type")
        if data["by_type"]:
            df_type = pd.DataFrame.from_dict(data["by_type"], orient="index", columns=["Count"]).sort_values("Count", ascending=False)
            st.bar_chart(df_type, color="#2196F3")
        else:
            st.caption("No data yet.")

    st.divider()
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
    with st.expander("📝 Full status breakdown", expanded=False):
        st.write("**By Processing Status**")
        st.dataframe(pd.DataFrame.from_dict(data["by_status"], orient="index", columns=["Count"]), use_container_width=True)
        st.write("**By File Type**")
        st.dataframe(pd.DataFrame.from_dict(data["by_type"], orient="index", columns=["Count"]), use_container_width=True)


with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("AML-3303 Capstone · Lambton College · April 2026")

    page = st.radio("Navigate", ["💬 Chat", "📊 Analytics"], horizontal=True, label_visibility="collapsed")
    st.divider()

    if st.button("🔄 Refresh sidebar", use_container_width=True):
        _cached_get.clear()
        st.rerun()

    health = _api_get("/health", use_cache=True)
    if health:
        ollama_icon = "🟢" if health.get("ollama_available") else "🟡"
        st.caption(
            f"Backend: **{health['status']}** | "
            f"Ollama: {ollama_icon} | "
            f"Docs indexed: **{health.get('documents_indexed', 0)}**"
        )
    st.divider()

    st.subheader("💬 Chat Sessions")

    if st.button("➕ New Session", use_container_width=True):
        new_session = _api_post("/api/chat/sessions", payload={"title": "New Chat"}, timeout=SHORT_TIMEOUT)
        if new_session:
            st.session_state["session_id"] = new_session["id"]
            st.rerun()

    sessions_data = _api_get("/api/chat/sessions", use_cache=True)
    sessions = sessions_data["sessions"] if sessions_data else []

    if sessions:
        session_map = {s["id"]: f"#{s['id']} — {s['title']}" for s in sessions}
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
    st.subheader("📄 Documents")

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
            st.session_state["_uploader_key"] += 1
            st.rerun()

    docs_data = _api_get("/api/documents", use_cache=True)
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
        st.info("Tip: run `python scripts/seed_elsevier.py` to bulk-load articles from the Elsevier OA archive.", icon="💡")

if page == "📊 Analytics":
    render_analytics()
    st.stop()

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
session_detail = _api_get(f"/api/chat/sessions/{session_id}", use_cache=False)
if not session_detail:
    st.error("Session not found — it may have been deleted.")
    st.stop()

st.title(f"💬 {session_detail['title']}")

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

user_input = st.chat_input("Ask a question about the research documents…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer… (this can take up to 60 s)"):
            result = _api_post("/api/chat/query", payload={"session_id": session_id, "question": user_input}, timeout=LLM_TIMEOUT)

        if result:
            st.markdown(result["answer"])
            sources = result.get("sources", [])
            if sources:
                with st.expander("📚 Sources", expanded=False):
                    for src in sources:
                        st.caption(f"• {src}")
            st.rerun()
        else:
            st.error("Could not get a response. Check that the FastAPI backend and Ollama are both running.")
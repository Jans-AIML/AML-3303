"""Streamlit frontend for the AI Research Support Assistant.

Run with:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import json

import requests
import streamlit as st

API_BASE = "http://localhost:8001"
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
        st.error("⚠️ Cannot reach the backend. Make sure `uvicorn app.main:app --reload --port 8001` is running.")
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
        st.error("⚠️ Cannot reach the backend. Make sure `uvicorn app.main:app --reload --port 8001` is running.")
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("AML-3303 Capstone · Lambton College · April 2026")

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

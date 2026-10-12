"""
DocuMind-RAG v2 - Premium No-Sidebar Streamlit Application
Author: D Rohith Subrahmanya Sai
GitHub: https://github.com/DRohithSubrahmanyaSai/DocuMind-RAG

Everything lives in the main content area. No sidebar.
Run: streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import uuid
import time
import tempfile
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT_DIR / ".env")

import streamlit as st


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
PREMIUM_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base: #06080f;
    --bg-surface: #0c1019;
    --bg-card: rgba(14,18,30,0.82);
    --bg-card-hover: rgba(22,28,48,0.88);
    --border: rgba(255,255,255,0.06);
    --border-accent: rgba(99,102,241,0.35);
    --accent: #818cf8;
    --accent2: #c084fc;
    --accent3: #f472b6;
    --cyan: #22d3ee;
    --emerald: #34d399;
    --text: #f1f5f9;
    --text2: #94a3b8;
    --text3: #64748b;
    --glow: 0 0 40px rgba(99,102,241,0.12);
    --r-sm: 10px;
    --r-md: 16px;
    --r-lg: 22px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide sidebar completely */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[kind="headerNoPadding"] {
    display: none !important;
    width: 0 !important;
    min-width: 0 !important;
}

/* Main container */
.main .block-container {
    padding: 1.2rem 2rem 6rem 2rem !important;
    max-width: 1200px !important;
}

/* ─── Chat input bar — professional alignment & styling ─── */

/* The bottom sticky container Streamlit renders */
[data-testid="stBottom"] {
    background: linear-gradient(to top, rgba(6,8,15,0.97) 75%, transparent) !important;
    backdrop-filter: blur(14px) !important;
    border-top: 1px solid rgba(99,102,241,0.12) !important;
    padding: 12px 1.5rem 16px !important;
}

/* Constrain the inner container width to match the page content */
[data-testid="stBottom"] > div {
    max-width: 1200px !important;
    margin: 0 auto !important;
    width: 100% !important;
}

/* The chat input component wrapper */
[data-testid="stChatInput"] {
    background: rgba(12,16,25,0.95) !important;
    border: 1.5px solid rgba(99,102,241,0.28) !important;
    border-radius: 16px !important;
    box-shadow:
        0 0 0 1px rgba(99,102,241,0.06) inset,
        0 8px 32px rgba(0,0,0,0.5) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    overflow: hidden !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(129,140,248,0.55) !important;
    box-shadow:
        0 0 0 3px rgba(99,102,241,0.12) inset,
        0 8px 32px rgba(0,0,0,0.5) !important;
}

/* Textarea inside the input */
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 0.93rem !important;
    font-weight: 400 !important;
    line-height: 1.5 !important;
    padding: 14px 18px 14px 18px !important;
    caret-color: #818cf8 !important;
    border: none !important;
    outline: none !important;
    resize: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #3d4f6b !important;
    font-weight: 400 !important;
    font-size: 0.9rem !important;
}

/* Send / submit button */
[data-testid="stChatInput"] button[kind="primaryFormSubmit"],
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    width: 38px !important;
    height: 38px !important;
    min-width: 38px !important;
    margin: 6px 8px 6px 4px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.45) !important;
    transition: all 0.18s ease !important;
    cursor: pointer !important;
}

[data-testid="stChatInput"] button:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.6) !important;
    transform: scale(1.06) !important;
}

[data-testid="stChatInput"] button svg {
    fill: white !important;
    color: white !important;
}

/* Bottom padding on main content so last message is NEVER behind input */
.main .block-container {
    padding-bottom: 8rem !important;
}

/* Also target the chat float bottom container specifically */
[data-testid="stChatFloatingInputContainer"] {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* ---- HERO ---- */
.hero {
    background: linear-gradient(135deg, rgba(30,27,75,0.45) 0%, rgba(15,23,42,0.22) 100%);
    border: 1px solid var(--border-accent);
    border-radius: var(--r-lg);
    padding: 30px 38px 24px;
    margin-bottom: 18px;
    backdrop-filter: blur(20px);
    box-shadow: var(--glow);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -70%;
    right: -12%;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(168,85,247,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -50%;
    left: -8%;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.09) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.14);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 3px 13px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1.05;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 45%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.hero-sub {
    color: var(--text2);
    font-size: 0.95rem;
    line-height: 1.55;
    max-width: 640px;
    margin-bottom: 6px;
}
.hero-author {
    font-size: 0.72rem;
    color: var(--text3);
}
.hero-author a {
    color: var(--accent);
    text-decoration: none;
    font-weight: 600;
}
.hero-author a:hover { color: var(--accent2); }

/* ---- CONFIG PANEL ---- */
.cfg-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 18px 22px 14px;
    margin-bottom: 14px;
    backdrop-filter: blur(12px);
}
.cfg-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 10px;
}

/* ---- STAT CARDS ---- */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 14px 18px;
    backdrop-filter: blur(12px);
    transition: all 0.22s ease;
    text-align: center;
}
.stat-card:hover {
    border-color: var(--border-accent);
    transform: translateY(-2px);
    box-shadow: 0 4px 24px rgba(99,102,241,0.12);
}
.s-lbl {
    font-size: 0.62rem;
    font-weight: 700;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 3px;
}
.s-val { font-size: 1.3rem; font-weight: 800; color: var(--text); }
.s-val.v1 { color: var(--accent); }
.s-val.v2 { color: var(--cyan); }
.s-val.v3 { color: var(--emerald); }

/* ---- DOCUMENT CHIPS ---- */
.doc-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 8px;
    padding: 6px 14px;
    margin: 3px 4px 3px 0;
    font-size: 0.78rem;
    color: #c7d2fe;
    font-weight: 500;
}
.doc-chip .dc-pages {
    color: var(--text3);
    font-size: 0.68rem;
}

/* ---- SOURCE CARDS ---- */
.src-card {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: var(--r-sm);
    padding: 12px 15px;
    margin: 7px 0;
}
.src-hdr {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-weight: 600;
    color: #c7d2fe;
    font-size: 0.82rem;
    margin-bottom: 6px;
}
.src-badge {
    background: rgba(99,102,241,0.22);
    color: #e0e7ff;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 700;
}
.src-snippet {
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.5;
    background: rgba(0,0,0,0.28);
    padding: 8px 12px;
    border-radius: 6px;
    border-left: 3px solid #6366f1;
}

/* ---- META ROW ---- */
.meta-row {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-top: 8px;
    font-size: 0.7rem;
    color: var(--text3);
}
.meta-row .ml { color: var(--cyan); font-weight: 600; }
.meta-row .mm { color: var(--accent); }
.meta-row .mc { color: var(--emerald); }

/* ---- DIVIDER ---- */
.thin-hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 12px 0;
}

/* ---- BUTTONS ---- */
.stButton > button {
    border-radius: var(--r-sm) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    font-size: 0.82rem !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 18px rgba(99,102,241,0.25) !important;
}

/* Chat avatars */
[data-testid="stChatMessage"] {
    margin-bottom: 2px;
}

/* File uploader styling */
[data-testid="stFileUploader"] {
    border-radius: var(--r-sm);
}
[data-testid="stFileUploader"] section {
    padding: 8px !important;
}

/* Selectbox / slider smaller */
[data-testid="stSelectbox"], [data-testid="stSlider"] {
    font-size: 0.85rem;
}

/* Expander tweak */
[data-testid="stExpander"] {
    border-color: rgba(99,102,241,0.15) !important;
    border-radius: var(--r-sm) !important;
}
</style>"""


def _inject_css():
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# RAG Engine (cached singleton)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading DocuMind-RAG engine and embedding model...")
def _get_rag_engine():
    from backend.rag_engine import RAGEngine
    return RAGEngine()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    for k, v in {
        "session_id": uuid.uuid4().hex[:12],
        "messages": [],
        "indexed_docs": [],
        "last_upload_msg": None,
        "show_config": True,
    }.items():
        st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# Ingest helper
# ---------------------------------------------------------------------------
def _ingest_files(uploaded_files):
    rag = _get_rag_engine()
    total_chunks = 0
    names = []
    try:
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.getvalue())
                tmp_path = tmp.name
            try:
                info = rag.ingest_pdf(
                    file_path=tmp_path,
                    document_name=uf.name,
                    session_id=st.session_state["session_id"],
                )
                total_chunks += info["total_chunks"]
                names.append(uf.name)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return True, f"Indexed {len(names)} file(s) with {total_chunks} chunks", total_chunks
    except Exception as exc:
        return False, f"Ingestion error: {exc}", 0


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------
def _run_query(question, top_k, model, temperature, search_mode):
    rag = _get_rag_engine()
    try:
        return rag.answer_query(
            question=question,
            session_id=st.session_state["session_id"],
            top_k=top_k,
            model_name=model,
            temperature=temperature,
            search_mode=search_mode,
        )
    except Exception as exc:
        return {
            "question": question,
            "answer": f"**Error during generation:**\n\n```\n{exc}\n```",
            "sources": [],
            "model_used": "error",
            "latency_ms": 0.0,
            "chunks_retrieved": 0,
        }


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def _render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources & Page Citations ({len(sources)})", expanded=False):
        for src in sources:
            sv = src.get("score")
            ss = f"Score: {sv:.4f}" if sv is not None else ""
            st.markdown(
                f'<div class="src-card">'
                f'<div class="src-hdr"><span>Document: <b>{src.get("document_name","Doc")}</b> '
                f'- Page {src.get("page_number",1)}</span>'
                f'<span class="src-badge">{ss}</span></div>'
                f'<div class="src-snippet">{src.get("snippet","")}</div></div>',
                unsafe_allow_html=True,
            )


def _render_meta(lat, mod, ch):
    st.markdown(
        f'<div class="meta-row">'
        f'<span class="ml">Latency: {lat:.0f} ms</span>'
        f'<span class="mm">Model: {mod}</span>'
        f'<span class="mc">Passages: {ch}</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def render_app():
    _inject_css()
    _init_state()
    rag = _get_rag_engine()

    # Refresh docs list
    docs = rag.list_documents(session_id=st.session_state["session_id"])
    st.session_state["indexed_docs"] = docs
    docs_count = len(docs)
    total_chunks = sum(d.get("total_chunks", 0) for d in docs)
    total_pages = sum(d.get("total_pages", 0) for d in docs)

    # ====================================================================
    # HERO BANNER
    # ====================================================================
    st.markdown(
        '<div class="hero">'
        '<div class="hero-badge">LangChain &middot; FAISS &middot; BM25 &middot; '
        'Groq Cloud &middot; Hybrid RRF</div>'
        '<h1>DocuMind-RAG</h1>'
        '<div class="hero-sub">Enterprise-grade multi-document AI assistant with '
        'hybrid semantic + keyword retrieval, surgical page citations, and '
        'zero-hallucination guardrails.</div>'
        '<div class="hero-author">Built by '
        '<a href="https://github.com/DRohithSubrahmanyaSai" target="_blank">'
        'D Rohith Subrahmanya Sai</a> &nbsp;&middot;&nbsp; '
        '<a href="https://github.com/DRohithSubrahmanyaSai/DocuMind-RAG" '
        'target="_blank">GitHub Repository</a></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ====================================================================
    # CONFIGURATION PANEL (collapsible)
    # ====================================================================
    with st.expander("Configuration & Document Upload", expanded=st.session_state.get("show_config", True)):

        # Row 1: Upload + Model + Temperature
        up_col, mod_col, temp_col = st.columns([2, 1.2, 0.8])

        with up_col:
            uploaded_files = st.file_uploader(
                "Upload PDF Documents",
                type=["pdf"],
                accept_multiple_files=True,
                key="file_uploader",
                label_visibility="collapsed",
                help="Drag & drop PDF files here",
            )

        with mod_col:
            model_choice = st.selectbox(
                "LLM Model",
                options=[
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "qwen/qwen3.8-27b",
                    "gemma2-9b-it",
                ],
                index=0,
                key="model_sel",
            )

        with temp_col:
            temperature = st.slider(
                "Temp",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                key="temp_sl",
            )

        # Row 2: Search mode + Top-K + Process button
        sm_col, tk_col, btn_col = st.columns([1.5, 1, 1.5])

        with sm_col:
            search_mode = st.selectbox(
                "Retrieval Mode",
                options=["hybrid", "dense", "sparse"],
                index=0,
                format_func=lambda x: {
                    "hybrid": "Hybrid (FAISS + BM25 + RRF)",
                    "dense": "Dense Vectors (FAISS only)",
                    "sparse": "Keyword Lexical (BM25 only)",
                }[x],
                key="search_mode_sel",
            )

        with tk_col:
            top_k = st.slider("Top-K Passages", 1, 10, 4, key="topk_sl")

        with btn_col:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if uploaded_files:
                if st.button(
                    "Process & Index PDFs",
                    use_container_width=True,
                    type="primary",
                    key="btn_upload",
                ):
                    with st.spinner("Extracting text, chunking & computing embeddings..."):
                        ok, msg, nc = _ingest_files(uploaded_files)
                    st.session_state["last_upload_msg"] = ("ok" if ok else "err", msg)
                    docs = rag.list_documents(session_id=st.session_state["session_id"])
                    st.session_state["indexed_docs"] = docs
                    st.rerun()
            else:
                st.button("Upload PDFs first", use_container_width=True, disabled=True, key="btn_upload_disabled")

        # Upload result
        if st.session_state.get("last_upload_msg"):
            kind, msg = st.session_state["last_upload_msg"]
            if kind == "ok":
                st.success(f"{msg}")
            else:
                st.error(f"{msg}")

    # ====================================================================
    # STATS RIBBON
    # ====================================================================
    mode_labels = {"hybrid": "Hybrid RRF", "dense": "FAISS Dense", "sparse": "BM25 Sparse"}
    groq_ok = bool(os.getenv("GROQ_API_KEY", ""))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="s-lbl">Documents</div>'
            f'<div class="s-val">{docs_count}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card"><div class="s-lbl">Chunks</div>'
            f'<div class="s-val v1">{total_chunks}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card"><div class="s-lbl">Pages</div>'
            f'<div class="s-val v2">{total_pages}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="stat-card"><div class="s-lbl">Retrieval</div>'
            f'<div class="s-val v3" style="font-size:0.95rem">'
            f'{mode_labels.get(search_mode, "Hybrid")}</div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        g_color = "#34d399" if groq_ok else "#f87171"
        g_text = "Active" if groq_ok else "No Key"
        st.markdown(
            f'<div class="stat-card"><div class="s-lbl">Groq API</div>'
            f'<div class="s-val" style="font-size:0.95rem;color:{g_color}">'
            f'{g_text}</div></div>',
            unsafe_allow_html=True,
        )

    # ====================================================================
    # INDEXED DOCUMENTS ROW
    # ====================================================================
    if docs:
        doc_chips = ""
        for d in docs:
            doc_chips += (
                f'<span class="doc-chip">{d.get("document_name","?")} '
                f'<span class="dc-pages">{d.get("total_pages",1)}p / '
                f'{d.get("total_chunks",0)}ch</span></span>'
            )

        act_col, chip_col = st.columns([1, 5])
        with act_col:
            if st.button("Clear Knowledge Base", key="btn_clear_kb", use_container_width=True):
                rag.clear_all()
                st.session_state["messages"] = []
                st.session_state["last_upload_msg"] = None
                st.rerun()
        with chip_col:
            st.markdown(doc_chips, unsafe_allow_html=True)

    st.markdown('<hr class="thin-hr">', unsafe_allow_html=True)

    # ====================================================================
    # CHAT ACTIONS BAR
    # ====================================================================
    if st.session_state.get("messages"):
        act1, act2, spacer = st.columns([1, 1, 4])
        with act1:
            if st.button("Clear Chat", key="btn_clear_chat", use_container_width=True):
                st.session_state["messages"] = []
                st.rerun()
        with act2:
            chat_md = "# DocuMind-RAG Conversation\n\n"
            for m in st.session_state["messages"]:
                chat_md += f"### {m['role'].upper()}:\n{m['content']}\n\n"
            st.download_button(
                "Export Chat",
                data=chat_md,
                file_name=f"documind_{st.session_state['session_id']}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # ====================================================================
    # CHAT HISTORY
    # ====================================================================
    messages = st.session_state.get("messages", [])
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_sources(msg.get("sources", []))
                if msg.get("latency_ms"):
                    _render_meta(
                        msg["latency_ms"],
                        msg.get("model_used", model_choice),
                        msg.get("chunks_retrieved", 0),
                    )

    # ====================================================================
    # PROMPT CHIPS (empty state)
    # ====================================================================
    if not messages and docs_count > 0:
        st.markdown(
            '<p style="color:#64748b;font-size:0.85rem;margin:4px 0">Suggested questions:</p>',
            unsafe_allow_html=True,
        )
        ch1, ch2, ch3 = st.columns(3)
        pending = None
        with ch1:
            if st.button("Summarize key takeaways", use_container_width=True, key="chip1"):
                pending = "Summarize the key takeaways and core objectives described in these documents."
        with ch2:
            if st.button("Extract statistics & figures", use_container_width=True, key="chip2"):
                pending = "Extract all key numerical metrics, statistics, and findings mentioned in the documents."
        with ch3:
            if st.button("Explain the methodology", use_container_width=True, key="chip3"):
                pending = "Explain the exact methodology and architectural approach used in these documents."
        if pending:
            st.session_state["pending_prompt"] = pending
            st.rerun()

    # Empty knowledge base state
    if docs_count == 0 and not messages:
        st.markdown(
            '<div style="text-align:center;padding:60px 20px;color:#64748b">'
            '<div style="font-size:3rem;margin-bottom:12px">&#128196;</div>'
            '<div style="font-size:1.1rem;font-weight:600;color:#94a3b8;margin-bottom:6px">'
            'No documents indexed yet</div>'
            '<div style="font-size:0.88rem">Upload PDF files using the Configuration panel above, '
            'then click <b>Process & Index PDFs</b> to build the knowledge base.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ====================================================================
    # CHAT INPUT
    # ====================================================================
    user_input = st.chat_input(
        "Ask anything about your uploaded documents...", key="chat_input"
    )
    prompt = user_input or st.session_state.pop("pending_prompt", None)

    if prompt:
        if docs_count == 0:
            st.warning("Please upload and index at least one PDF document first.")
            st.stop()

        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving passages & generating answer..."):
                result = _run_query(
                    question=prompt,
                    top_k=top_k,
                    model=model_choice,
                    temperature=temperature,
                    search_mode=search_mode,
                )

            answer = result.get("answer", "No response received.")
            sources = result.get("sources", [])
            latency = result.get("latency_ms", 0.0)
            model_used = result.get("model_used", model_choice)
            chunks_n = result.get("chunks_retrieved", len(sources))

            st.markdown(answer)
            _render_sources(sources)
            _render_meta(latency, model_used, chunks_n)

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "latency_ms": latency,
            "model_used": model_used,
            "chunks_retrieved": chunks_n,
        })
        st.rerun()


if __name__ == "__main__":
    render_app()


import os
import sys
import uuid
import json
import time
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Suppress torchvision and tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Page Configuration
st.set_page_config(
    page_title="DocuMind-RAG | Precision Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS Styles
CSS_PATH = Path(__file__).parent / "styles.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_url" not in st.session_state:
    st.session_state.backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000")

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []


# ==============================================================================
# Backend Communication Layer (REST API)
# ==============================================================================
def check_api_health(url: str) -> Optional[Dict[str, Any]]:
    """Verify connectivity with FastAPI backend."""
    try:
        resp = requests.get(f"{url.rstrip('/')}/api/health", timeout=2.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def fetch_documents(url: str, session_id: str) -> List[Dict[str, Any]]:
    """Retrieve document list from backend API."""
    try:
        resp = requests.get(
            f"{url.rstrip('/')}/api/documents",
            params={"session_id": session_id},
            timeout=3.0,
        )
        if resp.status_code == 200:
            return resp.json().get("documents", [])
    except Exception:
        pass
    return []


def upload_files_to_rag(
    url: str, files, session_id: str
) -> Tuple[bool, str, int]:
    """Upload PDF files to FastAPI backend."""
    try:
        files_payload = [
            ("files", (f.name, f.getvalue(), "application/pdf")) for f in files
        ]
        resp = requests.post(
            f"{url.rstrip('/')}/api/upload",
            files=files_payload,
            data={"session_id": session_id},
            timeout=60.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return True, data.get("message", "Files uploaded successfully."), data.get("total_chunks_added", 0)
        else:
            return False, f"Upload error: {resp.text}", 0
    except Exception as e:
        return False, f"Connection error: {str(e)}", 0


def execute_rag_query(
    url: str,
    question: str,
    session_id: str,
    top_k: int,
    model: str,
    temperature: float,
    search_mode: str,
) -> Dict[str, Any]:
    """Execute query via FastAPI backend."""
    try:
        payload = {
            "question": question,
            "session_id": session_id,
            "top_k": top_k,
            "model": model,
            "temperature": temperature,
            "search_mode": search_mode,
        }
        resp = requests.post(
            f"{url.rstrip('/')}/api/chat",
            json=payload,
            timeout=45.0,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {
                "question": question,
                "answer": f"API Error ({resp.status_code}): {resp.text}",
                "sources": [],
                "model_used": "error",
                "latency_ms": 0.0,
                "chunks_retrieved": 0,
            }
    except Exception as e:
        return {
            "question": question,
            "answer": (
                "⚠️ Backend API unreachable. Please ensure the FastAPI backend is running:\n\n"
                "`python main.py server`\n\n"
                f"Error: {str(e)}"
            ),
            "sources": [],
            "model_used": "offline",
            "latency_ms": 0.0,
            "chunks_retrieved": 0,
        }


def delete_single_document(url: str, document_name: str) -> bool:
    """Delete a document from index."""
    try:
        resp = requests.delete(f"{url.rstrip('/')}/api/documents/{document_name}", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def clear_all_knowledge(url: str) -> bool:
    """Clear all documents and vectors."""
    try:
        resp = requests.delete(f"{url.rstrip('/')}/api/documents", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


# ==============================================================================
# UI Sidebar: Controls & Document Management
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚡ **DocuMind-RAG Engine**")
    st.caption("Engineered by **D Rohith Subrahmanya Sai**")

    # Mode and Health Check
    health_data = check_api_health(st.session_state.backend_url)
    if health_data:
        st.success("🟢 FastAPI Backend: Connected")
    else:
        st.warning("🟠 Backend: Disconnected (Start with: python main.py server)")

    # API Settings Accordion
    with st.expander("⚙️ Backend Service URL", expanded=False):
        api_url_input = st.text_input(
            "FastAPI Backend Endpoint",
            value=st.session_state.backend_url,
            help="URL of your running FastAPI service (Local: http://localhost:8000 or Cloud URL)",
        )
        if api_url_input != st.session_state.backend_url:
            st.session_state.backend_url = api_url_input
            st.rerun()

    st.markdown("---")

    # Document Uploader
    st.markdown("#### 📄 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF files to index into FAISS and BM25",
    )

    if uploaded_files:
        if st.button("🚀 Process & Index PDFs", use_container_width=True, type="primary"):
            with st.spinner("Extracting text, chunking & computing embeddings..."):
                ok, msg, chunks_count = upload_files_to_rag(
                    st.session_state.backend_url,
                    uploaded_files,
                    st.session_state.session_id,
                )
                if ok:
                    st.success(f"✅ Ingested successfully (+{chunks_count} chunks)")
                    st.session_state.indexed_docs = fetch_documents(
                        st.session_state.backend_url, st.session_state.session_id
                    )
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

    # Active Documents List
    st.session_state.indexed_docs = fetch_documents(
        st.session_state.backend_url, st.session_state.session_id
    )
    docs = st.session_state.indexed_docs

    st.markdown(f"#### 📚 Knowledge Base ({len(docs)} files)")
    if docs:
        for d in docs:
            col_doc, col_del = st.columns([4, 1])
            with col_doc:
                st.markdown(
                    f"**{d['document_name']}**  \n"
                    f"<small style='color:#9ca3af;'>{d.get('total_pages', 1)} pages • {d.get('total_chunks', 0)} chunks</small>",
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑️", key=f"del_{d['document_name']}", help="Delete document"):
                    delete_single_document(st.session_state.backend_url, d["document_name"])
                    st.rerun()

        if st.button("🧹 Clear Entire Knowledge Base", use_container_width=True):
            clear_all_knowledge(st.session_state.backend_url)
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("No documents indexed yet. Upload a PDF above.")

    st.markdown("---")

    # RAG Retrieval Parameters
    st.markdown("#### 🎛️ Retrieval Parameters")

    search_mode = st.selectbox(
        "Retrieval Algorithm",
        options=["hybrid", "dense", "sparse"],
        index=0,
        format_func=lambda x: {
            "hybrid": "⚡ Hybrid RAG (FAISS + BM25 + RRF)",
            "dense": "🧬 Dense Vectors Only (FAISS)",
            "sparse": "🔍 Keyword Lexical Only (BM25)",
        }[x],
        help="Hybrid RRF yields highest accuracy across semantic meaning and specific terms.",
    )

    top_k = st.slider("Retrieved Passages (Top-K)", min_value=1, max_value=10, value=4)

    model_choice = st.selectbox(
        "Groq LLM Model",
        options=[
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen/qwen3.8-27b",
        ],
        index=0,
    )

    temperature = st.slider(
        "Temperature (Strict vs Creative)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
    )

    st.markdown("---")

    # Conversation Actions
    if st.button("🔄 Clear Chat Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        chat_export_md = "# DocuMind-RAG Conversation Log\n\n"
        for m in st.session_state.messages:
            chat_export_md += f"### {m['role'].upper()}:\n{m['content']}\n\n"
        st.download_button(
            "📥 Export Chat (Markdown)",
            data=chat_export_md,
            file_name=f"documind_chat_{st.session_state.session_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ==============================================================================
# Main Content Area
# ==============================================================================

# Hero Header
st.markdown(
    """
    <div class="hero-container">
        <div class="badge-pill">⚡ Powered by LangChain • FAISS • BM25 • Groq Cloud</div>
        <div class="hero-title">DocuMind-RAG</div>
        <div class="hero-tagline">
            Next-Gen Multi-Document Hybrid RAG Engine with surgical page citations,
            zero hallucination guardrails, and ultra-low latency.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Statistics Ribbon
docs_count = len(st.session_state.indexed_docs)
total_chunks = sum(d.get("total_chunks", 0) for d in st.session_state.indexed_docs)
last_latency = (
    st.session_state.messages[-1].get("latency_ms", 0.0)
    if st.session_state.messages
    else 0.0
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-label">Indexed Documents</div>
            <div class="stat-value">{docs_count}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-label">Total Knowledge Chunks</div>
            <div class="stat-value">{total_chunks}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-label">Groq Model</div>
            <div class="stat-value" style="font-size: 0.95rem;">{model_choice.split('-')[0].upper()} 70B</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-label">Retrieval Mode</div>
            <div class="stat-value" style="font-size: 0.95rem; color: #a5b4fc;">{search_mode.upper()} RRF</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# Render Past Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "⚡"):
        st.markdown(msg["content"])

        # Display Source Citations if present
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📑 Verified Sources & Page Citations ({len(sources)})", expanded=False):
                for idx, src in enumerate(sources, 1):
                    score_str = f"Score: {src['score']:.4f}" if src.get("score") is not None else ""
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-header">
                                <span>📄 <b>{src['document_name']}</b> — Page {src['page_number']}</span>
                                <span class="source-badge">{score_str}</span>
                            </div>
                            <div class="source-snippet">{src['snippet']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Meta footer (latency & model)
        if msg["role"] == "assistant" and msg.get("latency_ms"):
            st.markdown(
                f"""<div class="meta-footer">
                    <span class="latency-badge">⚡ Latency: {msg['latency_ms']} ms</span>
                    <span>🤖 Model: {msg.get('model_used', 'Groq')}</span>
                    <span>📑 Passages: {msg.get('chunks_retrieved', 0)}</span>
                </div>""",
                unsafe_allow_html=True,
            )


# Suggested Starter Prompt Chips (only if chat is empty)
if not st.session_state.messages and docs_count > 0:
    st.markdown("<p style='color:#9ca3af; font-size:0.9rem;'>Suggested questions:</p>", unsafe_allow_html=True)
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    suggested_q = None
    with chip_col1:
        if st.button("📌 Summarize the core takeaways", use_container_width=True):
            suggested_q = "Summarize the key takeaways and core objectives described in these documents."
    with chip_col2:
        if st.button("📊 Extract key statistics & figures", use_container_width=True):
            suggested_q = "Extract all key numerical metrics, statistics, and findings mentioned in the text."
    with chip_col3:
        if st.button("🔍 Explain the methodology used", use_container_width=True):
            suggested_q = "Explain the exact methodology and architectural approach used in these documents."

    if suggested_q:
        st.session_state.pending_prompt = suggested_q
        st.rerun()


# Chat Input Handling
user_input = st.chat_input("Ask any question across your uploaded documents...")
prompt_to_run = user_input or st.session_state.pop("pending_prompt", None)

if prompt_to_run:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt_to_run})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt_to_run)

    # Process Assistant Response
    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Retrieving hybrid passages & querying Groq LLM..."):
            result = execute_rag_query(
                url=st.session_state.backend_url,
                question=prompt_to_run,
                session_id=st.session_state.session_id,
                top_k=top_k,
                model=model_choice,
                temperature=temperature,
                search_mode=search_mode,
            )

        answer_text = result.get("answer", "No response received.")
        sources = result.get("sources", [])
        latency_ms = result.get("latency_ms", 0.0)
        model_used = result.get("model_used", model_choice)
        chunks_retrieved = result.get("chunks_retrieved", len(sources))

        st.markdown(answer_text)

        if sources:
            with st.expander(f"📑 Verified Sources & Page Citations ({len(sources)})", expanded=False):
                for idx, src in enumerate(sources, 1):
                    score_str = f"Score: {src['score']:.4f}" if src.get("score") is not None else ""
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-header">
                                <span>📄 <b>{src['document_name']}</b> — Page {src['page_number']}</span>
                                <span class="source-badge">{score_str}</span>
                            </div>
                            <div class="source-snippet">{src['snippet']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown(
            f"""<div class="meta-footer">
                <span class="latency-badge">⚡ Latency: {latency_ms} ms</span>
                <span>🤖 Model: {model_used}</span>
                <span>📑 Passages: {chunks_retrieved}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # Store in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "latency_ms": latency_ms,
            "model_used": model_used,
            "chunks_retrieved": chunks_retrieved,
        })

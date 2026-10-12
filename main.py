"""
DocuMind-RAG Master CLI & Application Controller
Author: D Rohith Subrahmanya Sai
GitHub: https://github.com/DRohithSubrahmanyaSai/DocuMind-RAG
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Fix Windows console UTF-8 emoji encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add workspace directory to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_backend(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Launch FastAPI backend server with Uvicorn."""
    import uvicorn
    from backend.config import settings

    print(f"\n🚀 Starting DocuMind-RAG FastAPI Backend on http://{host}:{port}")
    print(f"📖 Interactive Swagger Docs: http://{host}:{port}/docs\n")
    uvicorn.run("backend.main:app", host=host, port=port, reload=reload)


def run_frontend(port: int = 8501):
    """Launch Streamlit frontend interface."""
    print(f"\n⚡ Starting DocuMind-RAG Streamlit UI on http://localhost:{port}\n")
    subprocess.run(["streamlit", "run", "app.py", "--server.port", str(port)])


def cli_ingest(file_path: str):
    """CLI utility to ingest a PDF into the RAG vector store."""
    from backend.rag_engine import rag_engine

    path = Path(file_path)
    if not path.exists():
        print(f"❌ Error: File '{file_path}' does not exist.")
        return

    print(f"📄 Ingesting '{path.name}' into FAISS & BM25 indices...")
    try:
        res = rag_engine.ingest_pdf(str(path), path.name)
        print(f"✅ Successfully ingested: {res['total_pages']} pages, {res['total_chunks']} chunks.")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")


def cli_ask(question: str):
    """CLI utility to ask a question to DocuMind-RAG."""
    from backend.rag_engine import rag_engine

    print(f"\n🔍 Query: {question}")
    print("⏳ Processing through Hybrid RAG & Groq LLM...")
    try:
        res = rag_engine.answer_query(question)
        print("\n💡 Answer:")
        print(res["answer"])
        print(f"\n⚡ Model: {res['model_used']} | Latency: {res['latency_ms']} ms")
        if res["sources"]:
            print("\n📑 Sources:")
            for s in res["sources"]:
                print(f"  • {s['document_name']} (Page {s['page_number']})")
    except Exception as e:
        print(f"❌ Query failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="DocuMind-RAG: Next-Gen Multi-Document Hybrid RAG Engine by D Rohith Subrahmanya Sai"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: server
    server_parser = subparsers.add_parser("server", help="Start the FastAPI backend server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    server_parser.add_argument("--port", type=int, default=8000, help="Port number")
    server_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")

    # Command: ui
    ui_parser = subparsers.add_parser("ui", help="Start the Streamlit Chat UI")
    ui_parser.add_argument("--port", type=int, default=8501, help="Port number")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF document into FAISS")
    ingest_parser.add_argument("file", help="Path to PDF file")

    # Command: ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question against indexed documents")
    ask_parser.add_argument("question", help="Question text")

    args = parser.parse_args()

    if args.command == "server":
        run_backend(host=args.host, port=args.port, reload=not args.no_reload)
    elif args.command == "ui":
        run_frontend(port=args.port)
    elif args.command == "ingest":
        cli_ingest(args.file)
    elif args.command == "ask":
        cli_ask(args.question)
    else:
        # Default action: launch UI
        print("💡 No command specified. Defaulting to launching Streamlit Chat UI.")
        run_frontend()


if __name__ == "__main__":
    main()

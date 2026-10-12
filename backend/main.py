import os
import shutil
import tempfile
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.rag_engine import rag_engine
from backend.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    DocumentInfo,
    HealthResponse,
    SourceCitation,
)

logger = logging.getLogger("documind.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title=f"{settings.PROJECT_NAME} API",
    version=settings.PROJECT_VERSION,
    description=(
        "Next-Gen Multi-Document Hybrid RAG Engine powered by "
        "FastAPI, LangChain, FAISS, BM25, and Groq Cloud LLMs.\n\n"
        f"**Author:** {settings.AUTHOR}\n"
        f"**Repository:** [{settings.REPO_URL}]({settings.REPO_URL})"
    ),
)

# Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "author": settings.AUTHOR,
        "version": settings.PROJECT_VERSION,
        "repository": settings.REPO_URL,
        "docs_url": "/docs",
        "health_check": "/api/health",
        "description": "High-accuracy Hybrid RAG with LangChain, FAISS, and Groq",
    }


@app.get("/api/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Returns the operational status and index metrics."""
    docs = rag_engine.list_documents()
    total_chunks = len(rag_engine.bm25_corpus)
    return HealthResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        author=settings.AUTHOR,
        version=settings.PROJECT_VERSION,
        groq_configured=bool(settings.GROQ_API_KEY),
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        indexed_documents_count=len(docs),
        indexed_chunks_count=total_chunks,
    )


@app.get("/api/models", tags=["RAG Configuration"])
async def get_available_models():
    """Lists supported Groq models."""
    return {
        "current_default": settings.DEFAULT_GROQ_MODEL,
        "available_models": [
            {
                "id": "llama-3.3-70b-versatile",
                "name": "Llama 3.3 70B Versatile",
                "speed": "Ultra Fast",
                "quality": "Maximum Reasoning",
            },
            {
                "id": "llama-3.1-8b-instant",
                "name": "Llama 3.1 8B Instant",
                "speed": "Near-Instant",
                "quality": "Great for concise queries",
            },
            {
                "id": "mixtral-8x7b-32768",
                "name": "Mixtral 8x7B (32k Context)",
                "speed": "Fast",
                "quality": "Extended context window",
            },
            {
                "id": "gemma2-9b-it",
                "name": "Gemma 2 9B IT",
                "speed": "Fast",
                "quality": "High precision",
            },
        ],
    }


@app.post("/api/upload", response_model=UploadResponse, tags=["Document Ingestion"])
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Upload one or multiple PDF documents.
    Extracts text, computes FAISS dense embeddings, builds BM25 index.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided.",
        )

    processed_docs: List[DocumentInfo] = []
    total_chunks = 0

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file '{file.filename}'. Only PDF files are supported.",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            info = rag_engine.ingest_pdf(
                file_path=tmp_path,
                document_name=file.filename,
                session_id=session_id,
            )
            processed_docs.append(
                DocumentInfo(
                    document_name=info["document_name"],
                    total_pages=info["total_pages"],
                    total_chunks=info["total_chunks"],
                    session_id=info["session_id"],
                )
            )
            total_chunks += info["total_chunks"]
        except Exception as e:
            logger.error(f"Error indexing {file.filename}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process {file.filename}: {str(e)}",
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return UploadResponse(
        status="success",
        message=f"Successfully ingested {len(processed_docs)} document(s).",
        documents=processed_docs,
        total_chunks_added=total_chunks,
    )


@app.post("/api/query", response_model=QueryResponse, tags=["RAG Chat & Q&A"])
@app.post("/api/chat", response_model=QueryResponse, tags=["RAG Chat & Q&A"])
async def query_documents(request: QueryRequest):
    """
    Ask a question against indexed documents using Hybrid RAG + Groq LLM.
    Returns generated answer with surgical page citations and RRF scores.
    """
    try:
        result = rag_engine.answer_query(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
            model_name=request.model,
            temperature=request.temperature,
            search_mode=request.search_mode,
        )

        sources = [
            SourceCitation(
                document_name=s["document_name"],
                page_number=s["page_number"],
                chunk_index=s["chunk_index"],
                score=s.get("score"),
                snippet=s["snippet"],
            )
            for s in result["sources"]
        ]

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            model_used=result["model_used"],
            latency_ms=result["latency_ms"],
            chunks_retrieved=result["chunks_retrieved"],
            session_id=result.get("session_id"),
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


@app.get("/api/documents", tags=["Document Ingestion"])
async def get_documents(session_id: Optional[str] = None):
    """List all currently indexed documents in FAISS and BM25."""
    docs = rag_engine.list_documents(session_id=session_id)
    return {
        "count": len(docs),
        "documents": docs,
    }


@app.delete("/api/documents/{document_name}", tags=["Document Ingestion"])
async def delete_document(document_name: str):
    """Delete a document and purge its vectors from FAISS."""
    success = rag_engine.delete_document(document_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_name}' not found.",
        )
    return {"status": "success", "message": f"Document '{document_name}' removed."}


@app.delete("/api/documents", tags=["Document Ingestion"])
async def clear_all_documents():
    """Clear all documents, vectors, and BM25 index."""
    rag_engine.clear_all()
    return {"status": "success", "message": "All documents and vector indices cleared."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)

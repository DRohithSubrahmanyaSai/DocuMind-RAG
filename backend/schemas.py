from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    document_name: str = Field(..., description="Original filename of the document")
    page_number: int = Field(..., description="1-indexed page number where snippet occurs")
    chunk_index: int = Field(..., description="Sequential chunk index")
    score: Optional[float] = Field(None, description="Relevance or RRF score")
    snippet: str = Field(..., description="Excerpt of the retrieved passage")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer based on documents")
    session_id: Optional[str] = Field(None, description="Optional session or user identifier")
    top_k: int = Field(4, ge=1, le=15, description="Number of passages to retrieve")
    model: Optional[str] = Field(None, description="Groq model name to use for generation")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Sampling temperature")
    search_mode: Literal["hybrid", "dense", "sparse"] = Field(
        "hybrid", description="Retrieval algorithm: hybrid (FAISS+BM25), dense (FAISS), sparse (BM25)"
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceCitation]
    model_used: str
    latency_ms: float
    chunks_retrieved: int
    session_id: Optional[str] = None


class DocumentInfo(BaseModel):
    document_name: str
    total_pages: int
    total_chunks: int
    session_id: Optional[str] = None


class UploadResponse(BaseModel):
    status: str
    message: str
    documents: List[DocumentInfo]
    total_chunks_added: int


class HealthResponse(BaseModel):
    status: str
    project: str
    author: str
    version: str
    groq_configured: bool
    embedding_model: str
    indexed_documents_count: int
    indexed_chunks_count: int

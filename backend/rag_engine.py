import os
import re
import time
import warnings
import logging
from typing import List, Dict, Any, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import pymupdf  # Modern PyMuPDF import
from rank_bm25 import BM25Okapi

# LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import settings

logger = logging.getLogger("documind.rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RAGEngine:
    """
    Enterprise-Grade Hybrid RAG Engine
    Combines FAISS dense vector search with BM25Okapi lexical retrieval
    using Reciprocal Rank Fusion (RRF), powered by LangChain and Groq LLMs.
    """

    def __init__(self):
        self.embeddings = None
        self.vector_store: Optional[FAISS] = None
        self.bm25_corpus: List[Dict[str, Any]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents_metadata: Dict[str, Dict[str, Any]] = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._init_embeddings()
        self._load_existing_store()

    def _init_embeddings(self):
        """Initialize local HuggingFace embedding model."""
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise e

    def _load_existing_store(self):
        """Load persisted FAISS index if available on disk."""
        index_file = settings.FAISS_INDEX_PATH / "index.faiss"
        if index_file.exists():
            try:
                logger.info(f"Loading existing FAISS index from {settings.FAISS_INDEX_PATH}")
                self.vector_store = FAISS.load_local(
                    folder_path=str(settings.FAISS_INDEX_PATH),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                # Reconstruct BM25 and metadata from stored documents
                self._rebuild_bm25_from_vectorstore()
                logger.info(f"Loaded existing index with {len(self.bm25_corpus)} chunks.")
            except Exception as e:
                logger.warning(f"Could not load existing FAISS index: {e}. Starting fresh.")
                self.vector_store = None
                self.bm25_corpus = []
                self.bm25_index = None

    def _save_store(self):
        """Persist FAISS index to disk."""
        if self.vector_store is not None:
            try:
                settings.FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(folder_path=str(settings.FAISS_INDEX_PATH))
                logger.info(f"FAISS index saved to {settings.FAISS_INDEX_PATH}")
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Normalize text into lowercase word tokens for BM25."""
        return re.findall(r"\b\w+\b", text.lower())

    def _rebuild_bm25_from_vectorstore(self):
        """Synchronize BM25 corpus from vectorstore documents."""
        if self.vector_store is None:
            return

        self.bm25_corpus = []
        self.documents_metadata = {}

        # Extract all stored docstore documents
        docstore = self.vector_store.docstore
        # FAISS docstore in LangChain stores documents in _dict
        if hasattr(docstore, "_dict"):
            for doc_id, doc in docstore._dict.items():
                meta = doc.metadata
                chunk_entry = {
                    "doc_id": doc_id,
                    "text": doc.page_content,
                    "document_name": meta.get("document_name", "unknown"),
                    "page_number": meta.get("page_number", 1),
                    "chunk_index": meta.get("chunk_index", 0),
                    "session_id": meta.get("session_id"),
                }
                self.bm25_corpus.append(chunk_entry)

                doc_name = chunk_entry["document_name"]
                if doc_name not in self.documents_metadata:
                    self.documents_metadata[doc_name] = {
                        "document_name": doc_name,
                        "total_pages": meta.get("page_number", 1),
                        "total_chunks": 0,
                        "session_id": meta.get("session_id"),
                    }
                self.documents_metadata[doc_name]["total_chunks"] += 1
                self.documents_metadata[doc_name]["total_pages"] = max(
                    self.documents_metadata[doc_name]["total_pages"],
                    meta.get("page_number", 1),
                )

        if self.bm25_corpus:
            tokenized_corpus = [self._tokenize(item["text"]) for item in self.bm25_corpus]
            self.bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self.bm25_index = None

    def extract_pdf_pages(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text content per page using PyMuPDF."""
        pages = []
        doc = pymupdf.open(file_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text()
            # Normalize whitespace
            clean_text = " ".join(text.split())
            if clean_text.strip():
                pages.append((clean_text, page_idx + 1))
        doc.close()
        return pages

    def ingest_pdf(
        self, file_path: str, document_name: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a PDF: Extract pages -> Chunk text -> Create FAISS vectors & BM25 entries.
        """
        pages = self.extract_pdf_pages(file_path)
        if not pages:
            raise ValueError(f"No readable text found in {document_name}")

        documents_to_add: List[Document] = []
        chunk_counter = 0

        for page_text, page_number in pages:
            split_chunks = self.text_splitter.split_text(page_text)
            for chunk_text in split_chunks:
                metadata = {
                    "document_name": document_name,
                    "page_number": page_number,
                    "chunk_index": chunk_counter,
                    "session_id": session_id or "default",
                }
                documents_to_add.append(
                    Document(page_content=chunk_text, metadata=metadata)
                )
                chunk_counter += 1

        if not documents_to_add:
            raise ValueError("Document yielded 0 chunks.")

        # Update or create FAISS vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents_to_add, self.embeddings)
        else:
            self.vector_store.add_documents(documents_to_add)

        # Update BM25 and metadata
        self._rebuild_bm25_from_vectorstore()
        self._save_store()

        total_pages = max(p[1] for p in pages) if pages else 1
        self.documents_metadata[document_name] = {
            "document_name": document_name,
            "total_pages": total_pages,
            "total_chunks": chunk_counter,
            "session_id": session_id,
        }

        return {
            "document_name": document_name,
            "total_pages": total_pages,
            "total_chunks": chunk_counter,
            "session_id": session_id,
        }

    def _dense_search(
        self, query: str, top_k: int, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Dense semantic search using FAISS vector store."""
        if self.vector_store is None:
            return []

        search_k = max(top_k * 4, 15)
        # Similarity search with score (returns doc and distance)
        docs_and_scores = self.vector_store.similarity_search_with_score(
            query, k=search_k
        )

        results = []
        for doc, score in docs_and_scores:
            meta = doc.metadata
            if session_id and meta.get("session_id") not in [session_id, "default"]:
                continue
            # Convert FAISS L2 distance to similarity score
            sim_score = 1.0 / (1.0 + float(score))
            results.append({
                "text": doc.page_content,
                "document_name": meta.get("document_name", "unknown"),
                "page_number": meta.get("page_number", 1),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(sim_score, 4),
            })
            if len(results) >= top_k:
                break
        return results

    def _sparse_search(
        self, query: str, top_k: int, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Sparse lexical keyword search using BM25."""
        if self.bm25_index is None or not self.bm25_corpus:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25_index.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices:
            score = scores[idx]
            if score <= 0:
                continue
            item = self.bm25_corpus[idx]
            if session_id and item.get("session_id") not in [session_id, "default"]:
                continue
            results.append({
                "text": item["text"],
                "document_name": item["document_name"],
                "page_number": item["page_number"],
                "chunk_index": item["chunk_index"],
                "score": float(score),
            })
            if len(results) >= top_k:
                break
        return results

    def _reciprocal_rank_fusion(
        self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) for combining vector and keyword ranks.
        Score = sum(1 / (k + rank_i))
        """
        rrf_k = settings.RRF_K
        scores: Dict[Tuple[str, int, int], float] = {}
        chunk_map: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

        # Dense ranks
        for rank, item in enumerate(dense_results, start=1):
            key = (item["document_name"], item["page_number"], item["chunk_index"])
            scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))
            chunk_map[key] = item

        # Sparse ranks
        for rank, item in enumerate(sparse_results, start=1):
            key = (item["document_name"], item["page_number"], item["chunk_index"])
            scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))
            if key not in chunk_map:
                chunk_map[key] = item

        # Sort by fused score
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]

        fused = []
        for key in sorted_keys:
            chunk = dict(chunk_map[key])
            chunk["rrf_score"] = round(scores[key], 5)
            fused.append(chunk)

        return fused

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        session_id: Optional[str] = None,
        search_mode: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document passages based on mode."""
        if not self.bm25_corpus:
            return []

        if search_mode == "dense":
            return self._dense_search(query, top_k, session_id)
        elif search_mode == "sparse":
            return self._sparse_search(query, top_k, session_id)
        else:
            # Hybrid search
            candidate_k = max(top_k * 3, 10)
            dense_res = self._dense_search(query, candidate_k, session_id)
            sparse_res = self._sparse_search(query, candidate_k, session_id)
            return self._reciprocal_rank_fusion(dense_res, sparse_res, top_k)

    def answer_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: int = 4,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        search_mode: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        Execute full RAG pipeline: Retrieve -> Build Prompt -> Generate via Groq LLM.
        """
        start_time = time.time()
        retrieved_chunks = self.retrieve(
            query=question, top_k=top_k, session_id=session_id, search_mode=search_mode
        )

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "No indexed documents found. Please upload a PDF document first.",
                "sources": [],
                "model_used": "none",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "chunks_retrieved": 0,
                "session_id": session_id,
            }

        # Build context with clear source labels
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            context_blocks.append(
                f"[Source Passage {idx} | Document: {chunk['document_name']} | Page: {chunk['page_number']}]\n"
                f"{chunk['text']}"
            )
        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "You are DocuMind-RAG, an advanced AI document analysis assistant.\n"
            "Your task is to answer the user's question with surgical precision based STRICTLY on the provided Context.\n\n"
            "RULES:\n"
            "1. Answer ONLY using information present in the Context. Do NOT assume, extrapolate, or hallucinate.\n"
            "2. If the context does not contain enough information to answer the question, state exactly:\n"
            "   'I could not find the answer in the provided document(s).'\n"
            "3. For every important claim or figure, cite the source page number clearly, e.g., (Source: filename.pdf, Page 3).\n"
            "4. Be structured, concise, and professional."
        )

        user_prompt_template = (
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Helpful and Accurate Answer (with citations):"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt_template),
        ])

        selected_model = model_name or settings.DEFAULT_GROQ_MODEL
        groq_api_key = settings.GROQ_API_KEY

        if not groq_api_key:
            return {
                "question": question,
                "answer": (
                    "⚠️ Groq API key is not configured. Please set `GROQ_API_KEY` in your `.env` file or environment.\n\n"
                    "**Retrieved Context Snippets:**\n"
                    + "\n".join([f"- Page {c['page_number']}: {c['text'][:180]}..." for c in retrieved_chunks])
                ),
                "sources": [
                    {
                        "document_name": c["document_name"],
                        "page_number": c["page_number"],
                        "chunk_index": c["chunk_index"],
                        "score": c.get("rrf_score") or c.get("score"),
                        "snippet": c["text"][:300],
                    }
                    for c in retrieved_chunks
                ],
                "model_used": "unconfigured",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "chunks_retrieved": len(retrieved_chunks),
                "session_id": session_id,
            }

        # Attempt generation with fallback mechanism
        candidates = [
            selected_model,
            settings.DEFAULT_GROQ_MODEL,
            settings.FALLBACK_GROQ_MODEL,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]
        models_to_try = list(dict.fromkeys([m for m in candidates if m]))
        last_error = None
        answer_text = ""

        for model in models_to_try:
            try:
                llm = ChatGroq(
                    model=model,
                    api_key=groq_api_key,
                    temperature=temperature,
                    max_retries=2,
                )
                chain = prompt | llm | StrOutputParser()
                answer_text = chain.invoke({"context": context_text, "question": question})
                selected_model = model
                break
            except Exception as e:
                logger.warning(f"Generation failed on model {model}: {e}")
                last_error = e

        if not answer_text:
            raise RuntimeError(f"Groq generation failed on all models: {last_error}")

        latency = round((time.time() - start_time) * 1000, 2)

        # Prepare source citations
        sources = [
            {
                "document_name": c["document_name"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "score": c.get("rrf_score") or c.get("score"),
                "snippet": c["text"][:300],
            }
            for c in retrieved_chunks
        ]

        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "model_used": selected_model,
            "latency_ms": latency,
            "chunks_retrieved": len(retrieved_chunks),
            "session_id": session_id,
        }

    def list_documents(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all indexed documents and page/chunk counts."""
        docs = []
        for doc_name, meta in self.documents_metadata.items():
            if session_id and meta.get("session_id") not in [session_id, "default", None]:
                continue
            docs.append(meta)
        return docs

    def delete_document(self, document_name: str) -> bool:
        """Remove a specific document from FAISS and BM25."""
        if self.vector_store is None:
            return False

        # In FAISS, rebuild excluding deleted document
        remaining_chunks = [
            item for item in self.bm25_corpus if item["document_name"] != document_name
        ]
        self.documents_metadata.pop(document_name, None)

        if not remaining_chunks:
            self.clear_all()
            return True

        new_docs = [
            Document(
                page_content=item["text"],
                metadata={
                    "document_name": item["document_name"],
                    "page_number": item["page_number"],
                    "chunk_index": item["chunk_index"],
                    "session_id": item.get("session_id"),
                },
            )
            for item in remaining_chunks
        ]
        self.vector_store = FAISS.from_documents(new_docs, self.embeddings)
        self._rebuild_bm25_from_vectorstore()
        self._save_store()
        return True

    def clear_all(self):
        """Reset all vectors, BM25 indices, and metadata."""
        self.vector_store = None
        self.bm25_corpus = []
        self.bm25_index = None
        self.documents_metadata = {}

        # Remove disk files
        for f in settings.FAISS_INDEX_PATH.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        logger.info("Cleared all indices and vector stores.")


# Singleton RAG instance
rag_engine = RAGEngine()

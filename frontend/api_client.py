"""
HTTP client for DocuMind-RAG backend API
Replaces direct RAGEngine imports with REST API calls
"""
import requests
import streamlit as st
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger("documind.frontend.api")

class HTTPRAGClient:
    def __init__(self, backend_url: str):
        backend_url = (backend_url or "").strip().rstrip('/')
        # Render's fromService hostport gives "host:port" (internal network, no scheme).
        # Normalize to a usable base URL.
        if backend_url and not backend_url.startswith(("http://", "https://")):
            backend_url = f"http://{backend_url}"
        self.backend_url = backend_url
        self.session = requests.Session()
        # Set timeout for requests
        self.session.timeout = 30
    
    def ingest_pdf(self, file_path: str, document_name: str, session_id: str):
        """Upload PDF to backend for processing"""
        try:
            with open(file_path, 'rb') as f:
                files = {'files': (document_name, f, 'application/pdf')}
                data = {'session_id': session_id}
                response = self.session.post(
                    f"{self.backend_url}/api/upload",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error uploading PDF: {e}")
            raise Exception(f"Failed to upload document: {str(e)}")
    
    def answer_query(self, question: str, session_id: str, top_k: int, 
                    model_name: str, temperature: float, search_mode: str):
        """Send query to backend"""
        try:
            payload = {
                "question": question,
                "session_id": session_id,
                "top_k": top_k,
                "model": model_name,
                "temperature": temperature,
                "search_mode": search_mode
            }
            response = self.session.post(
                f"{self.backend_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying backend: {e}")
            raise Exception(f"Query failed: {str(e)}")
    
    def list_documents(self, session_id: Optional[str] = None):
        """Get list of indexed documents"""
        try:
            params = {"session_id": session_id} if session_id else {}
            response = self.session.get(
                f"{self.backend_url}/api/documents",
                params=params
            )
            response.raise_for_status()
            return response.json().get("documents", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing documents: {e}")
            raise Exception(f"Failed to list documents: {str(e)}")
    
    def clear_all(self):
        """Clear all documents"""
        try:
            response = self.session.delete(f"{self.backend_url}/api/documents")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error clearing documents: {e}")
            raise Exception(f"Failed to clear documents: {str(e)}")
    
    def health_check(self):
        """Check backend health"""
        try:
            response = self.session.get(f"{self.backend_url}/api/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Backend health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

def get_http_rag_client():
    """Factory function to get HTTP RAG client"""
    # Try to get backend URL from Streamlit secrets first
    backend_url = None
    try:
        backend_url = st.secrets.get("BACKEND_API_URL")
    except:
        pass
    
    # Fallback to experimental query params or default
    if not backend_url:
        try:
            backend_url = st.experimental_get_query_params().get("backend_url", [""])[0]
        except:
            pass
    
    # If still not found, show error
    if not backend_url:
        st.error("🔧 Configuration Error: BACKEND_API_URL not configured. Please set it in Streamlit secrets or as a query parameter.")
        st.info("For local development, you can set it via: ?backend_url=http://localhost:8000")
        st.stop()
    
    logger.info(f"Initializing HTTP RAG client with backend URL: {backend_url}")
    return HTTPRAGClient(backend_url)
# Deployment Guide - Streamlit + Render

## Changes Deployed (commit: 7e2e951)

- frontend/api_client.py (NEW)
- frontend/app.py (MODIFIED - uses HTTP client)
- render.yaml (MODIFIED - 2 services)

## Render Setup Steps

### 1. Go to Render Dashboard
URL: https://dashboard.render.com/

### 2. Create Blueprint
- Click "New +" → "Blueprint"
- Connect repository: DRohithSubrahmanyaSai/DocuMind-RAG
- Render auto-detects render.yaml

### 3. Configure Services

**Backend (documind-rag-backend):**
- Runtime: Python
- Build: `pip install --upgrade pip && pip install -r requirements.txt`
- Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Health: `/api/health`

**Frontend (documind-rag-frontend):**
- Runtime: Python
- Build: `pip install --upgrade pip && pip install -r requirements.txt`
- Start: `streamlit run frontend/app.py --server.port=$PORT --server.address=0.0.0.0`
- Health: `/_stcore/health`

### 4. Environment Variables (Add to BOTH services)
- `GROQ_API_KEY`: Your Groq API key (sync: false)
- `PYTHON_VERSION`: 3.11.9

### 5. Click Apply
Render will deploy both services automatically.

## Service URLs (after deployment)

- Backend: https://documind-rag-backend.onrender.com
- Frontend: https://documind-rag-frontend.onrender.com

## Verification

1. Backend Health: https://documind-rag-backend.onrender.com/api/health
2. Frontend UI: Access via Render-provided URL
3. Test upload PDF and query functionality

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


class Settings:
    # Project Identity
    PROJECT_NAME: str = "DocuMind-RAG"
    PROJECT_VERSION: str = "2.0.0"
    AUTHOR: str = "D Rohith Subrahmanya Sai"
    REPO_URL: str = "https://github.com/DRohithSubrahmanyaSai/DocuMind-RAG"

    # API Keys & LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEFAULT_GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    FALLBACK_GROQ_MODEL: str = "openai/gpt-oss-20b"

    # Embedding Model
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS: list[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "*",
    ]

    # RAG Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "4"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))

    # Storage Paths
    DATA_DIR: Path = ROOT_DIR / "data"
    FAISS_INDEX_PATH: Path = DATA_DIR / "faiss_index"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()

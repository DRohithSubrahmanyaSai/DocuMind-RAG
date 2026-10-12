"""
DocuMind-RAG Root Entry Point
Allows single-click deployment on Streamlit Community Cloud and local execution:
    streamlit run app.py

Author: D Rohith Subrahmanya Sai
GitHub: https://github.com/DRohithSubrahmanyaSai/DocuMind-RAG
"""

import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute frontend application
from frontend.app import *
"""
DocuMind-RAG Root Entry Point
Streamlit Community Cloud and local execution:
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

# Set page config BEFORE any other streamlit call (must be first)
import streamlit as st
st.set_page_config(
    page_title="DocuMind-RAG | AI Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://github.com/DRohithSubrahmanyaSai/DocuMind-RAG",
        "Report a Bug": "https://github.com/DRohithSubrahmanyaSai/DocuMind-RAG/issues",
        "About": "DocuMind-RAG v2.0 by D Rohith Subrahmanya Sai",
    },
)

from frontend.app import render_app

# Execute application on every run and rerun
render_app()
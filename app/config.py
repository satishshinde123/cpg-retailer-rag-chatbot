import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent


# Dataset
DATA_DIR = BASE_DIR / "data"


# FAISS
VECTOR_STORE_DIR = BASE_DIR / "store" / "vector_index"


# Chunk storage
CHUNKS_DIR = BASE_DIR / "store" / "chunks"


# Create directories
VECTOR_STORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHUNKS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please add it to your .env file."
    )


# Models
EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = "openai/gpt-oss-120b"


# RAG configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
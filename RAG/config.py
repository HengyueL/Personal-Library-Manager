from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DOC_SUMMARY_PATH = PROJECT_ROOT / "doc_summary"
CHROMA_DB_PATH = PROJECT_ROOT / "RAG" / "chroma_db"
MODEL_CACHE_DIR = PROJECT_ROOT / "RAG" / "models"

COLLECTION_NAME = "personal_library"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K_CHUNKS = 8   # number of chunks retrieved from Chroma
TOP_K_DOCS = 5     # number of unique documents after dedup

LLM_MODEL_ID = "google/gemma-4-26B-A4B-it"
LLM_BASE_URL = "https://router.huggingface.co/v1"

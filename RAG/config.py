from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DOC_RAW_PATH = PROJECT_ROOT / "doc_raw"
DOC_SUMMARY_PATH = PROJECT_ROOT / "doc_summary"
RELATION_TABLE_PATH = PROJECT_ROOT / "doc_relation_table.csv"
CHROMA_DB_PATH = PROJECT_ROOT / "RAG" / "chroma_db"

COLLECTION_NAME = "personal_library"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800    # characters (~180 tokens, safely under all-MiniLM-L6-v2's 256-token cap)
CHUNK_OVERLAP = 150

TOP_K_CHUNKS = 8   # number of chunks retrieved from Chroma
TOP_K_DOCS = 5     # number of unique documents after dedup

LLM_MODEL_ID = "openai/gpt-oss-120b"
LLM_BASE_URL = "https://router.huggingface.co/v1"

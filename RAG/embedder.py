import logging

from RAG.config import EMBEDDING_MODEL, MODEL_CACHE_DIR

_model = None


def _is_cached(model_name: str) -> bool:
    hub_dir = "models--" + model_name.replace("/", "--")
    return (MODEL_CACHE_DIR / hub_dir).exists()


def get_model():
    """
    Lazy-load and cache the SentenceTransformer model in-process.
    On first ever run, downloads ~80MB to RAG/models/; subsequent runs load
    from that project-local cache.
    """
    global _model
    if _model is None:
        for noisy in ("sentence_transformers", "httpx", "huggingface_hub"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        from sentence_transformers import SentenceTransformer
        if _is_cached(EMBEDDING_MODEL):
            print(f"Loading embedding model '{EMBEDDING_MODEL}' from local cache...")
        else:
            print(f"Downloading embedding model '{EMBEDDING_MODEL}' (~300MB, one-time)...")
        _model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(MODEL_CACHE_DIR), trust_remote_code=True)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the local SentenceTransformer model.

    Returns:
        List of embedding vectors (normalized for cosine similarity).
    """
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()

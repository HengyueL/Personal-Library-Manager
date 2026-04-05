from RAG.config import EMBEDDING_MODEL

_model = None


def get_model():
    """
    Lazy-load and cache the SentenceTransformer model.
    Downloads ~80MB on first call; subsequent calls return the cached model.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL}' (downloads ~80MB on first run)...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model loaded.")
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

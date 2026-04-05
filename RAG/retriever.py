"""
    Query the Chroma index and return ranked source documents + raw chunks for synthesis.
"""

import chromadb

from RAG.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    TOP_K_CHUNKS,
    TOP_K_DOCS,
)
from RAG import embedder


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(
    query_text: str,
    top_k_chunks: int = TOP_K_CHUNKS,
    top_k_docs: int = TOP_K_DOCS,
) -> dict:
    """
    Embed the query, search Chroma, deduplicate results at document level.

    Returns:
        {
            "ranked_docs": [
                {"file_name": str, "url": str, "score": float, "content_type": str},
                ...  # up to top_k_docs unique files, sorted by best score desc
            ],
            "chunks": [
                {"file_name": str, "text": str, "score": float,
                 "content_type": str, "chunk_index": int},
                ...  # raw chunks only, for LLM synthesis context
            ]
        }
    """
    collection = _get_collection()

    query_vec = embedder.embed([query_text])
    results = collection.query(
        query_embeddings=query_vec,
        n_results=min(top_k_chunks, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # cosine distance in Chroma is 1 - cos_sim (range 0–2); convert to 0–1 score
    scores = [max(0.0, 1.0 - d) for d in distances]

    # Deduplicate: keep best score per source file
    best_per_file: dict[str, dict] = {}
    for doc_id, text, meta, score in zip(ids, documents, metadatas, scores):
        file_name = meta["source"]
        if file_name not in best_per_file or score > best_per_file[file_name]["score"]:
            best_per_file[file_name] = {
                "file_name": file_name,
                "url": meta.get("original_url", ""),
                "score": score,
                "content_type": meta.get("content_type", ""),
            }

    ranked_docs = sorted(best_per_file.values(), key=lambda x: x["score"], reverse=True)
    ranked_docs = ranked_docs[:top_k_docs]

    # Collect all results for LLM synthesis context
    chunks = []
    for doc_id, text, meta, score in zip(ids, documents, metadatas, scores):
        chunks.append({
            "file_name": meta["source"],
            "text": text,
            "score": score,
            "content_type": meta["content_type"],
            "chunk_index": meta.get("chunk_index", 0),
        })

    # Sort chunks by score descending
    chunks.sort(key=lambda x: x["score"], reverse=True)

    return {"ranked_docs": ranked_docs, "chunks": chunks}

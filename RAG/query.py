"""
    RAG query interface — CLI and Python API.

    CLI usage:
        python RAG/query.py "what is harness design?"
        python RAG/query.py "..." --top-k 3
        python RAG/query.py "..." --no-answer          # retrieval only, no LLM call
        python RAG/query.py "..." --rebuild-index      # rebuild Chroma index first

    Python API:
        from RAG import query
        result = query("what is harness design?")
        print(result["answer"])
        print(result["sources"])
"""

import argparse

import chromadb

from RAG.config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K_DOCS
from RAG import retriever


def _ensure_index_built() -> None:
    """Auto-build the index on first run if the collection is empty or missing."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            return
    except Exception:
        pass

    print("Index is empty. Building from doc_raw/ and doc_summary/...")
    from RAG.index import build_index
    build_index()


def query(
    user_query: str,
    top_k_docs: int = TOP_K_DOCS,
    synthesize: bool = True,
) -> dict:
    """
    Run a RAG query against the PersonalLibrary index.

    Args:
        user_query: Natural-language question.
        top_k_docs: Number of source documents to return.
        synthesize: If True, generate an LLM answer. Requires HF_TOKEN env var.

    Returns:
        {
            "query": str,
            "sources": [{"file_name", "url", "score", "content_type"}, ...],
            "answer": str  # empty string if synthesize=False
        }
    """
    _ensure_index_built()

    results = retriever.retrieve(user_query, top_k_docs=top_k_docs)
    ranked_docs = results["ranked_docs"]
    chunks = results["chunks"]

    answer = ""
    if synthesize:
        from RAG.synthesizer import synthesize_answer
        answer = synthesize_answer(user_query, chunks)

    return {
        "query": user_query,
        "sources": ranked_docs,
        "answer": answer,
    }


def _print_results(result: dict) -> None:
    print("\n=== Sources ===")
    for i, doc in enumerate(result["sources"], start=1):
        print(f"{i}. [{doc['score']:.2f}] {doc['file_name']}")
        if doc["url"]:
            print(f"   {doc['url']}")
    if not result["sources"]:
        print("No matching sources found.")

    if result["answer"]:
        print("\n=== Answer ===")
        print(result["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query your PersonalLibrary using RAG."
    )
    parser.add_argument("query_text", help="Your question.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DOCS,
        dest="top_k",
        help=f"Number of source documents to retrieve (default: {TOP_K_DOCS}).",
    )
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Return ranked sources only; skip LLM answer synthesis.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the Chroma index before querying.",
    )
    args = parser.parse_args()

    if args.rebuild_index:
        from RAG.index import build_index
        build_index(rebuild=True)

    result = query(
        user_query=args.query_text,
        top_k_docs=args.top_k,
        synthesize=not args.no_answer,
    )
    _print_results(result)


if __name__ == "__main__":
    main()

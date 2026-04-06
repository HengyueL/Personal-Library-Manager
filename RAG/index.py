"""
    Build and update the Chroma vector index from doc_summary/.

    Index scheme:
    - Summary docs: ID "summary::{file_name}", content_type "summary"

    Usage:
        python RAG/index.py            # incremental (skip already-indexed files)
        python RAG/index.py --rebuild  # wipe and rebuild from scratch
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb

from RAG.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    DOC_SUMMARY_PATH,
)
from RAG.chunking import strip_frontmatter
from RAG import embedder


def get_collection(rebuild: bool = False):
    """Open (or create) the persistent Chroma collection."""
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def get_indexed_ids(collection) -> set[str]:
    """Return the set of all IDs currently in the collection."""
    result = collection.get(include=[])
    return set(result["ids"])


def _read_summary_doc(file_name: str) -> tuple[str, str, str]:
    """Read a summary doc; return (doc_id, body, url)."""
    summary_path = DOC_SUMMARY_PATH / file_name
    with open(summary_path, "r", encoding="utf-8") as f:
        text = f.read()
    body, frontmatter = strip_frontmatter(text)
    url = frontmatter.get("url", "")
    return f"summary::{file_name}", body, url


def index_summary_doc(file_name: str, collection) -> None:
    """Read a summary doc, extract URL from frontmatter, embed it, and upsert into the collection."""
    summary_path = DOC_SUMMARY_PATH / file_name
    with open(summary_path, "r", encoding="utf-8") as f:
        text = f.read()

    body, frontmatter = strip_frontmatter(text)
    url = frontmatter.get("url", "")

    vectors = embedder.embed([body])
    collection.upsert(
        ids=[f"summary::{file_name}"],
        embeddings=vectors,
        documents=[body],
        metadatas=[{"source": file_name, "content_type": "summary", "original_url": url}],
    )


def build_index(rebuild: bool = False) -> None:
    """
    Scan doc_summary/, embed and upsert documents into Chroma.
    Skips already-indexed files unless rebuild=True.
    """
    if not DOC_SUMMARY_PATH.exists():
        print(f"Error: doc_summary directory not found: {DOC_SUMMARY_PATH}")
        return

    collection = get_collection(rebuild=rebuild)
    existing_ids = get_indexed_ids(collection)

    summary_files = sorted(DOC_SUMMARY_PATH.glob("*.md"))
    if not summary_files:
        print("No markdown files found in doc_summary/.")
        return

    to_index = [
        doc_path.name for doc_path in summary_files
        if f"summary::{doc_path.name}" not in existing_ids or rebuild
    ]
    to_skip = [
        doc_path.name for doc_path in summary_files
        if f"summary::{doc_path.name}" in existing_ids and not rebuild
    ]

    for file_name in to_skip:
        print(f"Skipping {file_name} (already indexed).")

    if to_index:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            records = list(executor.map(_read_summary_doc, to_index))

        bodies = [body for _, body, _ in records]
        vectors = embedder.embed(bodies)

        collection.upsert(
            ids=[doc_id for doc_id, _, _ in records],
            embeddings=vectors,
            documents=bodies,
            metadatas=[
                {"source": fn, "content_type": "summary", "original_url": url}
                for fn, (_, _, url) in zip(to_index, records)
            ],
        )
        for file_name in to_index:
            print(f"Indexed {file_name}.")

    total = len(to_index)
    print(f"\nDone. Indexed {total} summary doc(s).")
    print(f"Total vectors in collection: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the PersonalLibrary RAG index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the existing index and rebuild from scratch.",
    )
    args = parser.parse_args()
    build_index(rebuild=args.rebuild)

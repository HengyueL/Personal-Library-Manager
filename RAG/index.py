"""
    Build and update the Chroma vector index from doc_raw/ and doc_summary/.

    Index scheme:
    - Raw doc chunks: ID "raw::{file_name}::{chunk_idx}", content_type "raw_chunk"
    - Summary docs:   ID "summary::{file_name}",          content_type "summary"

    Usage:
        python RAG/index.py            # incremental (skip already-indexed files)
        python RAG/index.py --rebuild  # wipe and rebuild from scratch
"""

import argparse
import csv

import chromadb

from RAG.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOC_RAW_PATH,
    DOC_SUMMARY_PATH,
    RELATION_TABLE_PATH,
)
from RAG.chunking import strip_frontmatter, chunk_text
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


def load_url_map() -> dict[str, str]:
    """Read doc_relation_table.csv and return {file_name: original_url}."""
    url_map = {}
    if not RELATION_TABLE_PATH.exists():
        return url_map
    with open(RELATION_TABLE_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url_map[row["file_name"]] = row.get("orignal_url", "")
    return url_map


def get_indexed_ids(collection) -> set[str]:
    """Return the set of all IDs currently in the collection."""
    result = collection.get(include=[])
    return set(result["ids"])


def index_raw_doc(file_name: str, url: str, collection) -> int:
    """
    Read a raw doc, chunk it, embed chunks, and upsert into the collection.

    Returns:
        Number of chunks indexed.
    """
    doc_path = DOC_RAW_PATH / file_name
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    body, frontmatter = strip_frontmatter(text)
    # Fall back to frontmatter URL if CSV didn't have one
    effective_url = url or frontmatter.get("url", "")

    chunks = chunk_text(body, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        print(f"  Warning: no chunks produced for {file_name}, skipping.")
        return 0

    vectors = embedder.embed(chunks)
    ids = [f"raw::{file_name}::{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": file_name,
            "chunk_index": i,
            "content_type": "raw_chunk",
            "original_url": effective_url,
        }
        for i in range(len(chunks))
    ]

    collection.upsert(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)
    return len(chunks)


def index_summary_doc(file_name: str, url: str, collection) -> None:
    """Read a summary doc, embed it as a single vector, and upsert into the collection."""
    summary_path = DOC_SUMMARY_PATH / file_name
    with open(summary_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Summaries have no frontmatter; strip is a no-op but safe to call
    body, _ = strip_frontmatter(text)

    vectors = embedder.embed([body])
    collection.upsert(
        ids=[f"summary::{file_name}"],
        embeddings=vectors,
        documents=[body],
        metadatas=[{"source": file_name, "content_type": "summary", "original_url": url}],
    )


def build_index(rebuild: bool = False) -> None:
    """
    Scan doc_raw/ and doc_summary/, embed and upsert documents into Chroma.
    Skips already-indexed files unless rebuild=True.
    """
    if not DOC_RAW_PATH.exists():
        print(f"Error: doc_raw directory not found: {DOC_RAW_PATH}")
        return
    if not DOC_SUMMARY_PATH.exists():
        print(f"Error: doc_summary directory not found: {DOC_SUMMARY_PATH}")
        return

    collection = get_collection(rebuild=rebuild)
    existing_ids = get_indexed_ids(collection)
    url_map = load_url_map()

    raw_files = sorted(DOC_RAW_PATH.glob("*.md"))
    if not raw_files:
        print("No markdown files found in doc_raw/.")
        return

    total_chunks = 0
    total_summaries = 0

    for doc_path in raw_files:
        file_name = doc_path.name
        url = url_map.get(file_name, "")

        # --- Raw chunks ---
        first_chunk_id = f"raw::{file_name}::0"
        if first_chunk_id in existing_ids and not rebuild:
            print(f"Skipping {file_name} (already indexed).")
        else:
            n = index_raw_doc(file_name, url, collection)
            print(f"Indexed {file_name}: {n} chunk(s).")
            total_chunks += n

        # --- Summary ---
        summary_path = DOC_SUMMARY_PATH / file_name
        summary_id = f"summary::{file_name}"
        if not summary_path.exists():
            print(f"  Warning: no summary found for {file_name}, skipping summary index.")
            continue
        if summary_id in existing_ids and not rebuild:
            pass  # already indexed alongside raw, no separate message needed
        else:
            index_summary_doc(file_name, url, collection)
            total_summaries += 1

    print(f"\nDone. Indexed {total_chunks} raw chunk(s) and {total_summaries} summary doc(s).")
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

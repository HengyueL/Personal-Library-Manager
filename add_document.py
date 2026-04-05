"""
    Starter script to add a new document to PersonalLibrary.

    Steps:
        1. Edit `URL` and `FILE_NAME` below.
        2. Run: python add_document.py

    This script fetches the document, generates its summary, updates the
    relation table, and incrementally updates the RAG index.
"""

from utils.fetch_html import fetch_html
from utils.generate_summary import process_document
from utils.update_relation_table import update_relation_table
from RAG.index import build_index

URL = "https://example.com/your-article"
FILE_NAME = "Your_Article_Name.md"

if __name__ == "__main__":
    print(f"=== Fetching: {URL} ===")
    fetch_html(url=URL, file_name=FILE_NAME)

    print(f"\n=== Generating summary for: {FILE_NAME} ===")
    process_document(file_name=FILE_NAME)

    print("\n=== Updating relation table ===")
    update_relation_table()

    print("\n=== Updating RAG index ===")
    build_index(rebuild=False)

    print(f"\nDone. '{FILE_NAME}' is now searchable via RAG.")

"""
    Starter script to add a new document to PersonalLibrary.

    Usage:
        python add_document.py <url> <file_name>

    Example:
        python add_document.py https://example.com/article My_Article.md

    This script fetches the document, generates its summary, updates the
    relation table, and incrementally updates the RAG index.
"""

import argparse
import logging

from utils.fetch_document import fetch_document
from utils.generate_summary import process_document
from utils.update_relation_table import update_relation_table
from RAG.index import build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a document and add it to PersonalLibrary."
    )
    parser.add_argument("--url", required=True, help="URL of the document to fetch")
    parser.add_argument("--file_name", required=True, help="Output filename (e.g. My_Article.md)")
    args = parser.parse_args()

    logger.info("Fetching: %s", args.url)
    fetch_document(url=args.url, file_name=args.file_name)

    logger.info("Generating summary for: %s", args.file_name)
    process_document(file_name=args.file_name)

    logger.info("Updating relation table")
    update_relation_table()

    logger.info("Updating RAG index")
    build_index(rebuild=False)

    logger.info("Done. '%s' is now searchable via RAG.", args.file_name)


if __name__ == "__main__":
    main()

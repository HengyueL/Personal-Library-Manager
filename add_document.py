"""
    Starter script to add a new document to PersonalLibrary.

    Usage:
        python add_document.py --url <url> --file_name <file_name>

    Example:
        python add_document.py --url https://example.com/article --file_name My_Article.md

    This script fetches the document, generates its summary (saved to doc_summary/
    with the source URL in YAML frontmatter), and incrementally updates the RAG index.
"""

import argparse
import logging

from utils.fetch_document import fetch_document
from utils.generate_summary import generate_summary, save_summary
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
    content = fetch_document(url=args.url)

    logger.info("Generating summary for: %s", args.file_name)
    summary = generate_summary(content)

    save_summary(file_name=args.file_name, summary_text=summary, url=args.url)

    logger.info("Updating RAG index")
    build_index(rebuild=False)

    logger.info("Done. '%s' is now searchable via RAG.", args.file_name)


if __name__ == "__main__":
    main()

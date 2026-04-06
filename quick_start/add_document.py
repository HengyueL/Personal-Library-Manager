"""
    Starter script to add a new document to PersonalLibrary.

    Usage:
        python add_document.py --url <url> [--name <file_name>]

    Examples:
        python add_document.py --url https://example.com/article --name My_Article.md
        python add_document.py --url https://example.com/article   # auto-generates filename

    This script fetches the document, generates its summary (saved to doc_summary/
    with the source URL in YAML frontmatter), and incrementally updates the RAG index.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.fetch_document import AuthRequiredError, fetch_document
from utils.generate_summary import generate_summary, generate_summary_with_filename, save_summary
from utils.file_naming import derive_file_name
from RAG.index import build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a document and add it to PersonalLibrary."
    )
    parser.add_argument("--url", required=True, help="URL of the document to fetch")
    parser.add_argument(
        "--name",
        default=None,
        metavar="FILENAME",
        help="Output filename (e.g. My_Article.md). Auto-generated from title if omitted.",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        metavar="FILE",
        help="Path to a Netscape cookies.txt file for login-gated URLs.",
    )
    args = parser.parse_args()

    logger.info("Fetching: %s", args.url)
    try:
        content = fetch_document(url=args.url, cookies_path=args.cookies)
    except AuthRequiredError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if args.name:
        file_name = args.name
        logger.info("Generating summary for: %s", file_name)
        summary = generate_summary(content)
    else:
        logger.info("Generating summary and filename via LLM...")
        summary, file_name = generate_summary_with_filename(content, args.url)
        logger.info("LLM-proposed filename: %s", file_name)

    save_summary(file_name=file_name, summary_text=summary, url=args.url)

    logger.info("Updating RAG index")
    build_index(rebuild=False)

    logger.info("Done. '%s' is now searchable via RAG.", file_name)


if __name__ == "__main__":
    main()

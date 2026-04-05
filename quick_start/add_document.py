"""
    Starter script to add a new document to PersonalLibrary.

    Usage:
        python add_document.py --url <url> [--file_name <file_name>]

    Examples:
        python add_document.py --url https://example.com/article --file_name My_Article.md
        python add_document.py --url https://example.com/article   # auto-generates filename

    This script fetches the document, generates its summary (saved to doc_summary/
    with the source URL in YAML frontmatter), and incrementally updates the RAG index.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.fetch_document import fetch_document
from utils.generate_summary import generate_summary, save_summary
from RAG.index import build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _derive_file_name(url: str, content: str) -> str:
    """Generate a 'Source-Title.md' filename from the URL domain and document title."""
    # --- Source: top-level domain name (e.g. anthropic.com → Anthropic) ---
    hostname = urlparse(url).hostname or ""
    # Strip www. prefix and take the first domain label
    hostname = re.sub(r"^www\.", "", hostname)
    source = hostname.split(".")[0].capitalize() if hostname else "Doc"

    # --- Title: first H1 heading in the markdown content ---
    title = ""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            title = line[2:].strip()
            break

    # Fall back to the URL path's last segment if no H1 found
    if not title:
        path_part = urlparse(url).path.rstrip("/").split("/")[-1]
        title = path_part or "Document"

    # Sanitize: keep alphanumerics, spaces, hyphens, underscores; collapse the rest
    def sanitize(s: str) -> str:
        s = re.sub(r"[^\w\s-]", "", s)          # remove punctuation
        s = re.sub(r"[\s_]+", "-", s.strip())   # spaces/underscores → hyphen
        s = re.sub(r"-{2,}", "-", s)             # collapse repeated hyphens
        return s

    source_part = sanitize(source)
    title_part = sanitize(title)

    return f"{source_part}-{title_part}.md"


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a document and add it to PersonalLibrary."
    )
    parser.add_argument("--url", required=True, help="URL of the document to fetch")
    parser.add_argument(
        "--file_name",
        default=None,
        help="Output filename (e.g. My_Article.md). Auto-generated from title if omitted.",
    )
    args = parser.parse_args()

    logger.info("Fetching: %s", args.url)
    content = fetch_document(url=args.url)

    if args.file_name:
        file_name = args.file_name
    else:
        file_name = _derive_file_name(args.url, content)
        logger.info("Auto-generated filename: %s", file_name)

    logger.info("Generating summary for: %s", file_name)
    summary = generate_summary(content)

    save_summary(file_name=file_name, summary_text=summary, url=args.url)

    logger.info("Updating RAG index")
    build_index(rebuild=False)

    logger.info("Done. '%s' is now searchable via RAG.", file_name)


if __name__ == "__main__":
    main()

import logging
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import html2text
import requests
from markitdown import MarkItDown

logger = logging.getLogger(__name__)

DESTINATION_PATH = Path(__file__).parent.parent / "doc_raw"


def _is_pdf(url: str) -> bool:
    if urlparse(url).path.lower().endswith(".pdf"):
        return True
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10)
        return "application/pdf" in resp.headers.get("Content-Type", "")
    except requests.RequestException:
        return False


def fetch_document(url: str, file_name: str):
    save_path = DESTINATION_PATH / file_name

    if save_path.exists():
        logger.error("File already exists: %s", save_path)
        return

    if _is_pdf(url):
        logger.info("Detected PDF; converting with MarkItDown")
        markdown = _fetch_pdf(url)
    else:
        logger.info("Detected HTML; converting with html2text")
        markdown = _fetch_html(url)

    yaml_frontmatter = f"---\nurl: {url}\n---\n\n"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(yaml_frontmatter + markdown)

    logger.info("Document saved to: %s", save_path)


def _fetch_html(url: str) -> str:
    html = requests.get(url).text
    return html2text.html2text(html)


def _fetch_pdf(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = Path(tmp.name)

    try:
        result = MarkItDown().convert(str(tmp_path))
        return result.text_content
    finally:
        tmp_path.unlink(missing_ok=True)

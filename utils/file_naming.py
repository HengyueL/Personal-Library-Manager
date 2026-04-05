"""Utility for deriving a safe filename from a document URL and content."""

import re
from urllib.parse import urlparse


def derive_file_name(url: str, content: str) -> str:
    """Generate a 'Source-Title.md' filename from the URL domain and document title."""
    # --- Source: top-level domain name (e.g. anthropic.com → Anthropic) ---
    hostname = urlparse(url).hostname or ""
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

    def sanitize(s: str) -> str:
        s = re.sub(r"[^\w\s-]", "", s)
        s = re.sub(r"[\s_]+", "-", s.strip())
        s = re.sub(r"-{2,}", "-", s)
        return s

    return f"{sanitize(source)}-{sanitize(title)}.md"

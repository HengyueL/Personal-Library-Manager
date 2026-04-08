"""
    Use LLM to generate an abstract/summary of a document and save it to doc_summary/.
"""

import re
from pathlib import Path
from urllib.parse import urlparse

from utils.llm_client import complete

DOC_SUMMARY_PATH = Path(__file__).parent.parent / "doc_summary"


def _parse_filename_and_summary(response: str, url: str) -> tuple[str, str]:
    """Extract (filename, summary) from a structured LLM response."""
    filename = ""
    summary = response

    lines = response.splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("FILENAME:"):
            raw = line.split(":", 1)[1].strip()
            # Keep only safe characters, ensure .md extension
            raw = re.sub(r"[^\w.\-]", "", raw)
            filename = raw if raw.endswith(".md") else raw + ".md"
            # Summary is everything after the FILENAME line, stripping a leading "SUMMARY:" marker
            rest = "\n".join(lines[i + 1:]).strip()
            if rest.upper().startswith("SUMMARY:"):
                rest = rest[len("SUMMARY:"):].strip()
            summary = rest
            break

    if not filename:
        # Fallback: derive source from URL
        hostname = urlparse(url).hostname or ""
        hostname = re.sub(r"^www\.", "", hostname)
        source = hostname.split(".")[0].capitalize() if hostname else "Doc"
        filename = f"{source}-Document.md"

    return summary, filename


def generate_summary(content: str, custom_prompt: str | None = None) -> str:
    """
    Use LLM to generate an abstract/summary of document content.

    Args:
        content: Raw document text to summarize.
        custom_prompt: Optional additional instructions appended to the
            default summarization prompt.

    Returns:
        Generated summary as a string.
    """
    prompt = f"""Please read the following document and generate a concise abstract/summary that captures the main points and key takeaways.

Document content:
{content}

Please provide a well-structured summary that includes:
1. The main topic/purpose of the document
2. Key points and findings (arrange them in a table)
3. Any important conclusions or recommendations

Summary:"""

    if custom_prompt:
        prompt += f"\n\nAdditional instructions from the user:\n{custom_prompt}"

    return complete(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
    )


def generate_summary_with_filename(content: str, url: str, custom_prompt: str | None = None) -> tuple[str, str]:
    """
    Use LLM to generate a summary and propose a filename in a single call.

    Args:
        content: Raw document text to summarize.
        url: Source URL, provided to the LLM as context for naming.
        custom_prompt: Optional additional instructions appended to the
            default summarization prompt.

    Returns:
        (summary, filename) where filename follows the 'Source-Title.md' pattern.
    """
    prompt = f"""Please read the following document (source: {url}) and:
1. Propose a filename in the format "Source-Title.md" where Source is the website or organization name and Title is a short descriptive title (use hyphens, no spaces, alphanumeric only).
2. Generate a concise abstract/summary that captures the main points and key takeaways.

Document content:
{content}

Respond in exactly this format (no extra text before FILENAME):
FILENAME: <proposed-filename.md>

SUMMARY:
<well-structured summary including main topic/purpose of the document, key points and findings (arrange them in a table), and conclusions>"""

    if custom_prompt:
        prompt += f"\n\nAdditional instructions from the user:\n{custom_prompt}"

    response = complete(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.7,
    ) or ""
    return _parse_filename_and_summary(response, url)


def save_summary(file_name: str, summary_text: str, url: str) -> None:
    """
    Save a summary to doc_summary/ with YAML frontmatter containing the source URL.

    Skips if the file already exists.

    Args:
        file_name: Output filename (e.g. "My_Article.md").
        summary_text: The summary content to save.
        url: Original source URL to embed in frontmatter.
    """
    summary_path = DOC_SUMMARY_PATH / file_name
    if summary_path.exists():
        print(f"Summary already exists for: {file_name}, skipping...")
        return

    DOC_SUMMARY_PATH.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"---\nurl: {url}\n---\n\n")
        f.write(f"# Summary of {file_name}\n\n")
        f.write(summary_text)

    print(f"Summary saved to: {summary_path}")

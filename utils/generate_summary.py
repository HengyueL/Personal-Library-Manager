"""
    Use LLM to generate an abstract/summary of a document and save it to doc_summary/.
"""

import os
from pathlib import Path
from openai import OpenAI

MODEL_ID = "openai/gpt-oss-120b"
_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

DOC_SUMMARY_PATH = Path(__file__).parent.parent / "doc_summary"


def generate_summary(content: str) -> str:
    """
    Use LLM to generate an abstract/summary of document content.

    Args:
        content: Raw document text to summarize.

    Returns:
        Generated summary as a string.
    """
    prompt = f"""Please read the following document and generate a concise abstract/summary that captures the main points and key takeaways.

Document content:
{content}

Please provide a well-structured summary that includes:
1. The main topic/purpose of the document
2. Key points and findings
3. Any important conclusions or recommendations

Summary:"""

    completion = _client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
    )

    return completion.choices[0].message.content


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

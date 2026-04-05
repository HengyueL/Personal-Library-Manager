"""
    Reads a document in @doc_raw, use LLM to generate an abstract/summary and save it to @doc_summary.
"""

import os
from pathlib import Path
from openai import OpenAI

MODEL_ID = "openai/gpt-oss-120b"
_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# Paths
DOC_RAW_PATH = Path(__file__).parent.parent / "doc_raw"
DOC_SUMMARY_PATH = Path(__file__).parent.parent / "doc_summary"


def generate_summary(doc_path: Path) -> str:
    """
    Read a document and use LLM to generate an abstract/summary.
    
    Args:
        doc_path: Path to the document file
        
    Returns:
        Generated summary as a string
    """
    # Read the document content
    with open(doc_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Create prompt for summary generation
    prompt = f"""Please read the following document and generate a concise abstract/summary that captures the main points and key takeaways.

Document content:
{content}

Please provide a well-structured summary that includes:
1. The main topic/purpose of the document
2. Key points and findings
3. Any important conclusions or recommendations

Summary:"""

    # Call LLM to generate summary
    completion = _client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
    )
    
    return completion.choices[0].message.content


def process_document(file_name: str) -> None:
    """
    Process a single document: read from doc_raw, generate summary, save to doc_summary.
    
    Args:
        file_name: Name of the document file to process
    """
    doc_path = DOC_RAW_PATH / file_name
    
    # Check if file exists
    if not doc_path.exists():
        print(f"Error: Document not found: {doc_path}")
        return
    
    # Check if summary already exists
    summary_path = DOC_SUMMARY_PATH / file_name
    if summary_path.exists():
        print(f"Summary already exists for: {file_name}, skipping...")
        return
    
    # Generate summary
    print(f"Generating summary for: {file_name}")
    summary = generate_summary(doc_path)
    
    # Ensure doc_summary directory exists
    DOC_SUMMARY_PATH.mkdir(parents=True, exist_ok=True)
    
    # Create summary file path (same name as original)
    summary_path = DOC_SUMMARY_PATH / file_name
    
    # Save summary
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(f"# Summary of {file_name}\n\n")
        file.write(summary)
    
    print(f"Summary saved to: {summary_path}")


def process_all_documents() -> None:
    """
    Process all documents in the doc_raw directory.
    """
    # Ensure doc_raw directory exists
    if not DOC_RAW_PATH.exists():
        print(f"Error: doc_raw directory not found: {DOC_RAW_PATH}")
        return
    
    # Find all markdown files in doc_raw
    markdown_files = list(DOC_RAW_PATH.glob("*.md"))
    
    if not markdown_files:
        print("No markdown files found in doc_raw directory")
        return
    
    print(f"Found {len(markdown_files)} document(s) to process")
    
    for doc_path in markdown_files:
        process_document(doc_path.name)


if __name__ == "__main__":
    # Process all documents in doc_raw
    process_all_documents()

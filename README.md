# PersonalLibrary
A private project to effectively manage the papers/projects that I am interested in.

## Design

1. `doc_summary/` stores LLM-generated summaries for each document, with a YAML frontmatter `url:` header recording the original source. Both HTML pages and PDF files are supported.
2. `RAG/` implements a RAG system for querying across all stored summaries — returns a ranked list of relevant sources and a synthesized, cited answer.
3. `tests/` contains the pytest suite covering all `utils/` and `RAG/` modules.

## Setup (Pre-requisites)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=<your_huggingface_token>
```

## Adding a Document

HTML pages and PDFs are both supported — the type is detected automatically from the URL extension or HTTP `Content-Type` header.

```bash
# Add an HTML article (explicit filename)
python quick_start/add_document.py --url https://example.com/article --file_name My_Article.md

# Add a PDF (by extension)
python quick_start/add_document.py --url https://example.com/paper.pdf --file_name My_Paper.md

# Auto-generate filename from domain + document title (Source-Title.md format)
python quick_start/add_document.py --url https://arxiv.org/pdf/2303.08774
```

Fetches and converts the document in memory, generates a summary, saves it to `doc_summary/` with the source URL in frontmatter, and incrementally updates the RAG index in one shot.

## Running Tests

```bash
# Preferred: via uv
uv run python -m pytest tests/ -q

# Alternative: direct venv binary
.venv/bin/python -m pytest tests/ -q
```

## Querying

```bash
python quick_start/retrieve_document.py --query "your question here"
python quick_start/retrieve_document.py --query "your question here" --top-k 3
python quick_start/retrieve_document.py --query "your question here" --no-answer   # retrieval only, no LLM call
```

Or use the lower-level CLI directly:

```bash
python RAG/query.py "what is harness design?"
python RAG/query.py "..." --no-answer
python RAG/query.py "..." --top-k 3
```

The first query call auto-builds the RAG index if it doesn't exist yet.

## Python API

```python
from RAG import query

result = query("how should long-running agent harnesses be designed?")
print(result["answer"])
for doc in result["sources"]:
    print(doc["score"], doc["file_name"], doc["url"])
```

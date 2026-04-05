# PersonalLibrary
A private project to effectively manage the papers/projects that I am interested in.

## Design

1. `doc_raw/` stores all raw documents fetched by `utils/fetch_document.py` as markdown with a YAML frontmatter `url:` header. Both HTML pages and PDF files are supported.
2. `doc_summary/` stores LLM-generated abstracts for each corresponding document in `doc_raw/`, produced by `utils/generate_summary.py`.
3. `doc_relation_table.csv` is a metadata index maintained by `utils/update_relation_table.py`.
4. `RAG/` implements a RAG system for querying across all stored documents — returns a ranked list of relevant sources and a synthesized, cited answer.
5. `tests/` contains the pytest suite covering all `utils/` and `RAG/` modules.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=<your_huggingface_token>
```

## Adding a Document

HTML pages and PDFs are both supported — the type is detected automatically from the URL extension or HTTP `Content-Type` header.

```bash
# Add an HTML article
python add_document.py --url https://example.com/article --file_name My_Article.md

# Add a PDF (by extension)
python add_document.py --url https://example.com/paper.pdf --file_name My_Paper.md

# Add a PDF (no .pdf extension — detected via Content-Type)
python add_document.py --url https://arxiv.org/pdf/2303.08774 --file_name Attention_Paper.md
```

Fetches and converts the document, generates a summary, updates the relation table, and incrementally updates the RAG index in one shot.

## Running Tests

```bash
# Preferred: via uv
uv run python -m pytest tests/ -q

# Alternative: direct venv binary
.venv/bin/python -m pytest tests/ -q
```

## Querying

```bash
python retrieve_document.py "your question here"
python retrieve_document.py "your question here" --top-k 3
python retrieve_document.py "your question here" --no-answer   # retrieval only, no LLM call
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

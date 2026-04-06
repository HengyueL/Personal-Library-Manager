# PersonalLibrary
A private project to effectively manage the papers/projects that I am interested in.

## Design

1. `doc_summary/` stores LLM-generated summaries for each document, with a YAML frontmatter `url:` header recording the original source. Both HTML pages and PDF files are supported.
2. `RAG/` implements a RAG system for querying across all stored summaries — returns a ranked list of relevant sources and a synthesized, cited answer.
3. `tests/` contains the pytest suite covering all `utils/` and `RAG/` modules.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .              # installs all deps and registers the `plib` CLI
export HF_TOKEN=<your_huggingface_token>
```

## Web GUI

```bash
plib gui
plib gui --port 8080 --share
```

Press **Ctrl+C** to shut down the server gracefully.

The GUI has four tabs:

| Tab | What it does |
|-----|--------------|
| **Add Document** | Fetch a URL (HTML or PDF), generate an AI summary, save it to the library, and register it in the RAG vector index. Progress streams in real-time. |
| **Find Document** | Ask a natural-language question; returns ranked source documents and an AI-synthesized answer (requires `HF_TOKEN`). Click any filename in the results table to jump directly to View Document. |
| **View Document** | Browse and render any document in `doc_summary/` as formatted markdown; shows the original source URL. Dropdown defaults to empty — select a document to display it. |
| **Rebuild Index** | Re-embed all documents into the vector index (full rebuild or incremental). Output streams in real-time. |

## CLI

```bash
# Add a document
plib add --url https://example.com/article --name My_Article.md
plib add --url https://arxiv.org/pdf/2303.08774        # LLM proposes filename from content

# Query
plib query --query "your question here"
plib query --query "your question here" --top-k 3
plib query --query "your question here" --retrieval-only

# Rebuild index
plib rebuild
plib rebuild --incremental
```

## Running Tests

```bash
uv run python -m pytest tests/ -q
```

## Python API

```python
from RAG import query

result = query("how should long-running agent harnesses be designed?")
print(result["answer"])
for doc in result["sources"]:
    print(doc["score"], doc["file_name"], doc["url"])
```

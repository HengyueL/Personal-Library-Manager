# PersonalLibrary
A private project to effectively manage the papers/projects that I am interested in.

## Design

1. `doc_raw/` stores all raw documents fetched by `utils/fetch_html.py` as markdown with a YAML frontmatter `url:` header.
2. `doc_summary/` stores LLM-generated abstracts for each corresponding document in `doc_raw/`, produced by `utils/generate_summary.py`.
3. `doc_relation_table.csv` is a metadata index maintained by `utils/update_relation_table.py`.
4. `RAG/` implements a RAG system for querying across all stored documents — returns a ranked list of relevant sources and a synthesized, cited answer.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=<your_huggingface_token>
```

## Adding a Document

Edit the `URL` and `FILE_NAME` variables in `add_document.py`, then run:

```bash
python add_document.py
```

This fetches the page, generates a summary, updates the relation table, and incrementally updates the RAG index in one shot.

## Querying

Edit the `user_query` variable in `retrieve_document.py`, then run:

```bash
python retrieve_document.py
```

Or use the CLI directly:

```bash
python RAG/query.py "what is harness design?"
python RAG/query.py "..." --no-answer   # retrieval only, no LLM call
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

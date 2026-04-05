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

## Workflow

```bash
# 1. Fetch a document
python utils/fetch_html.py

# 2. Generate its summary
python utils/generate_summary.py

# 3. Update the relation table
python utils/update_relation_table.py

# 4. Build (or update) the RAG index
python RAG/index.py

# 5. Query
python RAG/query.py "what is harness design?"
python RAG/query.py "..." --no-answer   # retrieval only
python RAG/query.py "..." --top-k 3
```

The first `RAG/query.py` call auto-builds the index if it doesn't exist yet.

## Python API

```python
from RAG import query

result = query("how should long-running agent harnesses be designed?")
print(result["answer"])
for doc in result["sources"]:
    print(doc["score"], doc["file_name"], doc["url"])
```

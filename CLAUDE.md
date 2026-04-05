# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PersonalLibrary** is a document management and RAG (Retrieval-Augmented Generation) system for managing papers and articles of interest. The workflow is: fetch HTML → convert to markdown → generate AI summary → update metadata index → build vector index → query with natural language.

## Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Required environment variable for summary generation
export HF_TOKEN=<your_huggingface_token>
```

## Common Commands

```bash
# Fetch a document from a URL and save to doc_raw/
python utils/fetch_html.py

# Generate AI summaries for all documents in doc_raw/
python utils/generate_summary.py

# Rebuild the metadata relation table (doc_relation_table.csv)
python utils/update_relation_table.py

# Build / incrementally update the RAG vector index
python RAG/index.py

# Rebuild the index from scratch
python RAG/index.py --rebuild

# Query (auto-builds index on first run; requires HF_TOKEN for answer synthesis)
python RAG/query.py "your question here"
python RAG/query.py "your question here" --no-answer   # retrieval only, no LLM call
python RAG/query.py "your question here" --top-k 3
```

## Architecture

### Directory Structure
- `doc_raw/` — Raw documents as markdown (with YAML frontmatter containing source URL)
- `doc_summary/` — AI-generated summaries, parallel to `doc_raw/` (same filenames)
- `utils/` — Standalone utility scripts (fetch, summarize, update index table)
- `RAG/` — RAG system implementation
- `RAG/chroma_db/` — Persistent Chroma vector store (gitignored, auto-created)
- `doc_relation_table.csv` — CSV metadata index: `index`, `file_name`, `orignal_url` (note: typo in column name is intentional/existing)

### Full Pipeline

```
fetch_html.py → doc_raw/*.md
                    ↓
          generate_summary.py → doc_summary/*.md
                    ↓
        update_relation_table.py → doc_relation_table.csv
                    ↓
            RAG/index.py → RAG/chroma_db/
                    ↓
            RAG/query.py → ranked sources + synthesized answer
```

### Data Flow Details

1. **`utils/fetch_html.py`** — Downloads HTML via `requests`, converts to markdown with `html2text`, prepends YAML frontmatter (`url: <source>`), saves to `doc_raw/<file_name>.md`. Refuses to overwrite existing files.

2. **`utils/generate_summary.py`** — Reads from `doc_raw/`, calls HuggingFace Inference API (OpenAI-compatible client, model `openai/gpt-oss-120b`) to produce structured summaries, saves to `doc_summary/<file_name>.md`. Skips already-summarized documents.

3. **`utils/update_relation_table.py`** — Scans `doc_raw/`, validates that corresponding `doc_summary/` files exist, extracts source URLs from YAML frontmatter, writes/updates `doc_relation_table.csv`. Removes stale entries for deleted files.

4. **`RAG/index.py`** — Reads `doc_raw/` and `doc_summary/`. Raw docs are stripped of frontmatter, chunked (800 chars, 150-char overlap), embedded via local `all-MiniLM-L6-v2`. Summaries are embedded whole. All vectors upserted into Chroma with cosine distance. Idempotent — skips already-indexed files.

5. **`RAG/query.py`** — Embeds the query, retrieves top-k chunks from Chroma, deduplicates results per source file, then calls the HF LLM to synthesize a cited answer. Returns `{query, sources, answer}`. Importable as `from RAG import query`.

### RAG Module Layout

| File | Role |
|---|---|
| `RAG/config.py` | Constants: paths, model IDs, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`, `TOP_K_CHUNKS=8`, `TOP_K_DOCS=5` |
| `RAG/chunking.py` | `strip_frontmatter(text)` + `chunk_text(text, size, overlap)` |
| `RAG/embedder.py` | Lazy-load `all-MiniLM-L6-v2` singleton + `embed(texts)` |
| `RAG/index.py` | `build_index(rebuild=False)` — Chroma upsert pipeline |
| `RAG/retriever.py` | `retrieve(query_text)` — Chroma search + per-file dedup |
| `RAG/synthesizer.py` | `synthesize_answer(query, chunks)` — LLM call with citation prompt |
| `RAG/query.py` | `query(user_query, top_k_docs, synthesize)` — orchestrator + CLI |

### Indexing Scheme

- Raw chunk IDs: `raw::{file_name}::{chunk_idx}`, metadata `content_type="raw_chunk"`
- Summary IDs: `summary::{file_name}`, metadata `content_type="summary"`
- Chroma collection uses cosine distance (`hnsw:space=cosine`); score = `1 - distance` (0–1, higher is better)
- Only raw chunks (not summaries) are sent to the LLM synthesis prompt

### Document Format

Every file in `doc_raw/` begins with:
```markdown
---
url: https://example.com/original-source
---
```

### External Dependencies
- **HuggingFace Inference API** — Used for summary generation and answer synthesis; requires `HF_TOKEN` env var. Not needed for index building or `--no-answer` queries.
- `sentence-transformers` — Local `all-MiniLM-L6-v2` for embeddings (~80MB, downloaded on first use)
- `chromadb` — Local persistent vector store
- `requests` — HTTP fetching
- `html2text` / `markdownify` — HTML-to-markdown conversion
- `openai` — Client library (pointed at HuggingFace endpoint)

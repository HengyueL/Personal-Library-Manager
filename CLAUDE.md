# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PersonalLibrary** is a document management and RAG (Retrieval-Augmented Generation) system for managing papers and articles of interest. The workflow is: fetch HTML/PDF → generate AI summary (saved with source URL) → build vector index → query with natural language.

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Required environment variable for summary generation
export HF_TOKEN=<your_huggingface_token>
```

## Running Tests

This project uses `pytest`. Because the venv has no `pip` symlink, invoke Python directly via `uv` or the venv binary:

```bash
# Preferred: via uv
uv run python -m pytest tests/ -q

# Alternative: direct venv binary
.venv/bin/python -m pytest tests/ -q

# Install pytest if missing (first time)
.venv/bin/python -m ensurepip && .venv/bin/pip3 install pytest
```

A **PostToolUse hook** (`.claude/settings.json`) runs `pytest` automatically after every edit to a `utils/` or `RAG/` Python file. Test results appear as a system message.

## Common Commands

```bash
# Add a new document (HTML or PDF — detected automatically)
python add_document.py --url https://example.com/article --file_name My_Article.md
python add_document.py --url https://arxiv.org/pdf/2303.08774 --file_name Paper.md

# Query the RAG system
python retrieve_document.py "your question here"
python retrieve_document.py "your question here" --top-k 3
python retrieve_document.py "your question here" --no-answer

# --- Individual steps ---

# Build / incrementally update the RAG vector index
python RAG/index.py

# Rebuild the index from scratch
python RAG/index.py --rebuild

# Query via CLI (auto-builds index on first run; requires HF_TOKEN for answer synthesis)
python RAG/query.py "your question here"
python RAG/query.py "your question here" --no-answer   # retrieval only, no LLM call
python RAG/query.py "your question here" --top-k 3
```

### Starter Scripts

- **`add_document.py`** — Accepts `--url` and `--file_name` CLI args; auto-detects HTML vs PDF, fetches and converts the document in memory, generates a summary via LLM, saves it to `doc_summary/` with YAML frontmatter containing the source URL, and incrementally updates the RAG index in one shot.
- **`retrieve_document.py`** — Accepts a positional query string plus `--top-k` and `--no-answer` flags; queries the RAG system and logs ranked sources + synthesized answer.

## Architecture

### Directory Structure
- `doc_summary/` — AI-generated summaries with YAML frontmatter containing the source URL
- `utils/` — Standalone utility scripts (fetch HTML/PDF, summarize)
- `RAG/` — RAG system implementation
- `tests/` — pytest test suite (`tests/RAG/`, `tests/utils/`)
- `RAG/chroma_db/` — Persistent Chroma vector store (gitignored, auto-created)
- `RAG/models/` — Project-local embedding model cache (gitignored, downloaded on first use)

### Full Pipeline

```
fetch_document(url) → markdown string (in memory)
                        ↓
        generate_summary(content) + save_summary(file_name, summary, url)
                        ↓
                doc_summary/{file_name}  (with ---\nurl: ...\n--- frontmatter)
                        ↓
                RAG/index.py → RAG/chroma_db/
                        ↓
                RAG/query.py → ranked sources + synthesized answer
```

### Data Flow Details

1. **`utils/fetch_document.py`** — `fetch_document(url) -> str`: auto-detects HTML vs PDF (URL extension then `Content-Type` HEAD request). HTML converted with `html2text`; PDFs downloaded to a temp file and converted with `markitdown`. Returns raw markdown string without frontmatter; does not write to disk.

2. **`utils/generate_summary.py`** — `generate_summary(content: str) -> str`: calls HuggingFace Inference API (OpenAI-compatible client, model `openai/gpt-oss-120b`) to produce a structured summary. `save_summary(file_name, summary_text, url)`: writes to `doc_summary/<file_name>` with YAML frontmatter (`url: <source>`). Skips if file already exists.

3. **`RAG/index.py`** — Reads `doc_summary/*.md`. Extracts the source URL from each file's YAML frontmatter via `strip_frontmatter()`. Embeds the summary as a single vector and upserts into Chroma. Idempotent — skips already-indexed files.

4. **`RAG/query.py`** — Embeds the query, retrieves top-k summaries from Chroma, deduplicates per source file, then calls the HF LLM to synthesize a cited answer. Returns `{query, sources, answer}`. Importable as `from RAG import query`.

### RAG Module Layout

| File | Role |
|---|---|
| `RAG/config.py` | Constants: paths, model IDs, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`, `TOP_K_CHUNKS=8`, `TOP_K_DOCS=5` |
| `RAG/chunking.py` | `strip_frontmatter(text)` + `chunk_text(text, size, overlap)` |
| `RAG/embedder.py` | Lazy-load `BAAI/bge-small-en-v1.5` singleton + `embed(texts)` |
| `RAG/index.py` | `build_index(rebuild=False)` — Chroma upsert pipeline |
| `RAG/retriever.py` | `retrieve(query_text)` — Chroma search + per-file dedup |
| `RAG/synthesizer.py` | `synthesize_answer(query, chunks)` — LLM call with citation prompt |
| `RAG/query.py` | `query(user_query, top_k_docs, synthesize)` — orchestrator + CLI |

### Indexing Scheme

- Summary IDs: `summary::{file_name}`, metadata `content_type="summary"`
- Chroma collection uses cosine distance (`hnsw:space=cosine`); score = `1 - distance` (0–1, higher is better)
- Summaries are used directly as LLM synthesis context

### Document Format

Every file in `doc_summary/` begins with:
```markdown
---
url: https://example.com/original-source
---
```

### External Dependencies
- **HuggingFace Inference API** — Used for summary generation and answer synthesis; requires `HF_TOKEN` env var. Not needed for index building or `--no-answer` queries.
- `sentence-transformers` — Local `BAAI/bge-small-en-v1.5` for embeddings (~130MB, downloaded on first use)
- `chromadb` — Local persistent vector store
- `requests` — HTTP fetching
- `html2text` / `markdownify` — HTML-to-markdown conversion
- `markitdown[pdf]` — PDF-to-markdown conversion (used by `fetch_document.py` for PDF URLs)
- `openai` — Client library (pointed at HuggingFace endpoint)

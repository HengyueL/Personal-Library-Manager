# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PersonalLibrary** is a document management and RAG (Retrieval-Augmented Generation) system for managing papers and articles of interest. The workflow is: fetch HTML/PDF → generate AI summary (saved with source URL) → build vector index → query with natural language.

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package and all dependencies (also registers the `plib` CLI)
pip install -e .

# Required env var when BACKEND = "huggingface" (default)
export HF_TOKEN=<your_huggingface_token>

# Required env var when BACKEND = "ollama"
export OLLAMA_API_KEY=<your_ollama_api_key>
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

After `pip install -e .`, use the `plib` CLI:

```bash
# Add a new document (HTML or PDF — detected automatically)
plib add --url https://example.com/article --name My_Article.md
plib add --url https://arxiv.org/pdf/2303.08774 --name Paper.md
plib add --url https://example.com/article          # LLM proposes filename from content

# Query the RAG system
plib query --query "your question here"
plib query --query "your question here" --top-k 3
plib query --query "your question here" --retrieval-only

# Rebuild the RAG index from scratch (incremental = skip already-indexed docs)
plib rebuild
plib rebuild --incremental

# Launch the web GUI
plib gui
plib gui --port 8080 --share

# --- Individual steps (direct Python, no install needed) ---

# Build / incrementally update the RAG vector index
python RAG/index.py

# Rebuild the index from scratch
python RAG/index.py --rebuild

# Query via CLI (requires HF_TOKEN for answer synthesis)
python RAG/query.py "your question here"
python RAG/query.py "your question here" --no-answer   # retrieval only, no LLM call
python RAG/query.py "your question here" --top-k 3
```

### Starter Scripts

Located in `quick_start/` — run from the repo root (they add the root to `sys.path` automatically):

- **`quick_start/add_document.py`** — Accepts `--url` and optional `--name` CLI args; auto-detects HTML vs PDF, fetches and converts the document in memory, generates a summary via LLM, saves it to `doc_summary/` with YAML frontmatter containing the source URL, and incrementally updates the RAG index in one shot. If `--name` is omitted, `generate_summary_with_filename` is called instead — a single LLM call that returns both the summary and a proposed `Source-Title.md` filename.
- **`quick_start/retrieve_document.py`** — Accepts `--query` plus `--top-k` and `--retrieval-only` flags; queries the RAG system and logs ranked sources + synthesized answer.
- **`quick_start/rebuild_knowledge_base.py`** — Rebuilds the RAG index from scratch.
- **`quick_start/cli.py`** — Unified `plib` CLI dispatcher (registered as an entry point via `pyproject.toml`).
- **`quick_start/gui.py`** — Gradio web UI with four tabs: Add Document, Find Document, View Document, Rebuild Index. The Add Document tab accepts an optional cookies path for auth-gated URLs. Streaming log output is handled by helpers in `quick_start/gui_utils/` (`streaming.py` for the thread+queue mechanism, `documents.py` for listing/reading `doc_summary/`). The View Document tab lists all files in `doc_summary/`, renders the selected document as formatted markdown, and displays its source URL; the dropdown defaults to empty (no document pre-selected). In the Find Document tab, clicking a filename in the "Document" column switches directly to the View Document tab. Ctrl+C triggers a graceful shutdown via `app.close()`. The embedding model (`BAAI/bge-small-en-v1.5`) is pre-loaded eagerly in `build_app()` via `RAG.embedder.get_model()` so the first search is not slow.

## Architecture

### Directory Structure
- `doc_summary/` — AI-generated summaries with YAML frontmatter containing the source URL
- `quick_start/` — End-user scripts: `add_document.py`, `retrieve_document.py`, `rebuild_knowledge_base.py`, `cli.py` (plib entry point), `gui.py` (Gradio UI)
- `utils/` — Standalone utility scripts (fetch HTML/PDF, summarize)
- `RAG/` — RAG system implementation
- `tests/` — pytest test suite (`tests/RAG/`, `tests/utils/`)
- `RAG/chroma_db/` — Persistent Chroma vector store (gitignored, auto-created)
- `RAG/models/` — Project-local embedding model cache (gitignored, downloaded on first use)

### Full Pipeline

```
fetch_document(url) → markdown string (in memory)
                        ↓
        with --name:  generate_summary(content)            → summary
        without --name: generate_summary_with_filename(content, url) → summary + file_name
                        ↓
        save_summary(file_name, summary, url)
                        ↓
                doc_summary/{file_name}  (with ---\nurl: ...\n--- frontmatter)
                        ↓
                RAG/index.py → RAG/chroma_db/
                        ↓
                RAG/query.py → ranked sources + synthesized answer
```

### Data Flow Details

1. **`utils/fetch_document.py`** — `fetch_document(url, cookies_path=None) -> str`: auto-detects HTML vs PDF (URL extension then `Content-Type` HEAD request). HTML converted with `html2text`; PDFs downloaded to a temp file and converted with `markitdown`. Returns raw markdown string without frontmatter; does not write to disk. Raises `AuthRequiredError` if a 401/403 is returned; caller should retry with `cookies_path` pointing to a Netscape cookies.txt file (or rely on auto browser-cookie detection via `browser-cookie3`).

2. **`utils/generate_summary.py`** — Two generation functions that delegate to `utils/llm_client.py`:
   - `generate_summary(content: str) -> str`: single-purpose summary call; used when the filename is already known.
   - `generate_summary_with_filename(content: str, url: str) -> tuple[str, str]`: single LLM call that returns `(summary, filename)` via a structured prompt; used when `--name` is omitted. Falls back to `Source-Document.md` if the LLM omits the filename line.
   - `save_summary(file_name, summary_text, url)`: writes to `doc_summary/<file_name>` with YAML frontmatter (`url: <source>`). Skips if file already exists.

3. **`RAG/index.py`** — Reads `doc_summary/*.md`. Extracts the source URL from each file's YAML frontmatter via `strip_frontmatter()`. Embeds the summary as a single vector and upserts into Chroma. Idempotent — skips already-indexed files.

4. **`utils/llm_client.py`** — Unified LLM adapter. `complete(messages, max_tokens, temperature) -> str` routes to the backend set by `BACKEND` in `RAG/config.py` (`"ollama"` or `"huggingface"`). Lazy-initializes the appropriate client singleton on first call. To switch backends, edit `BACKEND` in `RAG/config.py`.

5. **`RAG/query.py`** — Embeds the query, retrieves top-k summaries from Chroma, deduplicates per source file, then synthesizes a cited answer via `utils/llm_client.py`. Returns `{query, sources, answer}`. Importable as `from RAG import query`.

### RAG Module Layout

| File | Role |
|---|---|
| `RAG/config.py` | Constants: paths, model IDs, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`, `TOP_K_CHUNKS=8`, `TOP_K_DOCS=5`, `BACKEND` (active LLM backend) |
| `RAG/chunking.py` | `strip_frontmatter(text)` + `chunk_text(text, size, overlap)` |
| `RAG/embedder.py` | `BAAI/bge-small-en-v1.5` singleton + `embed(texts)`; pre-loaded at GUI startup via `get_model()` |
| `RAG/index.py` | `build_index(rebuild=False)` — Chroma upsert pipeline |
| `RAG/retriever.py` | `retrieve(query_text)` — Chroma search + per-file dedup |
| `RAG/synthesizer.py` | `synthesize_answer(query, chunks)` — delegates to `utils/llm_client.complete` with citation prompt |
| `utils/llm_client.py` | `complete(messages, max_tokens, temperature)` — backend adapter; reads `BACKEND` from `RAG/config.py` |
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

### LLM Backend Configuration

Backend is selected by the `BACKEND` constant in `RAG/config.py` (edit the file to switch):

| `BACKEND` value | Required env var | Endpoint |
|---|---|---|
| `"huggingface"` | `HF_TOKEN` | `https://router.huggingface.co/v1` (OpenAI-compatible) |
| `"ollama"` | `OLLAMA_API_KEY` | `https://ollama.com` (default, overridable via `OLLAMA_HOST`) |

Model IDs are overridable via `LLM_MODEL_ID` (HF) and `OLLAMA_MODEL_ID` env vars. Neither backend is needed for index building or `--retrieval-only` queries.

### External Dependencies
- `sentence-transformers` — Local `BAAI/bge-small-en-v1.5` for embeddings (~130MB, downloaded on first use)
- `chromadb` — Local persistent vector store
- `requests` — HTTP fetching
- `html2text` / `markdownify` — HTML-to-markdown conversion
- `markitdown[pdf]` — PDF-to-markdown conversion (used by `fetch_document.py` for PDF URLs)
- `openai` — Client library used for HuggingFace backend
- `ollama` — Client library used for Ollama backend
- `browser-cookie3` — Auto-reads browser cookies for auth-gated URLs when no cookies file is provided

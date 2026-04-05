# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PersonalLibrary** is a document management and RAG (Retrieval-Augmented Generation) system for managing papers and articles. The workflow is: fetch HTML → convert to markdown → generate AI summary → update metadata index → (future) RAG query interface.

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
```

## Architecture

### Directory Structure
- `doc_raw/` — Raw documents as markdown (with YAML frontmatter containing source URL)
- `doc_summary/` — AI-generated summaries, parallel to `doc_raw/` (same filenames)
- `utils/` — Standalone utility scripts
- `RAG/` — Retrieval system (placeholder, not yet implemented)
- `doc_relation_table.csv` — CSV metadata index: `index`, `file_name`, `orignal_url` (note: typo in column name is intentional/existing)

### Data Flow

1. **`utils/fetch_html.py`** — Downloads HTML via `requests`, converts to markdown with `html2text`, prepends YAML frontmatter (`url: <source>`), saves to `doc_raw/<file_name>.md`. Refuses to overwrite existing files.

2. **`utils/generate_summary.py`** — Reads from `doc_raw/`, calls HuggingFace Inference API (OpenAI-compatible client, model `openai/gpt-oss-120b`) to produce structured summaries, saves to `doc_summary/<file_name>.md`. Skips already-summarized documents.

3. **`utils/update_relation_table.py`** — Scans `doc_raw/`, validates that corresponding `doc_summary/` files exist, extracts source URLs from YAML frontmatter, writes/updates `doc_relation_table.csv`. Removes stale entries for deleted files.

### Document Format

Every file in `doc_raw/` begins with:
```markdown
---
url: https://example.com/original-source
---
```

### External Dependencies
- **HuggingFace Inference API** — Used for summary generation via OpenAI-compatible client; requires `HF_TOKEN` env var
- `requests` — HTTP fetching
- `html2text` / `markdownify` — HTML-to-markdown conversion
- `openai` — Client library (pointed at HuggingFace endpoint)

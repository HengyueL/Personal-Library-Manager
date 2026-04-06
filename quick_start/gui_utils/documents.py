"""Document-listing helpers for the Gradio GUI."""

from pathlib import Path


def _doc_summary_dir() -> Path:
    # quick_start/gui_utils/ -> quick_start/ -> repo root -> doc_summary/
    return Path(__file__).resolve().parent.parent.parent / "doc_summary"


def list_documents() -> list[str]:
    return sorted(p.name for p in _doc_summary_dir().glob("*.md"))

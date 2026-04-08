"""Document-listing helpers for the Gradio GUI."""

from pathlib import Path


def _doc_summary_dir() -> Path:
    # quick_start/gui_utils/ -> quick_start/ -> repo root -> doc_summary/
    return Path(__file__).resolve().parent.parent.parent / "doc_summary"


def list_documents() -> list[str]:
    return sorted(p.name for p in _doc_summary_dir().glob("*.md"))


def delete_document(file_name: str) -> None:
    """Delete the summary file and remove it from the Chroma index."""
    from RAG.index import remove_from_index
    path = _doc_summary_dir() / file_name
    if path.exists():
        path.unlink()
        print(f"Deleted file: {file_name}")
    remove_from_index(file_name)
    print(f"Done. '{file_name}' removed from library.")

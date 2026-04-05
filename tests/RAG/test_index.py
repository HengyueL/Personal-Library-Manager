from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _write_md(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# get_indexed_ids
# ---------------------------------------------------------------------------

class TestGetIndexedIds:
    def test_returns_set_of_ids(self):
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": ["summary::doc.md", "summary::other.md"]}
        from RAG.index import get_indexed_ids
        ids = get_indexed_ids(mock_col)
        assert ids == {"summary::doc.md", "summary::other.md"}

    def test_empty_collection_returns_empty_set(self):
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": []}
        from RAG.index import get_indexed_ids
        assert get_indexed_ids(mock_col) == set()


# ---------------------------------------------------------------------------
# index_summary_doc
# ---------------------------------------------------------------------------

class TestIndexSummaryDoc:
    def test_upserts_single_vector_with_summary_id(self, tmp_path, monkeypatch):
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        _write_md(doc_summary / "doc.md", "---\nurl: https://x.com\n---\n\nThis is the summary text.")
        monkeypatch.setattr("RAG.index.DOC_SUMMARY_PATH", doc_summary)

        mock_col = MagicMock()
        with patch("RAG.index.embedder.embed", return_value=[[0.1, 0.2]]):
            from RAG.index import index_summary_doc
            index_summary_doc("doc.md", mock_col)

        call_kwargs = mock_col.upsert.call_args[1]
        assert call_kwargs["ids"] == ["summary::doc.md"]
        assert call_kwargs["metadatas"][0]["content_type"] == "summary"
        assert call_kwargs["metadatas"][0]["source"] == "doc.md"

    def test_extracts_url_from_frontmatter(self, tmp_path, monkeypatch):
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        _write_md(doc_summary / "doc.md", "---\nurl: https://example.com/article\n---\n\nSummary body.")
        monkeypatch.setattr("RAG.index.DOC_SUMMARY_PATH", doc_summary)

        mock_col = MagicMock()
        with patch("RAG.index.embedder.embed", return_value=[[0.1]]):
            from RAG.index import index_summary_doc
            index_summary_doc("doc.md", mock_col)

        call_kwargs = mock_col.upsert.call_args[1]
        assert call_kwargs["metadatas"][0]["original_url"] == "https://example.com/article"

    def test_body_without_frontmatter_used_for_embedding(self, tmp_path, monkeypatch):
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        _write_md(doc_summary / "doc.md", "---\nurl: https://x.com\n---\n\nSummary content here.")
        monkeypatch.setattr("RAG.index.DOC_SUMMARY_PATH", doc_summary)

        mock_col = MagicMock()
        embedded_texts = []

        def capture_embed(texts):
            embedded_texts.extend(texts)
            return [[0.1]]

        with patch("RAG.index.embedder.embed", side_effect=capture_embed):
            from RAG.index import index_summary_doc
            index_summary_doc("doc.md", mock_col)

        assert any("Summary content here." in t for t in embedded_texts)
        assert not any("url:" in t for t in embedded_texts)


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_skips_already_indexed_file(self, tmp_path, monkeypatch, capsys):
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        _write_md(doc_summary / "doc.md", "---\nurl: https://x.com\n---\n\nSummary.")
        monkeypatch.setattr("RAG.index.DOC_SUMMARY_PATH", doc_summary)

        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": ["summary::doc.md"]}
        mock_col.count.return_value = 1

        with patch("RAG.index.get_collection", return_value=mock_col):
            from RAG.index import build_index
            build_index(rebuild=False)

        mock_col.upsert.assert_not_called()
        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()

    def test_indexes_new_file(self, tmp_path, monkeypatch, capsys):
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        _write_md(doc_summary / "doc.md", "---\nurl: https://x.com\n---\n\nSummary.")
        monkeypatch.setattr("RAG.index.DOC_SUMMARY_PATH", doc_summary)

        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": []}
        mock_col.count.return_value = 1

        with patch("RAG.index.get_collection", return_value=mock_col):
            with patch("RAG.index.embedder.embed", return_value=[[0.1]]):
                from RAG.index import build_index
                build_index(rebuild=False)

        mock_col.upsert.assert_called_once()

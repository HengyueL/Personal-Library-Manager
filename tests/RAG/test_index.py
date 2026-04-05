import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "file_name", "orignal_url"])
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# load_url_map
# ---------------------------------------------------------------------------

class TestLoadUrlMap:
    def test_returns_empty_dict_when_no_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr("RAG.index.RELATION_TABLE_PATH", tmp_path / "missing.csv")
        from RAG.index import load_url_map
        assert load_url_map() == {}

    def test_parses_csv_correctly(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "table.csv"
        _write_csv(csv_path, [
            {"index": 1, "file_name": "doc.md", "orignal_url": "https://example.com"},
        ])
        monkeypatch.setattr("RAG.index.RELATION_TABLE_PATH", csv_path)
        from RAG.index import load_url_map
        url_map = load_url_map()
        assert url_map["doc.md"] == "https://example.com"

    def test_multiple_entries(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "table.csv"
        _write_csv(csv_path, [
            {"index": 1, "file_name": "a.md", "orignal_url": "https://a.com"},
            {"index": 2, "file_name": "b.md", "orignal_url": "https://b.com"},
        ])
        monkeypatch.setattr("RAG.index.RELATION_TABLE_PATH", csv_path)
        from RAG.index import load_url_map
        url_map = load_url_map()
        assert len(url_map) == 2
        assert url_map["b.md"] == "https://b.com"


# ---------------------------------------------------------------------------
# get_indexed_ids
# ---------------------------------------------------------------------------

class TestGetIndexedIds:
    def test_returns_set_of_ids(self):
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": ["raw::doc.md::0", "summary::doc.md"]}
        from RAG.index import get_indexed_ids
        ids = get_indexed_ids(mock_col)
        assert ids == {"raw::doc.md::0", "summary::doc.md"}

    def test_empty_collection_returns_empty_set(self):
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": []}
        from RAG.index import get_indexed_ids
        assert get_indexed_ids(mock_col) == set()


# ---------------------------------------------------------------------------
# index_raw_doc
# ---------------------------------------------------------------------------

class TestIndexRawDoc:
    def test_indexes_chunks_and_returns_count(self, tmp_path, monkeypatch):
        doc_raw = tmp_path / "doc_raw"
        doc_raw.mkdir()
        _write_md(doc_raw / "doc.md", "---\nurl: https://x.com\n---\n\nParagraph one.\n\nParagraph two.")
        monkeypatch.setattr("RAG.index.DOC_RAW_PATH", doc_raw)

        mock_col = MagicMock()
        fake_vectors = [[0.1, 0.2]]

        with patch("RAG.index.embedder.embed", return_value=fake_vectors):
            from RAG.index import index_raw_doc
            n = index_raw_doc("doc.md", "https://x.com", mock_col)

        assert n > 0
        mock_col.upsert.assert_called_once()

    def test_upserted_ids_follow_naming_scheme(self, tmp_path, monkeypatch):
        doc_raw = tmp_path / "doc_raw"
        doc_raw.mkdir()
        _write_md(doc_raw / "doc.md", "---\nurl: https://x.com\n---\n\nSome content here.")
        monkeypatch.setattr("RAG.index.DOC_RAW_PATH", doc_raw)

        mock_col = MagicMock()
        with patch("RAG.index.embedder.embed", return_value=[[0.1]]):
            from RAG.index import index_raw_doc
            index_raw_doc("doc.md", "https://x.com", mock_col)

        call_kwargs = mock_col.upsert.call_args[1]
        for doc_id in call_kwargs["ids"]:
            assert doc_id.startswith("raw::doc.md::")

    def test_empty_body_returns_zero_and_skips_upsert(self, tmp_path, monkeypatch):
        doc_raw = tmp_path / "doc_raw"
        doc_raw.mkdir()
        # frontmatter only, no body
        _write_md(doc_raw / "empty.md", "---\nurl: https://x.com\n---\n\n")
        monkeypatch.setattr("RAG.index.DOC_RAW_PATH", doc_raw)

        mock_col = MagicMock()
        with patch("RAG.index.embedder.embed", return_value=[]):
            from RAG.index import index_raw_doc
            n = index_raw_doc("empty.md", "https://x.com", mock_col)

        assert n == 0
        mock_col.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# index_summary_doc
# ---------------------------------------------------------------------------

class TestIndexSummaryDoc:
    def test_upserts_single_vector_with_summary_id(self, tmp_path, monkeypatch):
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        _write_md(doc_summary / "doc.md", "This is the summary text.")
        monkeypatch.setattr("RAG.index.DOC_SUMMARY_PATH", doc_summary)

        mock_col = MagicMock()
        with patch("RAG.index.embedder.embed", return_value=[[0.1, 0.2]]):
            from RAG.index import index_summary_doc
            index_summary_doc("doc.md", "https://x.com", mock_col)

        call_kwargs = mock_col.upsert.call_args[1]
        assert call_kwargs["ids"] == ["summary::doc.md"]
        assert call_kwargs["metadatas"][0]["content_type"] == "summary"
        assert call_kwargs["metadatas"][0]["source"] == "doc.md"

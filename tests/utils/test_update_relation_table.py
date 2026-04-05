import csv
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_md(path: Path, url: str = "https://example.com"):
    path.write_text(f"---\nurl: {url}\n---\n\nContent.", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "file_name", "orignal_url"])
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# extract_url_from_yaml_frontmatter
# ---------------------------------------------------------------------------

class TestExtractUrl:
    def test_extracts_url_from_frontmatter(self, tmp_path):
        doc = tmp_path / "doc.md"
        _write_md(doc, url="https://example.com/article")
        from utils.update_relation_table import extract_url_from_yaml_frontmatter
        assert extract_url_from_yaml_frontmatter(doc) == "https://example.com/article"

    def test_returns_none_when_no_frontmatter(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text("No frontmatter here.", encoding="utf-8")
        from utils.update_relation_table import extract_url_from_yaml_frontmatter
        assert extract_url_from_yaml_frontmatter(doc) is None

    def test_returns_none_when_url_key_missing(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text("---\ntitle: something\n---\n\nBody.", encoding="utf-8")
        from utils.update_relation_table import extract_url_from_yaml_frontmatter
        assert extract_url_from_yaml_frontmatter(doc) is None

    def test_strips_whitespace_from_url(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text("---\nurl:   https://example.com   \n---\n\nBody.", encoding="utf-8")
        from utils.update_relation_table import extract_url_from_yaml_frontmatter
        result = extract_url_from_yaml_frontmatter(doc)
        assert result == "https://example.com"


# ---------------------------------------------------------------------------
# read_existing_relation_table
# ---------------------------------------------------------------------------

class TestReadExistingRelationTable:
    def test_returns_empty_dict_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.update_relation_table.RELATION_TABLE_PATH", tmp_path / "missing.csv")
        from utils.update_relation_table import read_existing_relation_table
        assert read_existing_relation_table() == {}

    def test_parses_entries_keyed_by_file_name(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "table.csv"
        _write_csv(csv_path, [{"index": 1, "file_name": "doc.md", "orignal_url": "https://x.com"}])
        monkeypatch.setattr("utils.update_relation_table.RELATION_TABLE_PATH", csv_path)
        from utils.update_relation_table import read_existing_relation_table
        result = read_existing_relation_table()
        assert "doc.md" in result
        assert result["doc.md"]["orignal_url"] == "https://x.com"

    def test_multiple_entries_all_parsed(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "table.csv"
        _write_csv(csv_path, [
            {"index": 1, "file_name": "a.md", "orignal_url": "https://a.com"},
            {"index": 2, "file_name": "b.md", "orignal_url": "https://b.com"},
        ])
        monkeypatch.setattr("utils.update_relation_table.RELATION_TABLE_PATH", csv_path)
        from utils.update_relation_table import read_existing_relation_table
        result = read_existing_relation_table()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# validate_existing_entries
# ---------------------------------------------------------------------------

class TestValidateExistingEntries:
    def test_valid_entry_kept_when_both_files_exist(self, tmp_path, monkeypatch):
        doc_raw = tmp_path / "doc_raw"
        doc_raw.mkdir()
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        (doc_raw / "doc.md").write_text("raw", encoding="utf-8")
        (doc_summary / "doc.md").write_text("summary", encoding="utf-8")

        monkeypatch.setattr("utils.update_relation_table.DOC_RAW_PATH", doc_raw)
        monkeypatch.setattr("utils.update_relation_table.DOC_SUMMARY_PATH", doc_summary)

        entries = {"doc.md": {"index": "1", "file_name": "doc.md", "orignal_url": "https://x.com"}}
        from utils.update_relation_table import validate_existing_entries
        valid, removed = validate_existing_entries(entries)
        assert "doc.md" in valid
        assert removed == []

    def test_entry_removed_when_raw_missing(self, tmp_path, monkeypatch):
        doc_raw = tmp_path / "doc_raw"
        doc_raw.mkdir()
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        (doc_summary / "doc.md").write_text("summary", encoding="utf-8")
        # raw file intentionally absent

        monkeypatch.setattr("utils.update_relation_table.DOC_RAW_PATH", doc_raw)
        monkeypatch.setattr("utils.update_relation_table.DOC_SUMMARY_PATH", doc_summary)

        entries = {"doc.md": {"index": "1", "file_name": "doc.md", "orignal_url": ""}}
        from utils.update_relation_table import validate_existing_entries
        valid, removed = validate_existing_entries(entries)
        assert "doc.md" not in valid
        assert "doc.md" in removed

    def test_entry_removed_when_summary_missing(self, tmp_path, monkeypatch):
        doc_raw = tmp_path / "doc_raw"
        doc_raw.mkdir()
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        (doc_raw / "doc.md").write_text("raw", encoding="utf-8")
        # summary file intentionally absent

        monkeypatch.setattr("utils.update_relation_table.DOC_RAW_PATH", doc_raw)
        monkeypatch.setattr("utils.update_relation_table.DOC_SUMMARY_PATH", doc_summary)

        entries = {"doc.md": {"index": "1", "file_name": "doc.md", "orignal_url": ""}}
        from utils.update_relation_table import validate_existing_entries
        valid, removed = validate_existing_entries(entries)
        assert "doc.md" not in valid
        assert "doc.md" in removed


# ---------------------------------------------------------------------------
# update_relation_table (integration)
# ---------------------------------------------------------------------------

class TestUpdateRelationTable:
    def _setup_dirs(self, tmp_path, docs: list[tuple[str, str]]):
        """Create doc_raw + doc_summary with matching files. docs = [(file_name, url), ...]"""
        doc_raw = tmp_path / "doc_raw"
        doc_raw.mkdir()
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        for file_name, url in docs:
            _write_md(doc_raw / file_name, url=url)
            (doc_summary / file_name).write_text("summary", encoding="utf-8")
        return doc_raw, doc_summary

    def test_creates_csv_with_correct_entries(self, tmp_path, monkeypatch):
        doc_raw, doc_summary = self._setup_dirs(tmp_path, [("a.md", "https://a.com")])
        csv_path = tmp_path / "table.csv"
        monkeypatch.setattr("utils.update_relation_table.DOC_RAW_PATH", doc_raw)
        monkeypatch.setattr("utils.update_relation_table.DOC_SUMMARY_PATH", doc_summary)
        monkeypatch.setattr("utils.update_relation_table.RELATION_TABLE_PATH", csv_path)

        from utils.update_relation_table import update_relation_table
        update_relation_table()

        rows = _read_csv(csv_path)
        assert len(rows) == 1
        assert rows[0]["file_name"] == "a.md"
        assert rows[0]["orignal_url"] == "https://a.com"

    def test_indices_are_sequential(self, tmp_path, monkeypatch):
        doc_raw, doc_summary = self._setup_dirs(tmp_path, [
            ("a.md", "https://a.com"),
            ("b.md", "https://b.com"),
            ("c.md", "https://c.com"),
        ])
        csv_path = tmp_path / "table.csv"
        monkeypatch.setattr("utils.update_relation_table.DOC_RAW_PATH", doc_raw)
        monkeypatch.setattr("utils.update_relation_table.DOC_SUMMARY_PATH", doc_summary)
        monkeypatch.setattr("utils.update_relation_table.RELATION_TABLE_PATH", csv_path)

        from utils.update_relation_table import update_relation_table
        update_relation_table()

        rows = _read_csv(csv_path)
        indices = [int(r["index"]) for r in rows]
        assert sorted(indices) == list(range(1, len(rows) + 1))

    def test_stale_entries_removed(self, tmp_path, monkeypatch):
        doc_raw, doc_summary = self._setup_dirs(tmp_path, [("a.md", "https://a.com")])
        csv_path = tmp_path / "table.csv"
        # Pre-populate CSV with a stale entry (ghost.md doesn't exist on disk)
        _write_csv(csv_path, [
            {"index": 1, "file_name": "ghost.md", "orignal_url": "https://ghost.com"},
            {"index": 2, "file_name": "a.md", "orignal_url": "https://a.com"},
        ])
        monkeypatch.setattr("utils.update_relation_table.DOC_RAW_PATH", doc_raw)
        monkeypatch.setattr("utils.update_relation_table.DOC_SUMMARY_PATH", doc_summary)
        monkeypatch.setattr("utils.update_relation_table.RELATION_TABLE_PATH", csv_path)

        from utils.update_relation_table import update_relation_table
        update_relation_table()

        rows = _read_csv(csv_path)
        file_names = [r["file_name"] for r in rows]
        assert "ghost.md" not in file_names
        assert "a.md" in file_names

    def test_skips_doc_raw_file_without_matching_summary(self, tmp_path, monkeypatch):
        doc_raw = tmp_path / "doc_raw"
        doc_raw.mkdir()
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        _write_md(doc_raw / "no_summary.md", url="https://x.com")
        # no_summary.md intentionally has no matching summary
        csv_path = tmp_path / "table.csv"
        monkeypatch.setattr("utils.update_relation_table.DOC_RAW_PATH", doc_raw)
        monkeypatch.setattr("utils.update_relation_table.DOC_SUMMARY_PATH", doc_summary)
        monkeypatch.setattr("utils.update_relation_table.RELATION_TABLE_PATH", csv_path)

        from utils.update_relation_table import update_relation_table
        update_relation_table()

        # CSV should not be written (no valid docs), or written with 0 entries
        if csv_path.exists():
            rows = _read_csv(csv_path)
            assert all(r["file_name"] != "no_summary.md" for r in rows)

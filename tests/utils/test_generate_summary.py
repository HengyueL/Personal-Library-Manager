from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_client(answer: str) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = answer
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


class TestGenerateSummary:
    def test_returns_llm_response(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text("# Title\n\nSome content.", encoding="utf-8")
        mock_client = _make_mock_client("This is a summary.")

        with patch("utils.generate_summary._client", mock_client):
            from utils.generate_summary import generate_summary
            result = generate_summary(doc)

        assert result == "This is a summary."

    def test_prompt_includes_document_content(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text("Unique document content XYZ123.", encoding="utf-8")
        mock_client = _make_mock_client("summary")

        with patch("utils.generate_summary._client", mock_client):
            from utils.generate_summary import generate_summary
            generate_summary(doc)

        prompt = mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]
        assert "Unique document content XYZ123." in prompt

    def test_llm_called_with_correct_model_params(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text("content", encoding="utf-8")
        mock_client = _make_mock_client("summary")

        with patch("utils.generate_summary._client", mock_client):
            from utils.generate_summary import generate_summary
            generate_summary(doc)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000


class TestProcessDocument:
    def test_skips_if_doc_not_found(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("utils.generate_summary.DOC_RAW_PATH", tmp_path / "doc_raw")
        monkeypatch.setattr("utils.generate_summary.DOC_SUMMARY_PATH", tmp_path / "doc_summary")

        from utils.generate_summary import process_document
        process_document("nonexistent.md")

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "Error" in captured.out

    def test_skips_if_summary_already_exists(self, tmp_path, monkeypatch, capsys):
        doc_raw = tmp_path / "doc_raw"
        doc_raw.mkdir()
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        (doc_raw / "doc.md").write_text("content", encoding="utf-8")
        (doc_summary / "doc.md").write_text("existing summary", encoding="utf-8")

        monkeypatch.setattr("utils.generate_summary.DOC_RAW_PATH", doc_raw)
        monkeypatch.setattr("utils.generate_summary.DOC_SUMMARY_PATH", doc_summary)

        mock_client = _make_mock_client("new summary")
        with patch("utils.generate_summary._client", mock_client):
            from utils.generate_summary import process_document
            process_document("doc.md")

        mock_client.chat.completions.create.assert_not_called()

    def test_generates_and_saves_summary(self, tmp_path, monkeypatch):
        doc_raw = tmp_path / "doc_raw"
        doc_raw.mkdir()
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        (doc_raw / "doc.md").write_text("Document content.", encoding="utf-8")

        monkeypatch.setattr("utils.generate_summary.DOC_RAW_PATH", doc_raw)
        monkeypatch.setattr("utils.generate_summary.DOC_SUMMARY_PATH", doc_summary)

        mock_client = _make_mock_client("Generated summary text.")
        with patch("utils.generate_summary._client", mock_client):
            from utils.generate_summary import process_document
            process_document("doc.md")

        summary_file = doc_summary / "doc.md"
        assert summary_file.exists()
        content = summary_file.read_text(encoding="utf-8")
        assert "Generated summary text." in content

    def test_summary_file_has_header(self, tmp_path, monkeypatch):
        doc_raw = tmp_path / "doc_raw"
        doc_raw.mkdir()
        doc_summary = tmp_path / "doc_summary"
        doc_summary.mkdir()
        (doc_raw / "my_doc.md").write_text("Content.", encoding="utf-8")

        monkeypatch.setattr("utils.generate_summary.DOC_RAW_PATH", doc_raw)
        monkeypatch.setattr("utils.generate_summary.DOC_SUMMARY_PATH", doc_summary)

        mock_client = _make_mock_client("summary body")
        with patch("utils.generate_summary._client", mock_client):
            from utils.generate_summary import process_document
            process_document("my_doc.md")

        content = (doc_summary / "my_doc.md").read_text(encoding="utf-8")
        assert "# Summary of my_doc.md" in content

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
    def test_returns_llm_response(self):
        mock_client = _make_mock_client("This is a summary.")

        with patch("utils.generate_summary._client", mock_client):
            from utils.generate_summary import generate_summary
            result = generate_summary("# Title\n\nSome content.")

        assert result == "This is a summary."

    def test_prompt_includes_document_content(self):
        mock_client = _make_mock_client("summary")

        with patch("utils.generate_summary._client", mock_client):
            from utils.generate_summary import generate_summary
            generate_summary("Unique document content XYZ123.")

        prompt = mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]
        assert "Unique document content XYZ123." in prompt

    def test_llm_called_with_correct_model_params(self):
        mock_client = _make_mock_client("summary")

        with patch("utils.generate_summary._client", mock_client):
            from utils.generate_summary import generate_summary
            generate_summary("content")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000


class TestSaveSummary:
    def test_saves_file_with_url_frontmatter(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.generate_summary.DOC_SUMMARY_PATH", tmp_path)

        from utils.generate_summary import save_summary
        save_summary("doc.md", "Summary text here.", "https://example.com/article")

        content = (tmp_path / "doc.md").read_text(encoding="utf-8")
        assert content.startswith("---\nurl: https://example.com/article\n---")

    def test_saves_file_with_summary_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.generate_summary.DOC_SUMMARY_PATH", tmp_path)

        from utils.generate_summary import save_summary
        save_summary("doc.md", "Summary body text.", "https://example.com")

        content = (tmp_path / "doc.md").read_text(encoding="utf-8")
        assert "Summary body text." in content

    def test_saves_file_with_header(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.generate_summary.DOC_SUMMARY_PATH", tmp_path)

        from utils.generate_summary import save_summary
        save_summary("my_doc.md", "body", "https://example.com")

        content = (tmp_path / "my_doc.md").read_text(encoding="utf-8")
        assert "# Summary of my_doc.md" in content

    def test_skips_if_summary_already_exists(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("utils.generate_summary.DOC_SUMMARY_PATH", tmp_path)
        (tmp_path / "doc.md").write_text("existing summary", encoding="utf-8")

        from utils.generate_summary import save_summary
        save_summary("doc.md", "new summary", "https://example.com")

        content = (tmp_path / "doc.md").read_text(encoding="utf-8")
        assert content == "existing summary"
        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()

    def test_creates_directory_if_missing(self, tmp_path, monkeypatch):
        summary_dir = tmp_path / "doc_summary"
        monkeypatch.setattr("utils.generate_summary.DOC_SUMMARY_PATH", summary_dir)

        from utils.generate_summary import save_summary
        save_summary("doc.md", "summary", "https://example.com")

        assert (summary_dir / "doc.md").exists()

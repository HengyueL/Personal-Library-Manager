from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGenerateSummary:
    def test_returns_llm_response(self):
        with patch("utils.generate_summary.complete", return_value="This is a summary."):
            from utils.generate_summary import generate_summary
            result = generate_summary("# Title\n\nSome content.")

        assert result == "This is a summary."

    def test_prompt_includes_document_content(self):
        mock_complete = MagicMock(return_value="summary")

        with patch("utils.generate_summary.complete", mock_complete):
            from utils.generate_summary import generate_summary
            generate_summary("Unique document content XYZ123.")

        prompt = mock_complete.call_args[1]["messages"][0]["content"]
        assert "Unique document content XYZ123." in prompt

    def test_llm_called_with_correct_model_params(self):
        mock_complete = MagicMock(return_value="summary")

        with patch("utils.generate_summary.complete", mock_complete):
            from utils.generate_summary import generate_summary
            generate_summary("content")

        call_kwargs = mock_complete.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000


class TestGenerateSummaryWithFilename:
    def test_returns_summary_and_filename(self):
        response = "FILENAME: Anthropic-Agents.md\n\nSUMMARY:\nThis paper covers agents."

        with patch("utils.generate_summary.complete", return_value=response):
            from utils.generate_summary import generate_summary_with_filename
            summary, filename = generate_summary_with_filename("content", "https://anthropic.com/agents")

        assert filename == "Anthropic-Agents.md"
        assert "This paper covers agents." in summary

    def test_filename_without_md_extension_gets_extension_added(self):
        response = "FILENAME: Anthropic-Agents\n\nSUMMARY:\nSummary text."

        with patch("utils.generate_summary.complete", return_value=response):
            from utils.generate_summary import generate_summary_with_filename
            _, filename = generate_summary_with_filename("content", "https://anthropic.com/agents")

        assert filename.endswith(".md")

    def test_fallback_filename_when_llm_omits_it(self):
        response = "Here is a summary without a filename line."

        with patch("utils.generate_summary.complete", return_value=response):
            from utils.generate_summary import generate_summary_with_filename
            summary, filename = generate_summary_with_filename("content", "https://example.com/doc")

        assert filename.endswith(".md")
        assert summary == response

    def test_url_included_in_prompt(self):
        mock_complete = MagicMock(return_value="FILENAME: X.md\n\nSUMMARY:\nText.")

        with patch("utils.generate_summary.complete", mock_complete):
            from utils.generate_summary import generate_summary_with_filename
            generate_summary_with_filename("content", "https://openai.com/research")

        prompt = mock_complete.call_args[1]["messages"][0]["content"]
        assert "https://openai.com/research" in prompt


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

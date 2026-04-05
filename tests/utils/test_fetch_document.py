from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests


# ---------------------------------------------------------------------------
# _is_pdf
# ---------------------------------------------------------------------------

class TestIsPdf:
    def test_url_ending_with_pdf_extension(self):
        from utils.fetch_document import _is_pdf
        assert _is_pdf("https://example.com/paper.pdf") is True

    def test_url_ending_with_uppercase_pdf_extension(self):
        from utils.fetch_document import _is_pdf
        assert _is_pdf("https://example.com/paper.PDF") is True

    def test_html_url_returns_false(self):
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        with patch("utils.fetch_document.requests.head", return_value=mock_resp):
            from utils.fetch_document import _is_pdf
            assert _is_pdf("https://example.com/article") is False

    def test_url_with_pdf_content_type_returns_true(self):
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/pdf"}
        with patch("utils.fetch_document.requests.head", return_value=mock_resp):
            from utils.fetch_document import _is_pdf
            assert _is_pdf("https://arxiv.org/pdf/2510.18234") is True

    def test_request_exception_returns_false(self):
        with patch("utils.fetch_document.requests.head", side_effect=requests.RequestException):
            from utils.fetch_document import _is_pdf
            assert _is_pdf("https://broken.example.com/doc") is False

    def test_missing_content_type_header_returns_false(self):
        mock_resp = MagicMock()
        mock_resp.headers = {}
        with patch("utils.fetch_document.requests.head", return_value=mock_resp):
            from utils.fetch_document import _is_pdf
            assert _is_pdf("https://example.com/no-header") is False


# ---------------------------------------------------------------------------
# _fetch_html
# ---------------------------------------------------------------------------

class TestFetchHtml:
    def test_returns_markdown_string(self):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><h1>Hello</h1></body></html>"
        with patch("utils.fetch_document.requests.get", return_value=mock_resp):
            from utils.fetch_document import _fetch_html
            result = _fetch_html("https://example.com")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_html_converted_to_markdown(self):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><h1>Title</h1><p>Body text.</p></body></html>"
        with patch("utils.fetch_document.requests.get", return_value=mock_resp):
            from utils.fetch_document import _fetch_html
            result = _fetch_html("https://example.com")
        assert "Title" in result
        assert "Body text." in result


# ---------------------------------------------------------------------------
# _fetch_pdf
# ---------------------------------------------------------------------------

class TestFetchPdf:
    def test_returns_text_content(self):
        mock_resp = MagicMock()
        mock_resp.content = b"%PDF-1.4 fake content"
        mock_resp.raise_for_status = MagicMock()

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.text_content = "Converted PDF text"
        mock_md_instance = MagicMock()
        mock_md_instance.convert.return_value = mock_markitdown_result

        with patch("utils.fetch_document.requests.get", return_value=mock_resp):
            with patch("utils.fetch_document.MarkItDown", return_value=mock_md_instance):
                from utils.fetch_document import _fetch_pdf
                result = _fetch_pdf("https://example.com/paper.pdf")

        assert result == "Converted PDF text"

    def test_temp_file_cleaned_up(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.content = b"fake pdf bytes"
        mock_resp.raise_for_status = MagicMock()

        mock_result = MagicMock()
        mock_result.text_content = "text"
        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result

        captured_tmp_paths = []

        original_ntf = __import__("tempfile").NamedTemporaryFile

        def tracking_ntf(*args, **kwargs):
            ntf = original_ntf(*args, **kwargs)
            captured_tmp_paths.append(Path(ntf.name))
            return ntf

        with patch("utils.fetch_document.requests.get", return_value=mock_resp):
            with patch("utils.fetch_document.MarkItDown", return_value=mock_md):
                with patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
                    from utils.fetch_document import _fetch_pdf
                    _fetch_pdf("https://example.com/paper.pdf")

        for p in captured_tmp_paths:
            assert not p.exists(), f"Temp file not cleaned up: {p}"


# ---------------------------------------------------------------------------
# fetch_document
# ---------------------------------------------------------------------------

class TestFetchDocument:
    def test_pdf_url_calls_fetch_pdf_and_returns_content(self):
        with patch("utils.fetch_document._is_pdf", return_value=True):
            with patch("utils.fetch_document._fetch_pdf", return_value="pdf markdown") as mock_pdf:
                with patch("utils.fetch_document._fetch_html") as mock_html:
                    from utils.fetch_document import fetch_document
                    result = fetch_document("https://example.com/paper.pdf")

        mock_pdf.assert_called_once_with("https://example.com/paper.pdf")
        mock_html.assert_not_called()
        assert result == "pdf markdown"

    def test_html_url_calls_fetch_html_and_returns_content(self):
        with patch("utils.fetch_document._is_pdf", return_value=False):
            with patch("utils.fetch_document._fetch_html", return_value="html markdown") as mock_html:
                with patch("utils.fetch_document._fetch_pdf") as mock_pdf:
                    from utils.fetch_document import fetch_document
                    result = fetch_document("https://example.com/article")

        mock_html.assert_called_once_with("https://example.com/article")
        mock_pdf.assert_not_called()
        assert result == "html markdown"

    def test_returns_string(self):
        with patch("utils.fetch_document._is_pdf", return_value=False):
            with patch("utils.fetch_document._fetch_html", return_value="some content"):
                from utils.fetch_document import fetch_document
                result = fetch_document("https://example.com/article")
        assert isinstance(result, str)

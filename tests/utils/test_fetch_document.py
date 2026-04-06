from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
import requests


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mock_session(status_code=200, url="https://example.com/page", headers=None, text="", content=b""):
    """Return a mock requests.Session whose get/head return a configured response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.url = url
    mock_resp.headers = headers or {}
    mock_resp.text = text
    mock_resp.content = content
    mock_resp.raise_for_status = MagicMock()

    session = MagicMock(spec=requests.Session)
    session.head.return_value = mock_resp
    session.get.return_value = mock_resp
    return session, mock_resp


# ---------------------------------------------------------------------------
# _is_pdf
# ---------------------------------------------------------------------------

class TestIsPdf:
    def test_url_ending_with_pdf_extension(self):
        from utils.fetch_document import _is_pdf
        session, _ = _mock_session()
        assert _is_pdf("https://example.com/paper.pdf", session) is True

    def test_url_ending_with_uppercase_pdf_extension(self):
        from utils.fetch_document import _is_pdf
        session, _ = _mock_session()
        assert _is_pdf("https://example.com/paper.PDF", session) is True

    def test_html_url_returns_false(self):
        from utils.fetch_document import _is_pdf
        session, _ = _mock_session(headers={"Content-Type": "text/html; charset=utf-8"})
        assert _is_pdf("https://example.com/article", session) is False

    def test_url_with_pdf_content_type_returns_true(self):
        from utils.fetch_document import _is_pdf
        session, _ = _mock_session(headers={"Content-Type": "application/pdf"})
        assert _is_pdf("https://arxiv.org/pdf/2510.18234", session) is True

    def test_request_exception_returns_false(self):
        from utils.fetch_document import _is_pdf
        session = MagicMock(spec=requests.Session)
        session.head.side_effect = requests.RequestException
        assert _is_pdf("https://broken.example.com/doc", session) is False

    def test_missing_content_type_header_returns_false(self):
        from utils.fetch_document import _is_pdf
        session, _ = _mock_session(headers={})
        assert _is_pdf("https://example.com/no-header", session) is False


# ---------------------------------------------------------------------------
# _fetch_html
# ---------------------------------------------------------------------------

class TestFetchHtml:
    def test_returns_markdown_string(self):
        from utils.fetch_document import _fetch_html
        session, _ = _mock_session(text="<html><body><h1>Hello</h1></body></html>")
        result = _fetch_html("https://example.com", session)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_html_converted_to_markdown(self):
        from utils.fetch_document import _fetch_html
        session, _ = _mock_session(text="<html><body><h1>Title</h1><p>Body text.</p></body></html>")
        result = _fetch_html("https://example.com", session)
        assert "Title" in result
        assert "Body text." in result


# ---------------------------------------------------------------------------
# _fetch_pdf
# ---------------------------------------------------------------------------

class TestFetchPdf:
    def test_returns_text_content(self):
        mock_markitdown_result = MagicMock()
        mock_markitdown_result.text_content = "Converted PDF text"
        mock_md_instance = MagicMock()
        mock_md_instance.convert.return_value = mock_markitdown_result

        session, _ = _mock_session(content=b"%PDF-1.4 fake content")

        with patch("utils.fetch_document.MarkItDown", return_value=mock_md_instance):
            from utils.fetch_document import _fetch_pdf
            result = _fetch_pdf("https://example.com/paper.pdf", session)

        assert result == "Converted PDF text"

    def test_temp_file_cleaned_up(self, tmp_path):
        mock_result = MagicMock()
        mock_result.text_content = "text"
        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result

        session, _ = _mock_session(content=b"fake pdf bytes")
        captured_tmp_paths = []

        original_ntf = __import__("tempfile").NamedTemporaryFile

        def tracking_ntf(*args, **kwargs):
            ntf = original_ntf(*args, **kwargs)
            captured_tmp_paths.append(Path(ntf.name))
            return ntf

        with patch("utils.fetch_document.MarkItDown", return_value=mock_md):
            with patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
                from utils.fetch_document import _fetch_pdf
                _fetch_pdf("https://example.com/paper.pdf", session)

        for p in captured_tmp_paths:
            assert not p.exists(), f"Temp file not cleaned up: {p}"


# ---------------------------------------------------------------------------
# fetch_document (routing)
# ---------------------------------------------------------------------------

class TestFetchDocument:
    def test_pdf_url_calls_fetch_pdf_and_returns_content(self):
        # .pdf extension triggers the fast path — _fetch_pdf called directly
        with patch("utils.fetch_document._fetch_pdf", return_value="pdf markdown") as mock_pdf:
            with patch("utils.fetch_document._auto_cookies", return_value=None):
                from utils.fetch_document import fetch_document
                result = fetch_document("https://example.com/paper.pdf")

        mock_pdf.assert_called_once_with("https://example.com/paper.pdf", ANY)
        assert result == "pdf markdown"

    def test_html_url_calls_html2text_and_returns_content(self):
        # Non-.pdf URL: single GET, Content-Type=text/html → html2text branch
        session, _ = _mock_session(
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html><body>body</body></html>",
        )
        with patch("utils.fetch_document.requests.Session", return_value=session):
            with patch("utils.fetch_document._auto_cookies", return_value=None):
                with patch("utils.fetch_document.html2text.html2text", return_value="html markdown") as mock_h2t:
                    from utils.fetch_document import fetch_document
                    result = fetch_document("https://example.com/article")

        session.get.assert_called_once()
        mock_h2t.assert_called_once()
        assert result == "html markdown"

    def test_non_extension_pdf_url_calls_fetch_pdf_from_response(self):
        # Non-.pdf URL but Content-Type=application/pdf → _fetch_pdf_from_response branch
        session, mock_resp = _mock_session(
            headers={"Content-Type": "application/pdf"},
            content=b"%PDF fake",
        )
        with patch("utils.fetch_document.requests.Session", return_value=session):
            with patch("utils.fetch_document._auto_cookies", return_value=None):
                with patch("utils.fetch_document._fetch_pdf_from_response", return_value="pdf text") as mock_pfr:
                    from utils.fetch_document import fetch_document
                    result = fetch_document("https://arxiv.org/pdf/2303.08774")

        session.get.assert_called_once()
        mock_pfr.assert_called_once()
        assert result == "pdf text"

    def test_returns_string(self):
        session, _ = _mock_session(
            headers={"Content-Type": "text/html"},
            text="<html><body>content</body></html>",
        )
        with patch("utils.fetch_document.requests.Session", return_value=session):
            with patch("utils.fetch_document._auto_cookies", return_value=None):
                with patch("utils.fetch_document.html2text.html2text", return_value="some content"):
                    from utils.fetch_document import fetch_document
                    result = fetch_document("https://example.com/article")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Cookie loading
# ---------------------------------------------------------------------------

class TestLoadCookies:
    def test_load_cookies_valid(self, tmp_path):
        cookies_file = tmp_path / "cookies.txt"
        cookies_file.write_text("# Netscape HTTP Cookie File\n")

        with patch("utils.fetch_document.MozillaCookieJar") as mock_jar_cls:
            mock_jar = MagicMock()
            mock_jar_cls.return_value = mock_jar
            from utils.fetch_document import _load_cookies
            result = _load_cookies(str(cookies_file))

        mock_jar_cls.assert_called_once_with(str(cookies_file))
        mock_jar.load.assert_called_once_with(ignore_discard=True, ignore_expires=True)
        assert result is mock_jar

    def test_explicit_cookies_file_used_when_provided(self, tmp_path):
        cookies_file = tmp_path / "cookies.txt"
        cookies_file.write_text("# Netscape HTTP Cookie File\n")

        session, _ = _mock_session(headers={"Content-Type": "text/html"}, text="content")
        with patch("utils.fetch_document._load_cookies") as mock_load:
            with patch("utils.fetch_document._auto_cookies") as mock_auto:
                with patch("utils.fetch_document.requests.Session", return_value=session):
                    with patch("utils.fetch_document.html2text.html2text", return_value="content"):
                        from utils.fetch_document import fetch_document
                        fetch_document("https://example.com", cookies_path=str(cookies_file))

        mock_load.assert_called_once_with(str(cookies_file))
        mock_auto.assert_not_called()

    def test_no_cookies_path_attempts_auto_detect(self):
        session, _ = _mock_session(headers={"Content-Type": "text/html"}, text="content")
        with patch("utils.fetch_document._auto_cookies", return_value=None) as mock_auto:
            with patch("utils.fetch_document._load_cookies") as mock_load:
                with patch("utils.fetch_document.requests.Session", return_value=session):
                    with patch("utils.fetch_document.html2text.html2text", return_value="content"):
                        from utils.fetch_document import fetch_document
                        fetch_document("https://example.com")

        mock_auto.assert_called_once_with("https://example.com")
        mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# Auto cookie detection
# ---------------------------------------------------------------------------

class TestAutoCookies:
    def test_chrome_success_returns_jar(self):
        mock_jar = MagicMock()
        mock_browser_cookie3 = MagicMock()
        mock_browser_cookie3.chrome.return_value = mock_jar

        with patch.dict("sys.modules", {"browser_cookie3": mock_browser_cookie3}):
            from utils.fetch_document import _auto_cookies
            result = _auto_cookies("https://example.com/page")

        assert result is mock_jar

    def test_all_loaders_fail_returns_none(self):
        mock_browser_cookie3 = MagicMock()
        mock_browser_cookie3.chrome.side_effect = Exception("locked")
        mock_browser_cookie3.firefox.side_effect = Exception("locked")
        mock_browser_cookie3.edge.side_effect = Exception("locked")

        with patch.dict("sys.modules", {"browser_cookie3": mock_browser_cookie3}):
            from utils.fetch_document import _auto_cookies
            result = _auto_cookies("https://example.com/page")

        assert result is None

    def test_import_error_returns_none(self):
        with patch.dict("sys.modules", {"browser_cookie3": None}):
            from utils.fetch_document import _auto_cookies
            result = _auto_cookies("https://example.com/page")

        assert result is None


# ---------------------------------------------------------------------------
# Auth failure detection
# ---------------------------------------------------------------------------

class TestCheckAuthFailure:
    def test_401_raises_auth_required_error(self):
        from utils.fetch_document import AuthRequiredError, _check_auth_failure
        resp = MagicMock()
        resp.status_code = 401
        resp.url = "https://example.com/resource"
        with pytest.raises(AuthRequiredError) as exc_info:
            _check_auth_failure(resp, "https://example.com/resource")
        assert "logged in" in str(exc_info.value)
        assert "--cookies" in str(exc_info.value)

    def test_403_raises_auth_required_error(self):
        from utils.fetch_document import AuthRequiredError, _check_auth_failure
        resp = MagicMock()
        resp.status_code = 403
        resp.url = "https://example.com/resource"
        with pytest.raises(AuthRequiredError):
            _check_auth_failure(resp, "https://example.com/resource")

    def test_login_redirect_raises_auth_required_error(self):
        from utils.fetch_document import AuthRequiredError, _check_auth_failure
        resp = MagicMock()
        resp.status_code = 200
        resp.url = "https://example.com/login?next=/article"
        with pytest.raises(AuthRequiredError):
            _check_auth_failure(resp, "https://example.com/article")

    def test_signin_redirect_raises_auth_required_error(self):
        from utils.fetch_document import AuthRequiredError, _check_auth_failure
        resp = MagicMock()
        resp.status_code = 200
        resp.url = "https://example.com/signin"
        with pytest.raises(AuthRequiredError):
            _check_auth_failure(resp, "https://example.com/article")

    def test_200_ok_url_does_not_raise(self):
        from utils.fetch_document import _check_auth_failure
        resp = MagicMock()
        resp.status_code = 200
        resp.url = "https://example.com/article"
        _check_auth_failure(resp, "https://example.com/article")  # should not raise

    def test_auth_failure_in_is_pdf_propagates(self):
        from utils.fetch_document import AuthRequiredError, _is_pdf
        session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.url = "https://example.com/doc"
        session.head.return_value = mock_resp
        with pytest.raises(AuthRequiredError):
            _is_pdf("https://example.com/doc", session)

    def test_auth_failure_in_fetch_html_propagates(self):
        from utils.fetch_document import AuthRequiredError, _fetch_html
        session, mock_resp = _mock_session(status_code=403, url="https://example.com/article")
        with pytest.raises(AuthRequiredError):
            _fetch_html("https://example.com/article", session)

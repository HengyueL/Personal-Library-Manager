import logging
import tempfile
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from urllib.parse import urlparse

import html2text
import requests
from markitdown import MarkItDown

logger = logging.getLogger(__name__)

_AUTH_URL_KEYWORDS = {"login", "signin", "auth", "sso"}


class AuthRequiredError(Exception):
    """Raised when a URL requires authentication that was not supplied."""
    pass


def _auth_error_message(url: str) -> str:
    return (
        f"Access to {url} was blocked — the site likely requires you to be logged in.\n\n"
        "Automatic cookie detection failed or found no valid session. To fix this:\n"
        "  1. Open the URL in your browser and log in.\n"
        '  2. Install the "Get cookies.txt LOCALLY" browser extension.\n'
        "  3. Export cookies for this site to a file (e.g. cookies.txt).\n"
        "  4. Re-run with --cookies /path/to/cookies.txt"
    )


def _check_auth_failure(response: requests.Response, original_url: str) -> None:
    """Raise AuthRequiredError if the response indicates an auth wall."""
    if response.status_code in {401, 403}:
        raise AuthRequiredError(_auth_error_message(original_url))
    final_url = response.url.lower()
    if any(kw in final_url for kw in _AUTH_URL_KEYWORDS):
        raise AuthRequiredError(_auth_error_message(original_url))


def _load_cookies(cookies_path: str) -> MozillaCookieJar:
    """Load a Netscape cookies.txt file into a MozillaCookieJar."""
    jar = MozillaCookieJar(cookies_path)
    jar.load(ignore_discard=True, ignore_expires=True)
    return jar


def _auto_cookies(url: str):
    """Try to read cookies for *url*'s domain from installed browsers.

    Returns a cookie jar on success, or None if unavailable.
    """
    try:
        import browser_cookie3
        domain = urlparse(url).netloc
        for loader in (browser_cookie3.chrome, browser_cookie3.firefox, browser_cookie3.edge):
            try:
                jar = loader(domain_name=domain)
                if jar:
                    return jar
            except Exception:
                continue
    except ImportError:
        pass
    return None


def _is_pdf(url: str, session: requests.Session) -> bool:
    if urlparse(url).path.lower().endswith(".pdf"):
        return True
    try:
        resp = session.head(url, allow_redirects=True, timeout=10)
        _check_auth_failure(resp, url)
        return "application/pdf" in resp.headers.get("Content-Type", "")
    except requests.RequestException:
        return False


def fetch_document(url: str, cookies_path: str | None = None) -> str:
    """
    Fetch a document from a URL and return its content as a markdown string.

    Args:
        url: The URL to fetch.
        cookies_path: Optional path to a Netscape cookies.txt file. If omitted,
            browser cookies are auto-detected. Pass explicitly to override.

    Returns:
        Markdown content string (without YAML frontmatter).

    Raises:
        AuthRequiredError: If the URL requires authentication and no valid
            session could be established.
    """
    session = requests.Session()
    if cookies_path:
        session.cookies = _load_cookies(cookies_path)
    else:
        auto = _auto_cookies(url)
        if auto:
            session.cookies = auto

    if _is_pdf(url, session):
        logger.info("Detected PDF; converting with MarkItDown")
        return _fetch_pdf(url, session)
    else:
        logger.info("Detected HTML; converting with html2text")
        return _fetch_html(url, session)


def _fetch_html(url: str, session: requests.Session) -> str:
    resp = session.get(url)
    _check_auth_failure(resp, url)
    return html2text.html2text(resp.text)


def _fetch_pdf(url: str, session: requests.Session) -> str:
    response = session.get(url)
    _check_auth_failure(response, url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = Path(tmp.name)

    try:
        result = MarkItDown().convert(str(tmp_path))
        return result.text_content
    finally:
        tmp_path.unlink(missing_ok=True)

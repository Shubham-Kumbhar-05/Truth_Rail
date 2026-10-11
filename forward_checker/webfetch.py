"""Fetch the text of a web page safely. Blocks private/internal addresses (SSRF), limits size and redirects."""
import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

import requests

from . import config


class WebFetchError(Exception):
    pass


URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)


def find_single_url(text):
    """Returns the URL if the message is essentially just one link, else None."""
    t = (text or "").strip()
    urls = URL_RE.findall(t)
    if len(urls) == 1 and len(t.replace(urls[0], "").strip()) < 20:
        return urls[0].rstrip(".,)")
    return None


def _resolve(host):
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def is_public_host(host, resolver=None):
    try:
        ips = (resolver or _resolve)(host)
    except OSError:
        return False
    if not ips:
        return False
    for ip in ips:
        try:
            if not ipaddress.ip_address(ip).is_global:
                return False
        except ValueError:
            return False
    return True


def _check_url(url, resolver):
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise WebFetchError("Only normal http/https links can be checked.")
    if p.port not in (None, 80, 443):
        raise WebFetchError("This link uses an unusual port and was skipped for safety.")
    if not is_public_host(p.hostname, resolver):
        raise WebFetchError("This link points to a private or unknown address and was skipped for safety.")


def fetch(url, getter=None, resolver=None):
    """Returns (content_type, bytes). Follows at most 4 redirects, re-checking every hop."""
    getter = getter or requests.get
    for _ in range(5):
        _check_url(url, resolver)
        try:
            resp = getter(url, stream=True, timeout=config.URL_FETCH_TIMEOUT, allow_redirects=False,
                          headers={"User-Agent": "Mozilla/5.0 (ForwardChecker; fact-checking bot)"})
        except requests.RequestException as e:
            raise WebFetchError(f"Could not open the link: {e}")
        if resp.status_code in (301, 302, 303, 307, 308):
            url = urljoin(url, resp.headers.get("Location", ""))
            continue
        if resp.status_code != 200:
            raise WebFetchError(f"The link returned HTTP {resp.status_code}.")
        data = b""
        for chunk in resp.iter_content(65536):
            data += chunk
            if len(data) > config.MAX_URL_BYTES:
                raise WebFetchError("The page is too large to check.")
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        return ctype, data
    raise WebFetchError("Too many redirects.")


def html_to_text(html):
    try:
        import trafilatura
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if text and len(text.strip()) > 40:
            return text.strip()
    except Exception:
        pass
    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", html)).strip()


def fetch_url_text(url, getter=None, resolver=None, pdf_reader=None):
    ctype, data = fetch(url, getter, resolver)
    if ctype == "application/pdf":
        if not pdf_reader:
            raise WebFetchError("PDF links are not supported here.")
        text = pdf_reader(data)
    elif ctype.startswith("text/") or "html" in ctype or ctype == "":
        text = html_to_text(data.decode("utf-8", errors="replace"))
    else:
        raise WebFetchError(f"This link is a '{ctype}' file, which cannot be checked.")
    text = text.strip()
    if len(text) < 30:
        raise WebFetchError("Could not find readable text on this page.")
    return text[: config.MAX_TEXT_CHARS]

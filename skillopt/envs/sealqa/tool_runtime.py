from __future__ import annotations

import html
import re
from urllib.request import Request, urlopen

DEFAULT_USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/135.0 Safari/537.36'
)
_MAX_FETCH_CHARS = 6000


def _strip_html(raw_html: str) -> str:
    cleaned = re.sub(r'(?is)<script.*?>.*?</script>', ' ', raw_html)
    cleaned = re.sub(r'(?is)<style.*?>.*?</style>', ' ', cleaned)
    cleaned = re.sub(r'(?is)<[^>]+>', ' ', cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def web_fetch(url: str, max_chars: int = _MAX_FETCH_CHARS) -> str:
    req = Request(url, headers={'User-Agent': DEFAULT_USER_AGENT})
    with urlopen(req, timeout=20) as response:
        body = response.read().decode('utf-8', errors='ignore')
    text = _strip_html(body)
    if len(text) > max_chars:
        omitted = len(text) - max_chars
        text = text[:max_chars] + f"\n\n[... {omitted} characters omitted ...]"
    return f"URL: {url}\n\n{text}"

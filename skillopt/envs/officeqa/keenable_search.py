"""Keenable web search client for the OfficeQA controller's search rounds.

Reached only through the explicit `search_mode=keenable` selection. Every call
sends each of the model's generated queries, and nothing else from the task,
to Keenable's search API. Without `KEENABLE_API_KEY` requests go to the keyless
endpoint; a key only raises the rate limits.
"""
from __future__ import annotations

import os
import time
from urllib.parse import urlsplit

import httpx

from skillopt import __version__

KEYLESS_URL = "https://api.keenable.ai/v1/search/public"
KEYED_URL = "https://api.keenable.ai/v1/search"
API_KEY_ENV = "KEENABLE_API_KEY"
# The keyless endpoint rejects requests that do not name the calling app.
APP_TITLE = "SkillOpt"
USER_AGENT = f"SkillOpt/{__version__} (+https://github.com/microsoft/SkillOpt)"
MAX_RESULTS_LIMIT = 50


def _render(index: int, query: str, results: list, max_num_results: int) -> str:
    blocks = []
    for item in results[:max_num_results]:
        if not isinstance(item, dict):
            raise RuntimeError("Keenable search returned an invalid result item")
        url = item.get("url")
        snippet = item.get("snippet") or ""
        if not isinstance(url, str) or urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).netloc:
            raise RuntimeError("Keenable search returned an invalid result URL")
        if not isinstance(snippet, str):
            raise RuntimeError("Keenable search returned an invalid snippet")
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            title = url
        blocks.append(f"[{len(blocks) + 1}] {title.strip()}\nURL: {url}\n{snippet.strip()[:6000]}")
    body = "\n\n".join(blocks) if blocks else "[no search results]"
    return f"## Query {index}\nQuery: {query}\n\n{body}"


def keenable_search(queries: list[str], *, max_num_results: int = 4, timeout: float = 20) -> str:
    """Search each query within one round deadline and render URL evidence.

    Service errors propagate to the controller; an exceeded deadline always
    raises the built-in `TimeoutError`.
    """
    if not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("Keenable search requires non-empty queries")
    if timeout <= 0 or not 1 <= max_num_results <= MAX_RESULTS_LIMIT:
        raise ValueError(f"Keenable search needs a positive timeout and 1-{MAX_RESULTS_LIMIT} results")
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    headers = {"User-Agent": USER_AGENT, "X-Keenable-Title": APP_TITLE}
    if api_key:
        headers["X-API-Key"] = api_key
    url = KEYED_URL if api_key else KEYLESS_URL
    deadline = time.monotonic() + timeout
    sections = []
    with httpx.Client(headers=headers, follow_redirects=False) as client:
        for index, query in enumerate(queries, start=1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Keenable search exceeded its {timeout}s deadline")
            try:
                response = client.post(
                    url, json={"query": query, "max_results": max_num_results}, timeout=remaining,
                )
            except httpx.TimeoutException as exc:
                raise TimeoutError(f"Keenable search exceeded its {timeout}s deadline") from exc
            # httpx applies `timeout` per phase, so a slow response can still end
            # past the round deadline; check it again once the response is in.
            if time.monotonic() > deadline:
                raise TimeoutError(f"Keenable search exceeded its {timeout}s deadline")
            if response.status_code != 200:
                # The error text reaches fail_reason, results.jsonl and the
                # reflection prompt, so never let an echoed key travel with it.
                # Redact before cutting, so a key straddling the cut can't leak a prefix.
                detail = response.text
                if api_key:
                    detail = detail.replace(api_key, "[redacted]")
                raise RuntimeError(f"Keenable search HTTP {response.status_code}: {detail[:1000]}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError("Keenable search returned invalid JSON") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise RuntimeError("Keenable search returned invalid results")
            sections.append(_render(index, query, payload["results"], max_num_results))
    return "\n\n".join(sections)

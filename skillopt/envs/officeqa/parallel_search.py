"""Parallel Search MCP client for the OfficeQA controller's search rounds.

Reached only through the explicit `search_mode=parallel` selection. Every call
sends the model's generated queries and the full task question (as the MCP
`objective`) to Parallel's hosted endpoint, anonymously, on the documented
lower-rate free tier.
"""
from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Implementation

from skillopt import __version__

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = f"SkillOpt/{__version__} (+https://github.com/microsoft/SkillOpt)"


async def _search(queries: list[str], objective: str, session_id: str, timeout: float) -> dict:
    # This anonymous connection deliberately does not read API keys or saved MCP
    # credentials. Keep the same project identity on every transport request.
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(timeout),
        follow_redirects=False,
    ) as client:
        async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
            async with ClientSession(
                read, write,
                read_timeout_seconds=timedelta(seconds=timeout),
                client_info=Implementation(name="SkillOpt", version=__version__),
            ) as session:
                await session.initialize()
                tools = await session.list_tools()
                if not any(tool.name == "web_search" for tool in tools.tools):
                    raise RuntimeError("Parallel MCP did not advertise web_search")
                result = await session.call_tool("web_search", {
                    "objective": objective,
                    "search_queries": queries,
                    "session_id": session_id,
                })
                text = "\n".join(block.text for block in result.content if block.type == "text")
                if result.isError:
                    raise RuntimeError(f"Parallel search error: {text[:1000]}")
                payload = result.structuredContent
                if payload is None:
                    try:
                        payload = json.loads(text)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError("Parallel search returned invalid JSON evidence") from exc
                if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                    raise RuntimeError("Parallel search returned invalid results")
                return payload


def parallel_search(
    queries: list[str], *, objective: str, session_id: str,
    max_num_results: int = 4, timeout: float = 20,
) -> str:
    """Search related queries with a bounded request and render URL evidence.

    The controller supplies one stable session ID per OfficeQA task, including
    subsequent rounds. Timeouts and cancellation propagate to the controller;
    an exceeded deadline always raises the built-in `TimeoutError`.
    """
    if not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("Parallel search requires non-empty queries")
    if timeout <= 0 or max_num_results <= 0:
        raise ValueError("Parallel search timeout and result limit must be positive")

    async def bounded() -> dict:
        return await asyncio.wait_for(_search(queries, objective, session_id, timeout), timeout=timeout)

    try:
        payload = asyncio.run(bounded())
    except Exception as exc:
        # SDK task groups may wrap a useful transport/tool error. Preserve the
        # underlying diagnostic, including explicit anonymous quota responses.
        error = exc
        while len(getattr(error, "exceptions", ())) == 1:
            error = error.exceptions[0]
        if isinstance(error, asyncio.TimeoutError):
            # Before 3.11 `asyncio.TimeoutError` is its own class rather than the
            # built-in one, so normalize the deadline for callers on every version.
            raise TimeoutError(f"Parallel search exceeded its {timeout}s deadline") from exc
        if error is not exc:
            raise error from exc
        raise
    blocks = []
    for item in payload["results"][:max_num_results]:
        if not isinstance(item, dict):
            raise RuntimeError("Parallel search returned an invalid result item")
        url = item.get("url")
        excerpts = item.get("excerpts")
        if not isinstance(url, str) or urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).netloc:
            raise RuntimeError("Parallel search returned an invalid result URL")
        if not isinstance(excerpts, list) or any(not isinstance(part, str) for part in excerpts):
            raise RuntimeError("Parallel search returned invalid excerpts")
        blocks.append(f"[{len(blocks) + 1}] {item.get('title') or url}\nURL: {url}\n" + "\n".join(excerpts)[:6000])
    return "\n\n".join(blocks) if blocks else "[no search results]"

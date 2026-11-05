"""OfficeQA search-mode selection, optional-dependency handling and MCP caller coverage."""
from __future__ import annotations

import asyncio
import builtins
import importlib
import importlib.util
import json
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from skillopt.envs.officeqa import rollout
from skillopt.envs.officeqa.adapter import OfficeQAAdapter

HAS_MCP = importlib.util.find_spec("mcp") is not None
requires_mcp = pytest.mark.skipif(not HAS_MCP, reason="requires the optional 'parallel' extra")
mcp_search = importlib.import_module("skillopt.envs.officeqa.parallel_search") if HAS_MCP else None

ITEM = {"id": "treasury", "question": "When was the declaration signed?", "ground_truth": "1776"}
PAYLOAD = {"results": [{"url": "https://example.org/treasury?year=1776", "title": "Treasury history", "excerpts": ["The declaration was signed in 1776."]}]}


@pytest.fixture(autouse=True)
def isolated_search_config(monkeypatch):
    for name in ("OFFICEQA_CUSTOM_SEARCH_AUTH", "OFFICEQA_SEARCH_API_URL", "PARALLEL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rollout, "get_target_backend", lambda: "openai_chat")


@pytest.fixture
def transport(monkeypatch):
    """Observe actual SDK HTTP requests without substituting its MCP protocol."""
    import httpx

    requests = []
    behavior = {"payload": PAYLOAD, "error": None, "delay": False}
    original_client = httpx.AsyncClient

    async def handler(request):
        body = json.loads(request.content) if request.content else {}
        requests.append((request, body))
        method = body.get("method")
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            if behavior["delay"]:
                await asyncio.Event().wait()
            if behavior["error"]:
                result = {"content": [{"type": "text", "text": behavior["error"]}], "isError": True}
            else:
                result = {"content": [{"type": "text", "text": json.dumps(behavior["payload"])}], "structuredContent": behavior["payload"]}
        else:
            raise AssertionError(f"Unexpected MCP request: {request.method} {body}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    def client(**kwargs):
        return original_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(mcp_search.httpx, "AsyncClient", client)
    return requests, behavior


def scripted_model(monkeypatch, *, rounds=1):
    calls = []

    def chat(**kwargs):
        # Copy now because the caller appends further messages after the call.
        messages = [dict(message) for message in kwargs["messages"]]
        calls.append(messages)
        if len(calls) <= rounds:
            text = '<search_queries>["declaration treasury history"]</search_queries>'
        else:
            assert "https://example.org/treasury?year=1776" in messages[-1]["content"]
            assert "signed in 1776" in messages[-1]["content"]
            text = "<answer>1776</answer>"
        return SimpleNamespace(content=text), {}

    monkeypatch.setattr(rollout, "chat_target_messages", chat)
    return calls


def block_mcp_import(monkeypatch):
    """Make `import mcp` fail the way an install without the extra would."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    for name in [key for key in sys.modules if key == "mcp" or key.startswith("mcp.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "skillopt.envs.officeqa.parallel_search", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded)


# ── Default selection: Parallel is never automatic ───────────────────────────

def test_shipped_config_keeps_the_offline_default():
    from scripts.train import get_adapter
    from skillopt.config import flatten_config, load_config

    cfg = flatten_config(load_config("configs/officeqa/default.yaml"))
    assert cfg["search_mode"] == "offline"
    adapter = get_adapter(cfg)
    assert adapter.search_mode == "offline"
    assert adapter.search_provider == "duckduckgo"
    assert rollout._normalize_search_mode(adapter.search_mode) == rollout._DEFAULT_SEARCH_MODE


@pytest.mark.parametrize("mode", ["auto", "", None, "fresh", "web", "parallel_search"])
def test_auto_and_unrecognized_modes_never_select_parallel(mode):
    assert rollout._normalize_search_mode(mode) == rollout._DEFAULT_SEARCH_MODE


@pytest.mark.parametrize("mode", ["parallel", "PARALLEL", " Parallel "])
def test_parallel_requires_explicit_selection(mode):
    assert rollout._normalize_search_mode(mode) == rollout._PARALLEL_SEARCH_MODE


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("offline", "offline"), ("custom", "custom_search"), ("custom_search", "custom_search"), ("azure", "azure_search"), ("azure_search", "azure_search")],
)
def test_existing_modes_resolve_unchanged(mode, expected):
    assert rollout._normalize_search_mode(mode) == expected


@pytest.mark.parametrize("mode", ["auto", "offline"])
def test_fresh_auto_config_runs_offline_without_search(tmp_path, monkeypatch, mode):
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr(rollout, "chat_target_messages", lambda **kwargs: (SimpleNamespace(content="<answer>1776</answer>"), {}))
    with (
        mock.patch.object(rollout, "_load_parallel_search", side_effect=AssertionError("parallel not selected")),
        mock.patch.object(rollout, "custom_search", side_effect=AssertionError("custom search not selected")),
        mock.patch.object(rollout, "run_tool", side_effect=AssertionError("local tools disabled")),
    ):
        row = rollout.process_one(ITEM, str(tmp_path / "out"), "", search_mode=mode, use_local_tools=False, data_dirs=[str(docs)])
    assert row["predicted_answer"] == "1776"


# ── Optional `mcp` dependency ────────────────────────────────────────────────

def test_officeqa_env_imports_without_mcp(monkeypatch):
    block_mcp_import(monkeypatch)
    for name in [key for key in sys.modules if key.startswith("skillopt.envs.officeqa")]:
        monkeypatch.delitem(sys.modules, name)
    reloaded = importlib.import_module("skillopt.envs.officeqa.adapter")
    assert reloaded.OfficeQAAdapter().search_mode == "offline"
    assert importlib.import_module("skillopt.envs.officeqa.rollout")._normalize_search_mode("offline") == "offline"


def test_parallel_without_the_extra_names_the_install_command(monkeypatch):
    block_mcp_import(monkeypatch)
    with pytest.raises(RuntimeError, match=r"skillopt\[parallel\]"):
        rollout._load_parallel_search()


def test_parallel_rollout_without_the_extra_fails_with_the_install_command(tmp_path, monkeypatch):
    block_mcp_import(monkeypatch)
    scripted_model(monkeypatch)
    row = rollout.process_one(ITEM, str(tmp_path), "", search_mode="parallel", max_tool_turns=2)
    assert "skillopt[parallel]" in row["fail_reason"]
    assert row["hard"] == 0


# ── Parallel MCP caller ──────────────────────────────────────────────────────

@requires_mcp
@pytest.mark.parametrize("backend", ["openai_chat", "qwen_chat"])
def test_explicit_parallel_runs_mcp_and_consumes_evidence(tmp_path, monkeypatch, transport, backend):
    monkeypatch.setattr(rollout, "get_target_backend", lambda: backend)
    calls = scripted_model(monkeypatch, rounds=2)
    adapter = OfficeQAAdapter(search_mode="parallel", workers=1, max_tool_turns=3)
    rows = adapter.rollout([ITEM], "Read evidence before answering.", str(tmp_path))
    assert rows[0]["predicted_answer"] == "1776"
    assert rows[0]["hard"] == 1
    assert len(calls) == 3
    requests, _ = transport
    methods = [body.get("method") for _, body in requests]
    assert methods.count("initialize") == methods.count("tools/list") == methods.count("tools/call") == 2
    searches = [body["params"] for _, body in requests if body.get("method") == "tools/call"]
    assert all(call["name"] == "web_search" for call in searches)
    assert searches[0]["arguments"]["session_id"] == searches[1]["arguments"]["session_id"]
    assert len(searches[0]["arguments"]["session_id"]) == 32
    assert searches[0]["arguments"]["search_queries"] == ["declaration treasury history"]
    # Documented data boundary: the whole task question leaves as the objective.
    assert searches[0]["arguments"]["objective"] == ITEM["question"]
    assert all(str(request.url) == "https://search.parallel.ai/mcp" for request, _ in requests)
    assert all(request.headers["user-agent"] == mcp_search.USER_AGENT for request, _ in requests)
    assert all("authorization" not in request.headers for request, _ in requests)
    conversation = json.loads((tmp_path / "predictions" / ITEM["id"] / "conversation.json").read_text())
    assert any(event.get("type") == "tool_call" and "signed in 1776" in event["obs"] for event in conversation)


@requires_mcp
def test_environment_key_is_not_sent_to_anonymous_mcp(monkeypatch, transport):
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-sent")
    text = mcp_search.parallel_search(["treasury"], objective="history", session_id="a" * 32)
    assert "https://example.org/treasury?year=1776" in text
    assert all("authorization" not in request.headers for request, _ in transport[0])


@requires_mcp
def test_explicit_parallel_ignores_legacy_credentials(tmp_path, monkeypatch, transport):
    monkeypatch.setenv("OFFICEQA_CUSTOM_SEARCH_AUTH", "configured-token")
    monkeypatch.setenv("OFFICEQA_SEARCH_API_URL", "https://configured.example")
    scripted_model(monkeypatch)
    row = rollout.process_one(ITEM, str(tmp_path), "", search_mode="parallel", search_provider="google", max_tool_turns=2)
    assert row["hard"] == 1
    assert transport[0]
    assert all("authorization" not in request.headers for request, _ in transport[0])


@requires_mcp
def test_saved_custom_provider_failure_does_not_use_parallel(tmp_path, monkeypatch, transport):
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setenv("OFFICEQA_CUSTOM_SEARCH_AUTH", "configured-token")
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(content='<search_queries>["treasury"]</search_queries>'), {}
        assert "configured service failed" in kwargs["messages"][-1]["content"]
        return SimpleNamespace(content="<answer>unknown</answer>"), {}

    monkeypatch.setattr(rollout, "chat_target_messages", chat)
    with mock.patch.object(rollout, "custom_search", side_effect=RuntimeError("configured service failed")) as custom:
        row = rollout.process_one(ITEM, str(tmp_path / "out"), "", search_mode="custom_search", search_api_url="https://configured.example", search_provider="google", data_dirs=[str(docs)], max_tool_turns=2)
    assert custom.call_args.kwargs["provider"] == "google"
    assert custom.call_args.kwargs["api_url"] == "https://configured.example"
    assert not transport[0]
    assert row["hard"] == 0


@requires_mcp
@pytest.mark.parametrize("error", ["anonymous rate limit exceeded", "upstream unavailable"])
def test_mcp_errors_fail_task_without_fallback(tmp_path, monkeypatch, transport, error):
    transport[1]["error"] = error
    calls = scripted_model(monkeypatch)
    with mock.patch.object(rollout, "custom_search", side_effect=AssertionError("no fallback")):
        row = rollout.process_one(ITEM, str(tmp_path), "", search_mode="parallel", max_tool_turns=2)
    assert error in row["fail_reason"]
    assert row["hard"] == 0
    assert len(calls) == 1


@requires_mcp
def test_whole_mcp_request_has_a_deadline(transport):
    transport[1]["delay"] = True
    with pytest.raises(TimeoutError):
        mcp_search.parallel_search(["treasury"], objective="history", session_id="a" * 32, timeout=0.05)


@requires_mcp
def test_cancellation_closes_actual_mcp_session(transport):
    transport[1]["delay"] = True

    async def cancel():
        task = asyncio.create_task(mcp_search._search(["treasury"], "history", "a" * 32, 20))
        for _ in range(500):
            if any(body.get("method") == "tools/call" for _, body in transport[0]):
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail("MCP caller never reached search")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())


@requires_mcp
@pytest.mark.parametrize("payload", [{"results": [{"url": "file:///etc/passwd", "excerpts": ["bad"]}]}, {"results": "malformed"}, {"results": [{"url": "https://example.org", "excerpts": "not a list"}]}])
def test_bad_mcp_evidence_is_rejected(transport, payload):
    transport[1]["payload"] = payload
    with pytest.raises(RuntimeError, match="invalid"):
        mcp_search.parallel_search(["treasury"], objective="history", session_id="a" * 32)

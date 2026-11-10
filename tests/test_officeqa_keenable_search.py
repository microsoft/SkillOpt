"""OfficeQA `search_mode=keenable` selection, request data boundary and evidence handling."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest

from skillopt.envs.officeqa import keenable_search as keenable
from skillopt.envs.officeqa import rollout
from skillopt.envs.officeqa.adapter import OfficeQAAdapter

ITEM = {"id": "treasury", "question": "When was the declaration signed?", "ground_truth": "1776"}
PAYLOAD = {"results": [{"url": "https://example.org/treasury?year=1776", "title": "Treasury history", "description": "", "snippet": "The declaration was signed in 1776."}]}


@pytest.fixture(autouse=True)
def isolated_search_config(monkeypatch):
    for name in ("OFFICEQA_CUSTOM_SEARCH_AUTH", "OFFICEQA_SEARCH_API_URL", "PARALLEL_API_KEY", "KEENABLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rollout, "get_target_backend", lambda: "openai_chat")


@pytest.fixture
def transport(monkeypatch):
    """Record the HTTP requests the client sends and answer them from `behavior`."""
    requests = []
    behavior = {"payload": PAYLOAD, "status": 200, "timeout": False, "response": None}
    original_client = httpx.Client

    def handler(request):
        requests.append((request, json.loads(request.content)))
        if behavior["response"] is not None:
            return behavior["response"]
        if behavior["timeout"]:
            raise httpx.ReadTimeout("read timed out", request=request)
        if behavior["status"] != 200:
            return httpx.Response(behavior["status"], json={"error": "Rate limit exceeded"})
        return httpx.Response(200, json=behavior["payload"])

    def client(**kwargs):
        return original_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(keenable.httpx, "Client", client)
    return requests, behavior


def scripted_model(monkeypatch, *, rounds=1, queries='["declaration treasury history"]'):
    calls = []

    def chat(**kwargs):
        messages = [dict(message) for message in kwargs["messages"]]
        calls.append(messages)
        if len(calls) <= rounds:
            text = f"<search_queries>{queries}</search_queries>"
        else:
            assert "https://example.org/treasury?year=1776" in messages[-1]["content"]
            assert "signed in 1776" in messages[-1]["content"]
            text = "<answer>1776</answer>"
        return SimpleNamespace(content=text), {}

    monkeypatch.setattr(rollout, "chat_target_messages", chat)
    return calls


# ── Selection: Keenable is never automatic ───────────────────────────────────

@pytest.mark.parametrize("mode", ["auto", "", None, "web", "keenable_search", "parallel"])
def test_auto_and_other_modes_never_select_keenable(mode):
    assert rollout._normalize_search_mode(mode) != rollout._KEENABLE_SEARCH_MODE


@pytest.mark.parametrize("mode", ["keenable", "KEENABLE", " Keenable "])
def test_keenable_requires_explicit_selection(mode):
    assert rollout._normalize_search_mode(mode) == rollout._KEENABLE_SEARCH_MODE


@pytest.mark.parametrize("mode", ["auto", "offline"])
def test_fresh_config_never_calls_keenable(tmp_path, monkeypatch, transport, mode):
    docs = tmp_path / "docs"
    docs.mkdir()
    # The model asks for search; outside explicit keenable mode nothing may reach Keenable.
    responses = iter(['<search_queries>["declaration treasury history"]</search_queries>'])
    monkeypatch.setattr(
        rollout, "chat_target_messages",
        lambda **kwargs: (SimpleNamespace(content=next(responses, "<answer>1776</answer>")), {}),
    )
    rollout.process_one(ITEM, str(tmp_path / "out"), "", search_mode=mode, use_local_tools=False, data_dirs=[str(docs)])
    assert transport[0] == []


# ── Rollout through the keyless endpoint ─────────────────────────────────────

@pytest.mark.parametrize("backend", ["openai_chat", "qwen_chat"])
def test_explicit_keenable_searches_and_consumes_evidence(tmp_path, monkeypatch, transport, backend):
    monkeypatch.setattr(rollout, "get_target_backend", lambda: backend)
    calls = scripted_model(monkeypatch, rounds=2)
    adapter = OfficeQAAdapter(search_mode="keenable", workers=1, max_tool_turns=3)
    rows = adapter.rollout([ITEM], "Read evidence before answering.", str(tmp_path))
    assert rows[0]["predicted_answer"] == "1776"
    assert rows[0]["hard"] == 1
    assert len(calls) == 3
    requests, _ = transport
    assert len(requests) == 2
    for request, body in requests:
        assert request.method == "POST"
        assert str(request.url) == keenable.KEYLESS_URL
        assert request.headers["x-keenable-title"] == "SkillOpt"
        assert request.headers["user-agent"] == keenable.USER_AGENT
        assert "x-api-key" not in request.headers and "authorization" not in request.headers
        # Documented data boundary: only the generated query leaves, never the task question.
        assert body == {"query": "declaration treasury history", "max_results": 4}
    conversation = json.loads((tmp_path / "predictions" / ITEM["id"] / "conversation.json").read_text())
    assert any(event.get("type") == "tool_call" and "signed in 1776" in event["obs"] for event in conversation)


def test_each_query_is_one_request_with_its_own_evidence_block(transport):
    text = keenable.keenable_search(["treasury", "declaration"], max_num_results=2)
    assert [body["query"] for _, body in transport[0]] == ["treasury", "declaration"]
    assert all(body["max_results"] == 2 for _, body in transport[0])
    assert text.index("## Query 1\nQuery: treasury") < text.index("## Query 2\nQuery: declaration")
    assert text.count("URL: https://example.org/treasury?year=1776") == 2


def test_results_beyond_the_limit_and_long_snippets_are_cut(transport):
    transport[1]["payload"] = {"results": [{"url": f"https://example.org/{n}", "title": f"T{n}", "snippet": "x" * 9000} for n in range(5)]}
    text = keenable.keenable_search(["treasury"], max_num_results=2)
    assert "https://example.org/1" in text and "https://example.org/2" not in text
    assert "x" * 6000 in text and "x" * 6001 not in text


def test_empty_results_render_a_placeholder(transport):
    transport[1]["payload"] = {"results": []}
    assert keenable.keenable_search(["treasury"]).endswith("[no search results]")


# ── Optional key and other credentials ───────────────────────────────────────

def test_api_key_switches_to_the_keyed_endpoint(monkeypatch, transport):
    monkeypatch.setenv("KEENABLE_API_KEY", "kn-test-key")
    keenable.keenable_search(["treasury"])
    request, _ = transport[0][0]
    assert str(request.url) == keenable.KEYED_URL
    assert request.headers["x-api-key"] == "kn-test-key"


def test_explicit_keenable_ignores_other_search_credentials(tmp_path, monkeypatch, transport):
    monkeypatch.setenv("OFFICEQA_CUSTOM_SEARCH_AUTH", "configured-token")
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-sent")
    scripted_model(monkeypatch)
    row = rollout.process_one(ITEM, str(tmp_path), "", search_mode="keenable", search_provider="google", max_tool_turns=2)
    assert row["hard"] == 1
    request, _ = transport[0][0]
    assert str(request.url) == keenable.KEYLESS_URL
    assert not {"authorization", "x-api-key"} & set(request.headers)
    assert "configured-token" not in str(request.headers) and "must-not-be-sent" not in str(request.headers)


# ── Failures: no fallback, bounded time, validated evidence ──────────────────

@pytest.mark.parametrize("status", [429, 503])
def test_service_errors_fail_the_task_without_fallback(tmp_path, monkeypatch, transport, status):
    transport[1]["status"] = status
    calls = scripted_model(monkeypatch)
    with (
        mock.patch.object(rollout, "custom_search", side_effect=AssertionError("no fallback")),
        mock.patch.object(rollout, "_load_parallel_search", side_effect=AssertionError("no fallback")),
    ):
        row = rollout.process_one(ITEM, str(tmp_path), "", search_mode="keenable", max_tool_turns=2)
    assert f"Keenable search HTTP {status}" in row["fail_reason"]
    assert row["hard"] == 0
    assert len(calls) == 1


def test_a_transport_timeout_raises_the_builtin_timeout_error(transport):
    transport[1]["timeout"] = True
    with pytest.raises(TimeoutError):
        keenable.keenable_search(["treasury"])


def test_one_deadline_covers_every_query_of_a_round(monkeypatch, transport):
    clock = iter([0.0, 0.0, 25.0])
    monkeypatch.setattr(keenable.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        keenable.keenable_search(["treasury", "declaration"], timeout=20)
    assert len(transport[0]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"results": [{"url": "file:///etc/passwd", "snippet": "bad"}]},
        {"results": "malformed"},
        {"results": [{"url": "https://example.org", "snippet": ["not", "text"]}]},
    ],
)
def test_bad_evidence_is_rejected(transport, payload):
    transport[1]["payload"] = payload
    with pytest.raises(RuntimeError, match="invalid"):
        keenable.keenable_search(["treasury"])


@pytest.mark.parametrize(("queries", "kwargs"), [([], {}), (["  "], {}), (["q"], {"timeout": 0}), (["q"], {"max_num_results": 51})])
def test_invalid_arguments_are_refused_before_any_request(transport, queries, kwargs):
    with pytest.raises(ValueError):
        keenable.keenable_search(queries, **kwargs)
    assert not transport[0]


# ── Follow-ups from the #368 review ──────────────────────────────────────────

def test_a_redirect_is_not_followed(monkeypatch, transport):
    # Following it would forward X-API-Key to another origin.
    monkeypatch.setenv("KEENABLE_API_KEY", "kn-test-key")
    transport[1]["response"] = httpx.Response(302, headers={"Location": "https://elsewhere.example/steal"})
    with pytest.raises(RuntimeError, match="Keenable search HTTP 302"):
        keenable.keenable_search(["treasury"])
    assert [str(request.url) for request, _ in transport[0]] == [keenable.KEYED_URL]


def test_a_non_json_body_fails_instead_of_returning_no_results(transport):
    transport[1]["response"] = httpx.Response(200, text="<html>maintenance</html>")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        keenable.keenable_search(["treasury"])


def test_a_key_echoed_in_an_error_body_is_redacted(tmp_path, monkeypatch, transport):
    monkeypatch.setenv("KEENABLE_API_KEY", "kn-secret-key")
    transport[1]["response"] = httpx.Response(401, text="invalid key kn-secret-key for this org")
    scripted_model(monkeypatch)
    row = rollout.process_one(ITEM, str(tmp_path), "", search_mode="keenable", max_tool_turns=2)
    assert "Keenable search HTTP 401" in row["fail_reason"]
    assert "[redacted]" in row["fail_reason"]
    assert "kn-secret-key" not in row["fail_reason"]
    assert "kn-secret-key" not in (tmp_path / "predictions" / ITEM["id"] / "conversation.json").read_text()


def test_a_key_straddling_the_error_cut_leaks_no_prefix(monkeypatch, transport):
    monkeypatch.setenv("KEENABLE_API_KEY", "kn-secret-key")
    transport[1]["response"] = httpx.Response(401, text="x" * 995 + "kn-secret-key")
    with pytest.raises(RuntimeError) as excinfo:
        keenable.keenable_search(["treasury"])
    assert "kn-se" not in str(excinfo.value)


@pytest.mark.parametrize("backend", ["claude_exec", "codex_exec", "azure_openai_responses"])
def test_keenable_refuses_a_non_chat_backend_before_any_request(tmp_path, monkeypatch, transport, backend):
    monkeypatch.setattr(rollout, "get_target_backend", lambda: backend)
    scripted_model(monkeypatch)
    row = rollout.process_one(ITEM, str(tmp_path), "", search_mode="keenable", max_tool_turns=2)
    assert "only supported with target_backend='openai_chat' or 'qwen_chat'" in row["fail_reason"]
    assert transport[0] == []


def test_a_response_that_ends_past_the_deadline_fails_the_round(monkeypatch, transport):
    # One query: checked before the request at t=0, the response lands at t=25.
    clock = iter([0.0, 0.0, 25.0])
    monkeypatch.setattr(keenable.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        keenable.keenable_search(["treasury"], timeout=20)
    assert len(transport[0]) == 1


@pytest.mark.parametrize("title", [None, "", "   ", 42, ["Treasury"]])
def test_a_missing_or_non_string_title_falls_back_to_the_url(transport, title):
    transport[1]["payload"] = {"results": [{"url": "https://example.org/t", "title": title, "snippet": "s"}]}
    assert "[1] https://example.org/t\nURL: https://example.org/t" in keenable.keenable_search(["treasury"])

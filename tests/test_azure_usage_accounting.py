"""Azure provider-usage accounting regressions: single-owner charging.

A backend that reports provider usage (AzureOpenAI / AzureResponses) must be
charged exactly once in ``_cached_call`` — with the provider's own token count,
never double-charged by the ``len//4`` length estimate, and never charged on a
cache hit. (The maintainer's reproduction: a 30-token provider usage was being
recorded as a 110-token length estimate because ``_call`` recorded usage and
``_cached_call`` then recorded the length estimate on top.)

Also covers the OpenCode error path routing through ``_record_delta``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from skillopt_sleep.backend import AzureOpenAIBackend, AzureResponsesBackend, OpenCodeCliBackend


class _ChatResp:
    def __init__(self, text, prompt_tokens, completion_tokens):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]
        self.usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class _FakeChatClient:
    """Scripted chat.completions.create returning a _ChatResp or raising."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _ResponsesResp:
    def __init__(self, text, input_tokens, output_tokens):
        self.output_text = text
        self.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


class _FakeResponsesClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _azure_chat(replies):
    be = AzureOpenAIBackend(deployment="gpt-5.5")
    be._client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeChatClient(replies)))
    return be


def _azure_responses(replies):
    be = AzureResponsesBackend(deployment="gpt-5.5", endpoints=["https://t.openai.azure.com/"])
    fake = SimpleNamespace(responses=_FakeResponsesClient(replies))
    be._next_endpoint = lambda: be.endpoints[0]
    be._client_for = lambda ep: fake
    return be


def test_azure_chat_single_call_charges_exact_usage():
    """A single Azure chat call must charge the provider usage, not len//4."""
    be = _azure_chat([_ChatResp("ok", 10, 20)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "x" * 400)
    assert out == "ok"
    # provider usage = 30; len//4 of 400+2 would be ~100 — must NOT be that.
    assert be._tokens == 30, f"expected exact provider usage 30, got {be._tokens}"
    assert be.token_delta() == 30


def test_azure_chat_empty_retry_then_success_accumulates():
    """Empty-response retry + success must accumulate usage across paid attempts."""
    be = _azure_chat([_ChatResp("", 7, 0), _ChatResp("ok", 10, 20)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "hello")
    assert out == "ok"
    assert be._tokens == 7 + 30, f"expected accumulated 37, got {be._tokens}"
    assert be.token_delta() == 37


def test_azure_chat_cache_hit_does_not_charge():
    """A cache hit must reset the call-local delta and leave the aggregate alone."""
    be = _azure_chat([_ChatResp("ok", 10, 20)])
    with mock.patch("time.sleep"):
        be._cached_call("k:1", "hello")
    assert be._tokens == 30
    assert be.token_delta() == 30
    with mock.patch("time.sleep"):
        out2 = be._cached_call("k:1", "hello")
    assert out2 == "ok"
    assert be._tokens == 30, "cache hit changed the aggregate"
    assert be.token_delta() == 0, "cache hit leaked the prior delta"


def test_azure_responses_single_call_charges_exact_usage():
    be = _azure_responses([_ResponsesResp("ok", 12, 18)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "x" * 400)
    assert out == "ok"
    assert be._tokens == 30, f"expected exact provider usage 30, got {be._tokens}"
    assert be.token_delta() == 30


def test_azure_responses_empty_retry_then_success_accumulates():
    be = _azure_responses([_ResponsesResp("", 5, 0), _ResponsesResp("ok", 12, 18)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "hello")
    assert out == "ok"
    assert be._tokens == 5 + 30, f"expected accumulated 35, got {be._tokens}"
    assert be.token_delta() == 35


def test_azure_responses_cache_hit_does_not_charge():
    be = _azure_responses([_ResponsesResp("ok", 12, 18)])
    with mock.patch("time.sleep"):
        be._cached_call("k:1", "hello")
    assert be._tokens == 30
    with mock.patch("time.sleep"):
        be._cached_call("k:1", "hello")
    assert be._tokens == 30
    assert be.token_delta() == 0


def test_opencode_error_path_uses_record_delta(monkeypatch):
    """The OpenCode error path must route prompt-only cost through _record_delta."""
    import contextlib
    from types import SimpleNamespace

    import skillopt_sleep.backend as bm
    from skillopt_sleep.backend import OpenCodeCliBackend

    b = OpenCodeCliBackend(model="", opencode_path="opencode", tool_replay=True)
    monkeypatch.setattr(bm, "_opencode_temporary_workspace", lambda *a, **k: contextlib.nullcontext())

    def _fail(*args, **kwargs):
        raise bm.OpenCodeError("boom", prompt_chars=100)

    monkeypatch.setattr(bm, "_prepare_opencode_replay_project", _fail)

    task = SimpleNamespace(intent="intent", context_excerpt="ctx")
    out, called = b.attempt_with_tools(task, skill="s", memory="m", tools=["search"])

    assert out == "" and called == []
    assert b.token_delta() == 100 // 4, "OpenCode error path did not route through _record_delta"
    assert b._tokens == 100 // 4, f"expected 25, got {b._tokens}"


def test_azure_chat_paid_empty_then_terminal_error_keeps_usage():
    """A paid empty response followed by exhausted exception retries must keep the
    exact provider usage (7), not fall back to the len//4 length estimate."""
    be = _azure_chat([_ChatResp("", 7, 0)] + [RuntimeError("boom")] * 4)
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "x" * 400)
    assert out == ""
    assert be._tokens == 7, f"expected paid usage 7, got {be._tokens}"
    assert be.token_delta() == 7, f"expected call-local delta 7, got {be.token_delta()}"

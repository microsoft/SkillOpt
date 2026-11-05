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

_UNSET = object()


class _ChatResp:
    def __init__(self, text, prompt_tokens=None, completion_tokens=None, usage=_UNSET):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]
        if usage is _UNSET:
            usage = SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
        self.usage = usage


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
    def __init__(self, text, input_tokens=None, output_tokens=None, usage=_UNSET):
        self.output_text = text
        if usage is _UNSET:
            usage = SimpleNamespace(
                input_tokens=input_tokens, output_tokens=output_tokens
            )
        self.usage = usage


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


# --- unknown-vs-zero usage: a missing usage block must not read as a free call ---


def test_azure_chat_success_without_usage_uses_length_estimate():
    """usage=None on a successful response must charge the len//4 estimate.

    Reviewed defect: the backend flipped charged_in_call=True even when the SDK
    response carried no usage, so both the aggregate and the call-local delta
    became 0. Repro shape: 400-char prompt, 40-char reply, main = 110 tokens.
    """
    be = _azure_chat([_ChatResp("y" * 40, usage=None)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "x" * 400)
    assert out == "y" * 40
    assert be._tokens == 400 // 4 + 40 // 4, f"expected estimate 110, got {be._tokens}"
    assert be.token_delta() == 110


def test_azure_responses_success_without_usage_uses_length_estimate():
    be = _azure_responses([_ResponsesResp("y" * 40, usage=None)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "x" * 400)
    assert out == "y" * 40
    assert be._tokens == 110, f"expected estimate 110, got {be._tokens}"
    assert be.token_delta() == 110


def test_azure_chat_usage_stub_without_counts_uses_length_estimate():
    """A usage object whose token fields are all None carries no information."""
    be = _azure_chat([
        _ChatResp("y" * 40, usage=SimpleNamespace(prompt_tokens=None, completion_tokens=None)),
    ])
    with mock.patch("time.sleep"):
        be._cached_call("k:1", "x" * 400)
    assert be._tokens == 110
    assert be.token_delta() == 110


def test_azure_chat_reported_zero_usage_is_authoritative():
    """A reported zero must stay zero — the estimate must NOT be substituted."""
    be = _azure_chat([_ChatResp("ok", 0, 0)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "x" * 400)
    assert out == "ok"
    assert be._tokens == 0, f"reported zero was overridden: {be._tokens}"
    assert be.token_delta() == 0


def test_azure_responses_reported_zero_usage_is_authoritative():
    be = _azure_responses([_ResponsesResp("ok", 0, 0)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "x" * 400)
    assert out == "ok"
    assert be._tokens == 0, f"reported zero was overridden: {be._tokens}"
    assert be.token_delta() == 0


def test_azure_chat_keeps_known_partial_retry_usage():
    """A paid empty attempt keeps its usage even when the later success reports none."""
    be = _azure_chat([_ChatResp("", 7, 0), _ChatResp("ok", usage=None)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "hello")
    assert out == "ok"
    assert be._tokens == 7, f"known partial retry usage lost: {be._tokens}"
    assert be.token_delta() == 7


def test_azure_responses_keeps_known_partial_retry_usage():
    be = _azure_responses([_ResponsesResp("", 5, 0), _ResponsesResp("ok", usage=None)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "hello")
    assert out == "ok"
    assert be._tokens == 5, f"known partial retry usage lost: {be._tokens}"
    assert be.token_delta() == 5


def test_azure_chat_exhausted_empty_retries_without_usage_uses_length_estimate():
    """Five empty responses that all omit usage must not read as a free call.

    Reviewed defect: the exhausted-retry exit charged ``usage_total`` (0) and set
    ``charged_in_call``, so unknown usage became an authoritative zero. Unknown
    usage belongs to the len//4 estimate (main reported 110 for this shape).
    """
    be = _azure_chat([_ChatResp("", usage=None) for _ in range(5)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "x" * 400)
    assert out == ""
    assert be._tokens == 400 // 4, f"expected length estimate 100, got {be._tokens}"
    assert be.token_delta() == 100


def test_azure_responses_exhausted_empty_retries_without_usage_uses_length_estimate():
    be = _azure_responses([_ResponsesResp("", usage=None) for _ in range(5)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "x" * 400)
    assert out == ""
    assert be._tokens == 100, f"expected length estimate 100, got {be._tokens}"
    assert be.token_delta() == 100


def test_azure_chat_exhausted_empty_retries_with_reported_zero_stays_zero():
    """A reported zero on every exhausted attempt is authoritative, not unknown."""
    be = _azure_chat([_ChatResp("", 0, 0) for _ in range(5)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "x" * 400)
    assert out == ""
    assert be._tokens == 0, f"reported zero overridden by estimate: {be._tokens}"
    assert be.token_delta() == 0


def test_azure_responses_exhausted_empty_retries_with_reported_zero_stays_zero():
    be = _azure_responses([_ResponsesResp("", 0, 0) for _ in range(5)])
    with mock.patch("time.sleep"):
        out = be._cached_call("k:1", "x" * 400)
    assert out == ""
    assert be._tokens == 0, f"reported zero overridden by estimate: {be._tokens}"
    assert be.token_delta() == 0

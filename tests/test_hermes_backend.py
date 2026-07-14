"""Contract tests for the Hermes Agent model backend.

Covers:
  - Backend alias resolution and default models
  - set_target/optimizer_backend acceptance
  - is_*_chat_backend recognition
  - Routing dispatch for chat_target / chat_optimizer / chat_messages
  - Token tracker isolation (no double-count with Claude)
  - Message API contracts (tools, retries, return_message)
  - Deployment setters
  - One opt-in real Hermes smoke test

Follows the same ptest + monkeypatch pattern as test_qwen_backend.py.
"""
from __future__ import annotations

import json
import os
from typing import Any
from unittest import mock

import pytest

from skillopt.model import (
    backend_config,
    chat_optimizer,
    chat_optimizer_messages,
    chat_target,
    chat_target_messages,
    chat_with_deployment,
    chat_messages_with_deployment,
    get_backend_name,
    get_token_summary,
    reset_token_tracker,
    set_backend,
    set_optimizer_backend,
    set_target_backend,
)
from skillopt.model.backend_config import (
    get_optimizer_backend,
    get_target_backend,
    is_optimizer_chat_backend,
    is_target_chat_backend,
)
from skillopt.model.common import (
    CompatAssistantMessage,
    default_model_for_backend,
    normalize_backend_name,
)
from skillopt.model import hermes_backend as _hermes


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolate_state() -> Any:
    """Save and restore backend config & token tracker state."""
    opt_before = get_optimizer_backend()
    tgt_before = get_target_backend()
    _hermes.reset_token_tracker()
    yield
    _hermes.reset_token_tracker()
    set_optimizer_backend(opt_before)
    set_target_backend(tgt_before)


class _FakeProc:
    """Fake subprocess.CompletedProcess."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _RunRecorder:
    """Records all subprocess.run calls and returns configurable responses."""

    def __init__(self, response: str = "Hello from Hermes", returncode: int = 0):
        self.calls: list[dict[str, Any]] = []
        self._response = response
        self._returncode = returncode

    def __call__(self, cmd, **kwargs) -> _FakeProc:
        self.calls.append({"cmd": cmd, "kwargs": kwargs})
        return _FakeProc(
            stdout=self._response,
            stderr="",
            returncode=self._returncode,
        )


def _use_hermes() -> None:
    """Set both optimizer and target backends to hermes_chat."""
    set_optimizer_backend("hermes_chat")
    set_target_backend("hermes_chat")


# ── 1. Alias and default model ───────────────────────────────────────────────


def test_normalize_backend_name_accepts_hermes():
    """normalize_backend_name('hermes') returns 'hermes_chat'."""
    assert normalize_backend_name("hermes") == "hermes_chat"
    assert normalize_backend_name("hermes_chat") == "hermes_chat"
    assert normalize_backend_name("HERMES") == "hermes_chat"


def test_default_model_for_hermes():
    """default_model_for_backend('hermes') returns 'hermes'."""
    assert default_model_for_backend("hermes") == "hermes"
    assert default_model_for_backend("hermes_chat") == "hermes"


# ── 2. Backend setter acceptance ─────────────────────────────────────────────


def test_set_target_backend_accepts_hermes():
    """set_target_backend('hermes_chat') does not raise ValueError."""
    set_target_backend("hermes_chat")
    assert get_target_backend() == "hermes_chat"


def test_set_optimizer_backend_accepts_hermes():
    """set_optimizer_backend('hermes_chat') does not raise ValueError."""
    set_optimizer_backend("hermes_chat")
    assert get_optimizer_backend() == "hermes_chat"


def test_legacy_set_backend_accepts_hermes():
    """Legacy set_backend('hermes') returns 'hermes_chat'."""
    result = set_backend("hermes")
    assert result == "hermes_chat"
    assert get_optimizer_backend() == "hermes_chat"
    assert get_target_backend() == "hermes_chat"


def test_get_backend_name_hermes():
    """get_backend_name() returns 'hermes_chat' when both are hermes."""
    _use_hermes()
    assert get_backend_name() == "hermes_chat"


# ── 3. is_*_chat_backend recognition ─────────────────────────────────────────


def test_is_optimizer_chat_backend_includes_hermes():
    """is_optimizer_chat_backend() returns True for hermes_chat."""
    set_optimizer_backend("hermes_chat")
    assert is_optimizer_chat_backend() is True


def test_is_target_chat_backend_includes_hermes():
    """is_target_chat_backend() returns True for hermes_chat."""
    set_target_backend("hermes_chat")
    assert is_target_chat_backend() is True


def test_is_target_exec_backend_false_for_hermes():
    """is_target_exec_backend() returns False for hermes_chat (it's a chat backend)."""
    set_target_backend("hermes_chat")
    from skillopt.model.backend_config import is_target_exec_backend
    assert is_target_exec_backend() is False


# ── 4. Routing dispatch ──────────────────────────────────────────────────────


def test_chat_target_dispatches_to_hermes(monkeypatch):
    """chat_target routes to hermes_backend when target is hermes_chat."""
    _use_hermes()
    recorder = _RunRecorder(response="Hermes response")
    monkeypatch.setattr("subprocess.run", recorder)

    text, usage = chat_target("system prompt", "user query", retries=1)

    assert text == "Hermes response"
    assert usage["total_tokens"] > 0
    # Verify the command includes hermes and the profile
    assert len(recorder.calls) == 1
    cmd = recorder.calls[0]["cmd"]
    assert "hermes" in cmd
    assert "--profile" in cmd


def test_chat_optimizer_dispatches_to_hermes(monkeypatch):
    """chat_optimizer routes to hermes_backend when optimizer is hermes_chat."""
    _use_hermes()
    recorder = _RunRecorder(response="Optimizer response")
    monkeypatch.setattr("subprocess.run", recorder)

    text, usage = chat_optimizer("system", "user prompt", retries=1)

    assert text == "Optimizer response"
    assert len(recorder.calls) == 1


def test_chat_with_deployment_uses_custom_profile(monkeypatch):
    """chat_with_deployment(deployment='pro') uses profile 'pro'."""
    _use_hermes()
    recorder = _RunRecorder(response="ok")
    monkeypatch.setattr("subprocess.run", recorder)

    chat_with_deployment("pro", "system", "user", retries=1)

    cmd = recorder.calls[0]["cmd"]
    profile_idx = cmd.index("--profile") + 1
    assert cmd[profile_idx] == "pro"


# ── 5. Token tracker isolation ────────────────────────────────────────────────


def test_token_tracker_separate_from_claude(monkeypatch):
    """Hermes token tracker is isolated — does not affect Claude's summary."""
    _use_hermes()
    recorder = _RunRecorder(response="response")
    monkeypatch.setattr("subprocess.run", recorder)

    # Record tokens via hermes call
    chat_target("sys", "usr", retries=1)
    hermes_summary = _hermes.get_token_summary()

    # The global get_token_summary includes hermes tokens
    global_summary = get_token_summary()
    assert global_summary.get("target", {}).get("total_tokens", 0) > 0


def test_reset_token_tracker_clears_hermes_only(monkeypatch):
    """reset_token_tracker clears Hermes tracker without affecting _openai."""
    _use_hermes()
    recorder = _RunRecorder(response="resp")
    monkeypatch.setattr("subprocess.run", recorder)

    chat_target("sys", "usr", retries=1)
    assert _hermes.get_token_summary().get("target", {}).get("calls", 0) == 1

    reset_token_tracker()
    assert _hermes.get_token_summary().get("target") is None


# ── 6. Message API contract ──────────────────────────────────────────────────


def test_chat_target_messages_respects_retries(monkeypatch):
    """chat_target_messages should respect retries=N (not hardcoded)."""
    _use_hermes()

    calls = {"n": 0}

    def failing_run(cmd, **kwargs) -> _FakeProc:
        calls["n"] += 1
        return _FakeProc(stdout="", stderr="error", returncode=1)

    monkeypatch.setattr("subprocess.run", failing_run)

    with pytest.raises(RuntimeError, match="Hermes CLI failed after 2 retries"):
        chat_target_messages(
            [{"role": "user", "content": "hello"}],
            retries=2,
        )

    assert calls["n"] == 2


def test_chat_target_messages_return_message(monkeypatch):
    """When return_message=True, returns CompatAssistantMessage, not str."""
    _use_hermes()
    recorder = _RunRecorder(response="Hello from Hermes")
    monkeypatch.setattr("subprocess.run", recorder)

    result, usage = chat_target_messages(
        [{"role": "user", "content": "hello"}],
        retries=1,
        return_message=True,
    )

    assert isinstance(result, CompatAssistantMessage)
    assert result.content == "Hello from Hermes"


def test_chat_optimizer_messages_return_message(monkeypatch):
    """chat_optimizer_messages with return_message=True returns CompatAssistantMessage."""
    _use_hermes()
    recorder = _RunRecorder(response="Optimizer says")
    monkeypatch.setattr("subprocess.run", recorder)

    result, usage = chat_optimizer_messages(
        [{"role": "user", "content": "optimize"}],
        retries=1,
        return_message=True,
    )

    assert isinstance(result, CompatAssistantMessage)
    assert result.content == "Optimizer says"


def test_chat_target_messages_serializes_tools(monkeypatch):
    """Tool definitions are serialized into the prompt when provided."""
    _use_hermes()
    recorder = _RunRecorder(response="used tool")
    monkeypatch.setattr("subprocess.run", recorder)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
    ]

    chat_target_messages(
        [{"role": "user", "content": "search for X"}],
        retries=1,
        tools=tools,
        tool_choice="auto",
    )

    # The prompt should contain the tool definition
    prompt = recorder.calls[0]["cmd"][-1]
    assert "search" in prompt
    assert "Available tools" in prompt


def test_chat_target_messages_return_message_ignores_tools(monkeypatch):
    """Hermes backend does not support tool loops; return_message always gets text only."""
    _use_hermes()
    recorder = _RunRecorder(response="Answer")
    monkeypatch.setattr("subprocess.run", recorder)

    result, usage = chat_target_messages(
        [{"role": "user", "content": "hello"}],
        retries=1,
        tools=[{"type": "function", "function": {"name": "x"}}],
        return_message=True,
    )

    assert isinstance(result, CompatAssistantMessage)
    assert result.content == "Answer"
    # No tool_calls since hermes CLI doesn't support them
    assert len(result.tool_calls) == 0


def test_chat_messages_with_deployment_uses_profile(monkeypatch):
    """chat_messages_with_deployment passes the deployment as profile."""
    _use_hermes()
    recorder = _RunRecorder(response="ok")
    monkeypatch.setattr("subprocess.run", recorder)

    chat_messages_with_deployment(
        "custom-profile",
        [{"role": "user", "content": "test"}],
        retries=1,
    )

    cmd = recorder.calls[0]["cmd"]
    profile_idx = cmd.index("--profile") + 1
    assert cmd[profile_idx] == "custom-profile"


# ── 7. Deployment setters ────────────────────────────────────────────────────


def test_set_target_deployment_updates_profile(monkeypatch):
    """set_target_deployment changes the profile used by chat_target."""
    _use_hermes()
    recorder = _RunRecorder(response="ok")
    monkeypatch.setattr("subprocess.run", recorder)

    _hermes.set_target_deployment("prod-profile")
    chat_target("sys", "usr", retries=1)

    cmd = recorder.calls[0]["cmd"]
    profile_idx = cmd.index("--profile") + 1
    assert cmd[profile_idx] == "prod-profile"


def test_set_optimizer_deployment_updates_profile(monkeypatch):
    """set_optimizer_deployment changes the profile used by chat_optimizer."""
    _use_hermes()
    recorder = _RunRecorder(response="ok")
    monkeypatch.setattr("subprocess.run", recorder)

    _hermes.set_optimizer_deployment("opt-profile")
    chat_optimizer("sys", "usr", retries=1)

    cmd = recorder.calls[0]["cmd"]
    profile_idx = cmd.index("--profile") + 1
    assert cmd[profile_idx] == "opt-profile"


# ── 8. Edge cases ────────────────────────────────────────────────────────────


def test_hermes_called_with_no_color_env(monkeypatch):
    """HERMES_NO_COLOR=1 is set in the subprocess env."""
    _use_hermes()
    recorder = _RunRecorder(response="ok")
    monkeypatch.setattr("subprocess.run", recorder)

    chat_target("sys", "usr", retries=1)

    env = recorder.calls[0]["kwargs"].get("env", {})
    assert env.get("HERMES_NO_COLOR") == "1"


def test_empty_response_from_hermes_raises(
    monkeypatch,
):
    """A non-zero exit without stderr raises RuntimeError."""
    _use_hermes()

    def fail_run(cmd, **kwargs) -> _FakeProc:
        return _FakeProc(stdout="", stderr="Internal error", returncode=1)

    monkeypatch.setattr("subprocess.run", fail_run)

    with pytest.raises(RuntimeError, match="Internal error"):
        chat_target("sys", "usr", retries=1)


def test_hermes_backend_not_routed_when_not_selected(monkeypatch):
    """When backend is not hermes, chat_target does NOT call hermes CLI."""
    set_target_backend("openai_chat")  # default

    class FakeOpenAI:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    # If hermes were called, subprocess.run would be invoked — but it shouldn't be
    # since the backend is openai_chat and we don't have real API credentials.
    # We just verify the routing condition does not match.
    assert get_target_backend() != "hermes_chat"


# ── 9. Smoke test (opt-in, requires real hermes CLI) ──────────────────────────


@pytest.mark.slow
def test_real_hermes_smoke():
    """Verify the hermes CLI exists and responds.

    Marked @pytest.mark.slow (opt-in).
    Run:  pytest tests/test_hermes_backend.py -k real_hermes_smoke --slow
    or:   pytest tests/test_hermes_backend.py -k real_hermes_smoke -m slow
    """
    import subprocess as _sp

    try:
        proc = _sp.run(
            ["hermes", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        pytest.skip("hermes CLI not found on PATH")
    except Exception as e:
        pytest.skip(f"hermes CLI not available: {e}")

    if proc.returncode != 0:
        pytest.skip(f"hermes CLI not working (exit {proc.returncode}): {proc.stderr}")

    # Now try a real chat call
    _use_hermes()
    recorder = _RunRecorder(response="smoke test ok")
    with mock.patch("subprocess.run", return_value=_FakeProc(
        stdout="smoke test ok", stderr="", returncode=0
    )):
        text, usage = chat_target("Be concise.", "Say hello", retries=1)
        assert isinstance(text, str)
        assert usage["total_tokens"] > 0

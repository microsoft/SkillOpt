"""A failed `claude -p` call must not be used as the model's answer.

In text mode the Claude CLI prints an error result to stdout (for example
"API Error: 529 ...") and exits 1. ClaudeCliBackend used to return that text,
so it was cached under the attempt key and scored like a real response.
"""

from __future__ import annotations

import subprocess

import pytest

from skillopt_sleep.backend import ClaudeCliBackend
from skillopt_sleep.replay import replay_one
from skillopt_sleep.types import TaskRecord

API_ERROR = (
    'API Error: 529 {"type":"error","error":'
    '{"type":"overloaded_error","message":"Overloaded"}}'
)


class _FakeClaude:
    """Stands in for subprocess.run and replays (stdout, returncode) pairs."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def __call__(self, cmd, **kwargs):
        self.calls += 1
        stdout, returncode = self.results.pop(0)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")


@pytest.fixture
def claude(monkeypatch):
    # An API key selects --bare, so no --help probe runs before the call.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return ClaudeCliBackend(claude_path="claude", timeout=5)


def _task():
    return TaskRecord(
        id="t1", project="/p", intent="What is 6*7? Reply with the number only.",
        reference_kind="exact", reference="42",
    )


def test_nonzero_exit_returns_no_answer_and_records_the_error(claude, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _FakeClaude((API_ERROR + "\n", 1)))

    assert claude._call("prompt") == ""
    assert claude.last_call_error.startswith("Claude CLI exited 1: API Error: 529")


def test_failed_call_is_not_cached_so_the_rescore_calls_the_cli_again(claude, monkeypatch):
    fake = _FakeClaude((API_ERROR + "\n", 1), ("42\n", 0))
    monkeypatch.setattr(subprocess, "run", fake)

    first = replay_one(claude, _task(), skill="", memory="")
    assert first.response == ""
    assert claude._cache == {}

    second = replay_one(claude, _task(), skill="", memory="")
    assert fake.calls == 2
    assert second.response == "42"
    assert second.hard == 1.0
    assert claude.last_call_error == ""


def test_tool_route_error_clears_when_the_next_call_succeeds(claude, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _FakeClaude(("Execution error\n", 1), ("42\n", 0)))

    assert claude.attempt_with_tools(_task(), "", "", tools=["search"])[0] == ""
    assert claude.last_call_error
    assert claude.attempt_with_tools(_task(), "", "", tools=["search"])[0] == "42"
    assert claude.last_call_error == ""


def test_tool_route_nonzero_exit_returns_no_answer(claude, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _FakeClaude(("Execution error\n", 1)))

    resp, _called = claude.attempt_with_tools(_task(), "", "", tools=["search"])

    assert resp == ""
    assert claude.last_call_error == "Claude CLI exited 1: Execution error"


def test_error_detail_is_redacted(claude, monkeypatch):
    key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"
    monkeypatch.setattr(subprocess, "run", _FakeClaude((f"Invalid API key {key}\n", 1)))

    assert claude._call("prompt") == ""
    assert "Invalid API key" in claude.last_call_error
    assert key not in claude.last_call_error


def test_successful_call_is_unchanged(claude, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _FakeClaude(("42\n", 0)))

    assert claude._call("prompt") == "42"
    assert claude.last_call_error == ""

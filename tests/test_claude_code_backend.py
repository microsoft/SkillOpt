"""claude_code_exec optimizer backend: trace parsing, dispatch, persistence, retries.

Covers the four highest-risk, previously-untested points introduced with
``claude_code_backend`` (issue #233):

- ``parse_claude_trace_steps`` extracts text / tool_call / tool_result and drops
  init / thinking_tokens bookkeeping (``skillopt/model/codex_harness.py``).
- the dispatcher routes ``chat_optimizer`` to the claude_code branch when the
  optimizer backend is ``claude_code_exec``.
- ``_persist_claude_artifacts`` writes ``claude_trace_steps.txt`` for the reflector.
- a non-JSON structured reply is retried and then surfaces as ``RuntimeError``.

Plus a gating regression: ``fmt_minibatch_trajectories`` only injects
``#### Claude Trace Steps`` when ``REFLACT_CLAUDE_TRACE_TO_OPTIMIZER == "1"``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from skillopt.gradient.reflect import fmt_minibatch_trajectories
from skillopt.model import codex_harness
from skillopt.model.codex_harness import (
    _json_dumps,
    _persist_claude_artifacts,
    format_claude_trace_steps,
    parse_claude_trace_steps,
)


class _OpenAIClientStub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def _install_openai_stub() -> None:
    if "openai" in sys.modules or importlib.util.find_spec("openai") is not None:
        return
    openai_stub = types.ModuleType("openai")
    openai_stub.AzureOpenAI = _OpenAIClientStub
    openai_stub.OpenAI = _OpenAIClientStub
    sys.modules["openai"] = openai_stub


@pytest.fixture(autouse=True)
def isolate_backend_state() -> Iterator[None]:
    _install_openai_stub()
    from skillopt.model import backend_config

    optimizer_backend = backend_config.get_optimizer_backend()
    target_backend = backend_config.get_target_backend()
    env = {
        key: os.environ.get(key)
        for key in (
            "OPTIMIZER_BACKEND",
            "TARGET_BACKEND",
            "OPTIMIZER_DEPLOYMENT",
            "TARGET_DEPLOYMENT",
        )
    }
    yield
    backend_config.set_optimizer_backend(optimizer_backend)
    backend_config.set_target_backend(target_backend)
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _sdk_payload() -> str:
    """Two SDK attempt blocks: bookkeeping noise + real steps + tool result."""
    block1 = {
        "messages": [
            {"subtype": "init", "content": []},
            {"subtype": "thinking_tokens", "content": []},
            {"data": {"type": "system", "text": "system banner"}, "content": []},
            {
                "content": [
                    {"type": "text", "text": "Let me read the task."},
                    {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "task.md"}},
                ],
                "data": {"type": "assistant"},
            },
            {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [{"type": "text", "content": "X" * 300}],
                        "is_error": False,
                    }
                ],
                "data": {"type": "user"},
            },
            {"result": "THE ANSWER", "data": {"type": "result"}},
        ]
    }
    block2 = {
        "messages": [
            {
                "content": [{"type": "text", "text": "Final answer body"}],
                "data": {"type": "assistant"},
            }
        ]
    }
    return (
        _json_dumps(block1)
        + "\n===== CLAUDE SDK ATTEMPT 2 =====\n"
        + _json_dumps(block2)
    )


def test_parse_claude_trace_steps_extracts_and_filters() -> None:
    steps = parse_claude_trace_steps(_sdk_payload())

    types_seen = [step["type"] for step in steps]
    # init / thinking_tokens / system bookkeeping are dropped.
    assert types_seen == ["text", "tool_call", "tool_result", "text", "text"]

    # Indices are renumbered sequentially across attempt blocks.
    assert [step["index"] for step in steps] == [1, 2, 3, 4, 5]

    assert steps[0]["summary"] == "Let me read the task."
    assert steps[1]["summary"] == "Read task.md"
    # tool_result is truncated to 200 chars + a [+N chars] trailer.
    assert steps[2]["summary"].startswith("X" * 200)
    assert "[+100 chars]" in steps[2]["summary"]
    assert steps[3]["summary"] == "THE ANSWER"
    assert steps[4]["summary"] == "Final answer body"


def test_parse_claude_trace_steps_marks_errors() -> None:
    block = {
        "messages": [
            {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_9",
                        "content": [{"type": "text", "content": "boom"}],
                        "is_error": True,
                    }
                ]
            }
        ]
    }
    steps = parse_claude_trace_steps(_json_dumps(block))
    assert len(steps) == 1
    assert steps[0]["type"] == "tool_result"
    assert steps[0]["summary"] == "[error] boom"


def test_format_claude_trace_steps_truncates_total() -> None:
    text = format_claude_trace_steps(_sdk_payload(), max_chars=40)
    trailer = "\n...[claude trace steps truncated]..."
    assert text.endswith(trailer)
    assert text == text[:40] + trailer


def test_persist_claude_artifacts_writes_trace_steps(tmp_path) -> None:
    work_dir = tmp_path / "pred" / "work"
    work_dir.mkdir(parents=True)

    _persist_claude_artifacts(str(work_dir), _sdk_payload(), "response")

    steps_path = tmp_path / "pred" / "claude_trace_steps.txt"
    assert steps_path.exists()
    content = steps_path.read_text(encoding="utf-8")
    assert content.strip()
    # text step is index 1, so the tool_call is index 2.
    assert "[2] tool_call: Read task.md" in content


def test_chat_optimizer_routes_to_claude_code_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillopt.model import backend_config, claude_code_backend
    from skillopt.model import azure_openai

    claude_calls: list[dict[str, Any]] = []

    def fake_claude_optimizer(**kwargs: Any) -> tuple[str, dict[str, int]]:
        claude_calls.append(kwargs)
        return "claude result", {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }

    def fail_openai_optimizer(**_kwargs: Any) -> tuple[str, dict[str, int]]:
        raise AssertionError("openai optimizer should not be called for claude_code_exec")

    monkeypatch.setattr(claude_code_backend, "chat_optimizer", fake_claude_optimizer)
    monkeypatch.setattr(azure_openai, "chat_optimizer", fail_openai_optimizer)
    backend_config.set_optimizer_backend("claude_code_exec")

    from skillopt.model import chat_optimizer

    text, usage = chat_optimizer("system", "user", retries=1, timeout=5)

    assert text == "claude result"
    assert usage["total_tokens"] == 3
    assert claude_calls[0]["system"] == "system"
    assert claude_calls[0]["user"] == "user"
    assert claude_calls[0]["timeout"] == 5


def test_reasoning_effort_forwarded_to_run_claude_code_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillopt.model import claude_code_backend

    calls: list[dict[str, Any]] = []

    def fake_chat(**kwargs: Any) -> tuple[str, dict[str, int]]:
        calls.append(kwargs)
        return "plain reply", {"prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(claude_code_backend, "run_claude_code_chat", fake_chat)
    claude_code_backend.set_reasoning_effort("high")
    try:
        text, _usage = claude_code_backend.chat_optimizer("s", "u", retries=1)
    finally:
        claude_code_backend.set_reasoning_effort(None)

    assert text == "plain reply"
    assert calls[0]["effort"] == "high"


def test_reasoning_effort_param_beats_module_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillopt.model import claude_code_backend

    calls: list[dict[str, Any]] = []

    def fake_chat(**kwargs: Any) -> tuple[str, dict[str, int]]:
        calls.append(kwargs)
        return "plain reply", {"prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(claude_code_backend, "run_claude_code_chat", fake_chat)
    claude_code_backend.set_reasoning_effort("low")
    try:
        claude_code_backend.chat_optimizer(
            "s", "u", retries=1, reasoning_effort="max"
        )
    finally:
        claude_code_backend.set_reasoning_effort(None)

    assert calls[0]["effort"] == "max"


def test_claude_code_backend_retry_on_bad_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillopt.model import claude_code_backend

    monkeypatch.setattr(
        claude_code_backend,
        "run_claude_code_chat",
        lambda **kwargs: ("this is not json", {"prompt_tokens": 3, "completion_tokens": 4}),
    )

    claude_code_backend.reset_token_tracker()
    try:
        # structured output (tools set) forces a json.loads on the reply; a
        # non-JSON reply must be retried, then surfaced as RuntimeError.
        with pytest.raises(RuntimeError, match="failed after 2 retries"):
            claude_code_backend.chat_optimizer_messages(
                [{"role": "user", "content": "hi"}],
                retries=2,
                tools=[{"name": "lookup"}],
            )

        summary = claude_code_backend.get_token_summary()
        optimizer = summary["optimizer"]
        assert optimizer["calls"] == 2
        assert optimizer["prompt_tokens"] == 6
        assert optimizer["completion_tokens"] == 8
    finally:
        claude_code_backend.reset_token_tracker()


@pytest.mark.parametrize(
    ("target_backend", "config_on", "expect_codex", "expect_claude"),
    [
        ("claude_code_exec", True, "0", "1"),
        ("claude_code_exec", False, "0", "0"),
        ("codex_exec", True, "1", "0"),
        ("codex_exec", False, "0", "0"),
        ("openai_chat", True, "0", "0"),
    ],
)
def test_trainer_configures_trace_gates(
    monkeypatch: pytest.MonkeyPatch,
    target_backend: str,
    config_on: bool,
    expect_codex: str,
    expect_claude: str,
) -> None:
    from skillopt.engine.trainer import _configure_trace_to_optimizer_gates

    monkeypatch.delenv("REFLACT_CODEX_TRACE_TO_OPTIMIZER", raising=False)
    monkeypatch.delenv("REFLACT_CLAUDE_TRACE_TO_OPTIMIZER", raising=False)
    _configure_trace_to_optimizer_gates(
        target_backend,
        {"codex_trace_to_optimizer": config_on, "claude_trace_to_optimizer": config_on},
    )
    assert os.environ["REFLACT_CODEX_TRACE_TO_OPTIMIZER"] == expect_codex
    assert os.environ["REFLACT_CLAUDE_TRACE_TO_OPTIMIZER"] == expect_claude


@pytest.mark.parametrize(
    ("gate_value", "expect_injected"),
    [("0", False), ("1", True)],
)
def test_claude_trace_steps_gated_in_fmt_minibatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    gate_value: str,
    expect_injected: bool,
) -> None:
    tid = "tid0"
    pred_dir = tmp_path / "predictions"
    (pred_dir / tid).mkdir(parents=True)
    (pred_dir / tid / "conversation.json").write_text(
        json.dumps([{"role": "assistant", "content": "hi"}]),
        encoding="utf-8",
    )
    (pred_dir / tid / "claude_trace_steps.txt").write_text(
        "[1] tool_call: Read task.md",
        encoding="utf-8",
    )

    monkeypatch.setenv("REFLACT_CLAUDE_TRACE_TO_OPTIMIZER", gate_value)

    formatted = fmt_minibatch_trajectories(
        [{"id": tid, "task_description": "t", "task_type": "q"}],
        str(pred_dir),
    )

    assert ("#### Claude Trace Steps" in formatted) is expect_injected

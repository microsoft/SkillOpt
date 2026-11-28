"""A single user request must reach the Claude CLI as written, not as a transcript."""
from __future__ import annotations

import json
from types import SimpleNamespace

from skillopt.model import claude_backend
from skillopt.model.claude_backend import _build_prompt_from_messages

_SYSTEM = "You review Python code."
_REQUEST = "Fix the bug:\n\ndef total(xs):\n    return sum(xs) + 1\n"
_TOOLS = [
    {
        "type": "function",
        "function": {"name": "run", "description": "Run code.", "parameters": {"type": "object"}},
    }
]


def test_single_user_turn_is_sent_verbatim() -> None:
    system, prompt, attachments = _build_prompt_from_messages(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": _REQUEST}]
    )

    assert system == _SYSTEM
    # Previously: "History:\n- User: Fix the bug:\n  \n  def total(xs):\n      return ..."
    # with every line after the first indented and "Answer the latest user request." appended.
    assert prompt == _REQUEST
    assert attachments == []


def test_chat_target_feeds_the_request_to_the_cli_unchanged(monkeypatch) -> None:
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["stdin"] = kwargs["input"]
        system_file = cmd[cmd.index("--append-system-prompt-file") + 1]
        with open(system_file, encoding="utf-8") as fh:
            captured["system"] = fh.read()
        payload = {"type": "result", "result": "ok", "usage": {}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(claude_backend.subprocess, "run", fake_run)

    text, _ = claude_backend.chat_target(_SYSTEM, _REQUEST, retries=1)

    assert text == "ok"
    assert captured["stdin"] == _REQUEST
    assert captured["system"] == _SYSTEM


def test_multi_turn_history_keeps_the_transcript_form() -> None:
    _, prompt, _ = _build_prompt_from_messages(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
    )

    assert prompt.startswith("History:\n- User: First question\n- Assistant: First answer")
    assert prompt.endswith("Answer the latest user request.")


def test_tool_and_structured_calls_keep_their_instructions() -> None:
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": _REQUEST}]

    _, with_tools, _ = _build_prompt_from_messages(messages, tools=_TOOLS, structured_output=True)
    _, structured, _ = _build_prompt_from_messages(messages, structured_output=True)

    assert with_tools.startswith("Available compatibility tools:")
    assert "History:\n- User: Fix the bug:" in with_tools
    assert structured.endswith("Return only JSON matching the provided schema.")


def test_request_that_starts_with_a_slash_keeps_the_transcript_form() -> None:
    # `claude -p` runs a prompt such as "/help" as a slash command without
    # calling the model, so a request starting with "/" stays wrapped.
    _, prompt, _ = _build_prompt_from_messages([{"role": "user", "content": "/help me read this log"}])

    assert prompt == "History:\n- User: /help me read this log\n\nAnswer the latest user request."


def test_empty_user_turn_never_yields_an_empty_prompt() -> None:
    _, prompt, _ = _build_prompt_from_messages([{"role": "user", "content": "   "}])

    assert prompt.strip()

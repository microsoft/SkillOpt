"""The claude_chat structured-output path must match the Claude CLI's flags and result shape."""
from __future__ import annotations

import json
from types import SimpleNamespace

from skillopt.model import claude_backend

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
        },
    }
]
_MESSAGES = [
    {"role": "system", "content": "You answer questions."},
    {"role": "user", "content": "What year was it signed?"},
]


def _install_fake_cli(monkeypatch, envelope: dict) -> dict:
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope) + "\n", stderr="")

    monkeypatch.setattr(claude_backend.subprocess, "run", fake_run)
    return captured


def test_structured_call_passes_json_schema_and_reads_structured_output(monkeypatch) -> None:
    # Shape of a Claude CLI 2.1.296 `-p --output-format json --json-schema ...` result:
    # the validated object is in `structured_output`, and `result` holds the same JSON as text.
    structured = {
        "content": "Looking it up.",
        "tool_calls": [{"name": "lookup", "arguments": '{"key": "year"}'}],
    }
    captured = _install_fake_cli(
        monkeypatch,
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": json.dumps(structured),
            "structured_output": structured,
            "usage": {"input_tokens": 7, "output_tokens": 5},
        },
    )

    message, usage = claude_backend.chat_target_messages(
        _MESSAGES, tools=_TOOLS, return_message=True, retries=1
    )

    cmd = captured["cmd"]
    # The CLI rejects `--schema` with "error: unknown option '--schema'".
    assert "--schema" not in cmd
    schema = json.loads(cmd[cmd.index("--json-schema") + 1])
    assert schema == claude_backend._assistant_message_schema()
    assert message.content == "Looking it up."
    assert [(call.function.name, call.function.arguments) for call in message.tool_calls] == [
        ("lookup", '{"key": "year"}')
    ]
    assert usage == {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}


def test_structured_call_without_structured_output_keeps_the_text_reply(monkeypatch) -> None:
    _install_fake_cli(
        monkeypatch,
        {"type": "result", "subtype": "success", "result": "1776", "usage": {}},
    )

    message, _ = claude_backend.chat_target_messages(_MESSAGES, return_message=True, retries=1)

    assert message.content == "1776"
    assert message.tool_calls == []


def test_plain_call_sends_no_schema(monkeypatch) -> None:
    captured = _install_fake_cli(
        monkeypatch,
        {"type": "result", "subtype": "success", "result": "1776", "usage": {}},
    )

    text, _ = claude_backend.chat_target("You answer questions.", "What year was it signed?", retries=1)

    assert text == "1776"
    assert "--json-schema" not in captured["cmd"]
    assert "--schema" not in captured["cmd"]

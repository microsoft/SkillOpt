"""Tests for model common utilities: usage parsing and compat message normalization."""
from __future__ import annotations

from types import SimpleNamespace

from skillopt.model.common import (
    TokenTracker,
    compat_message_from_chat_message,
    compat_message_from_responses_output,
    usage_from_openai_usage,
    usage_from_responses_usage,
)


class TestUsageParsing:
    def test_usage_from_openai_usage_dict(self) -> None:
        usage = {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}
        info = usage_from_openai_usage(usage)
        assert info == {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}

    def test_usage_from_openai_usage_dict_derived_total(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        info = usage_from_openai_usage(usage)
        assert info == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_usage_from_openai_usage_object(self) -> None:
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=15, total_tokens=20)
        info = usage_from_openai_usage(usage)
        assert info == {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}

    def test_usage_from_openai_usage_none(self) -> None:
        assert usage_from_openai_usage(None) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_usage_from_responses_usage_dict(self) -> None:
        usage = {"input_tokens": 8, "output_tokens": 16}
        info = usage_from_responses_usage(usage)
        assert info == {"prompt_tokens": 8, "completion_tokens": 16, "total_tokens": 24}

    def test_usage_from_responses_usage_object(self) -> None:
        usage = SimpleNamespace(input_tokens=7, output_tokens=14)
        info = usage_from_responses_usage(usage)
        assert info == {"prompt_tokens": 7, "completion_tokens": 14, "total_tokens": 21}

    def test_usage_from_responses_usage_none(self) -> None:
        assert usage_from_responses_usage(None) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


class TestCompatMessages:
    def test_compat_message_from_dict(self) -> None:
        msg = {
            "content": "Running function",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "query_db", "arguments": '{"q": "test"}'},
                }
            ],
        }
        compat = compat_message_from_chat_message(msg)
        assert compat.content == "Running function"
        assert len(compat.tool_calls) == 1
        assert compat.tool_calls[0].id == "call_1"
        assert compat.tool_calls[0].function.name == "query_db"
        assert compat.tool_calls[0].function.arguments == '{"q": "test"}'

    def test_compat_message_from_object(self) -> None:
        msg = SimpleNamespace(
            content="Hello",
            tool_calls=[
                SimpleNamespace(
                    id="call_2",
                    function=SimpleNamespace(name="search", arguments='{"k": "v"}'),
                )
            ],
        )
        compat = compat_message_from_chat_message(msg)
        assert compat.content == "Hello"
        assert len(compat.tool_calls) == 1
        assert compat.tool_calls[0].id == "call_2"
        assert compat.tool_calls[0].function.name == "search"

    def test_compat_message_from_responses_output_dict(self) -> None:
        output = [
            {"type": "message", "content": [{"type": "text", "text": "Thought: "}]},
            {"type": "message", "content": [{"type": "text", "text": "Ready."}]},
            {
                "type": "function_call",
                "call_id": "c123",
                "name": "lookup",
                "arguments": '{"term": "test"}',
            },
        ]
        compat = compat_message_from_responses_output(output)
        assert compat.content == "Thought: Ready."
        assert len(compat.tool_calls) == 1
        assert compat.tool_calls[0].id == "c123"
        assert compat.tool_calls[0].function.name == "lookup"

    def test_compat_message_from_responses_output_empty_or_none(self) -> None:
        assert compat_message_from_responses_output([]).content == ""
        assert compat_message_from_responses_output(None).content == ""  # type: ignore[arg-type]


class TestTokenTracker:
    def test_record_with_none_tokens_does_not_crash(self) -> None:
        tracker = TokenTracker()
        tracker.record("test_stage", None, None)  # type: ignore[arg-type]
        summary = tracker.summary()
        assert summary["test_stage"]["calls"] == 1
        assert summary["test_stage"]["prompt_tokens"] == 0
        assert summary["test_stage"]["completion_tokens"] == 0

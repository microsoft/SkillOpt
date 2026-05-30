"""Shared model utilities for ReflACT backends."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any


_RESPONSES_API_MODELS = {
    "gpt-5.3-codex",
    "gpt-5.1-codex",
    "gpt-5.2-codex",
    "gpt-5-codex",
    "codex-mini",
    "gpt-5.4-pro",
}

_BACKEND_DEFAULT_MODELS = {
    "azure_openai": "gpt-4o",
    "openai_chat": "gpt-4o",
    "codex": "gpt-4o",
    "codex_exec": "gpt-4o",
    "claude": "claude-sonnet-4-6",
    "claude_chat": "claude-sonnet-4-6",
    "claude_code_exec": "claude-sonnet-4-6",
    "qwen_chat": "Qwen/Qwen3.5-4B",
    "minimax_chat": "MiniMax-M2.7",
}

_BACKEND_ALIASES = {
    "azure": "azure_openai",
    "azure_openai": "azure_openai",
    "azure-openai": "azure_openai",
    "openai_chat": "openai_chat",
    "openai": "codex",
    "codex": "codex",
    "codex_exec": "codex_exec",
    "claude": "claude_chat",
    "claude_chat": "claude_chat",
    "claude_code_exec": "claude_code_exec",
    "anthropic": "claude_chat",
    "qwen": "qwen_chat",
    "qwen_chat": "qwen_chat",
    "minimax": "minimax_chat",
    "minimax_chat": "minimax_chat",
}


def normalize_backend_name(name: str | None) -> str:
    normalized = str(name or "").strip().lower()
    return _BACKEND_ALIASES.get(normalized, normalized or "azure_openai")


def default_model_for_backend(backend: str | None) -> str:
    return _BACKEND_DEFAULT_MODELS.get(
        normalize_backend_name(backend),
        _BACKEND_DEFAULT_MODELS["azure_openai"],
    )


def needs_responses_api(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return any(
        normalized == prefix or normalized.startswith(prefix + "-")
        for prefix in _RESPONSES_API_MODELS
    )


class TokenTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, int]] = {}

    def record(self, stage: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            if stage not in self._data:
                self._data[stage] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            entry = self._data[stage]
            entry["calls"] += 1
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens

    def summary(self) -> dict[str, dict[str, int]]:
        with self._lock:
            out: dict[str, dict[str, int]] = {}
            total_prompt = total_completion = total_calls = 0
            for stage, entry in sorted(self._data.items()):
                prompt_tokens = entry["prompt_tokens"]
                completion_tokens = entry["completion_tokens"]
                out[stage] = {
                    "calls": entry["calls"],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                total_prompt += prompt_tokens
                total_completion += completion_tokens
                total_calls += entry["calls"]
            out["_total"] = {
                "calls": total_calls,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            }
            return out

    def reset(self) -> None:
        with self._lock:
            self._data.clear()


tracker = TokenTracker()


@dataclass
class CompatToolFunction:
    name: str
    arguments: str

    def model_dump(self, mode: str = "json") -> dict[str, str]:
        del mode
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class CompatToolCall:
    id: str
    function: CompatToolFunction
    type: str = "function"

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        del mode
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.model_dump(),
        }


@dataclass
class CompatAssistantMessage:
    content: str
    tool_calls: list[CompatToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        del mode
        data: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [tool_call.model_dump() for tool_call in self.tool_calls]
        return data


def usage_from_openai_usage(usage: Any) -> dict[str, int]:
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def usage_from_responses_usage(usage: Any) -> dict[str, int]:
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
    completion_tokens = getattr(usage, "output_tokens", 0) or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def compat_message_from_chat_message(message: Any) -> CompatAssistantMessage:
    content = getattr(message, "content", "") or ""
    tool_calls = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        tool_calls.append(
            CompatToolCall(
                id=getattr(tool_call, "id", "") or "",
                function=CompatToolFunction(
                    name=getattr(function, "name", "") or "",
                    arguments=getattr(function, "arguments", "") or "{}",
                ),
            )
        )
    return CompatAssistantMessage(content=content, tool_calls=tool_calls)


def compat_message_from_responses_output(output: list[Any]) -> CompatAssistantMessage:
    text_parts: list[str] = []
    tool_calls: list[CompatToolCall] = []
    for item in output:
        item_type = getattr(item, "type", "") or ""
        if item_type == "function_call":
            raw_arguments = getattr(item, "arguments", None)
            if raw_arguments is None:
                raw_arguments = json.dumps(getattr(item, "input", {}) or {})
            tool_calls.append(
                CompatToolCall(
                    id=getattr(item, "call_id", "") or getattr(item, "id", "") or "",
                    function=CompatToolFunction(
                        name=getattr(item, "name", "") or "",
                        arguments=str(raw_arguments or "{}"),
                    ),
                )
            )
            continue
        if item_type != "message":
            continue
        for part in getattr(item, "content", []) or []:
            part_type = getattr(part, "type", "") or ""
            if part_type in {"output_text", "text"}:
                text_parts.append(getattr(part, "text", "") or "")
    return CompatAssistantMessage(content="".join(text_parts), tool_calls=tool_calls)

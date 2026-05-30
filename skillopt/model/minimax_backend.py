"""OpenAI-compatible MiniMax chat backend for the target path."""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from skillopt.model.common import (
    CompatAssistantMessage,
    CompatToolCall,
    CompatToolFunction,
    TokenTracker,
    default_model_for_backend,
)

BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
API_KEY = os.environ.get("MINIMAX_API_KEY", "")
TIMEOUT_SECONDS = float(os.environ.get("MINIMAX_TIMEOUT_SECONDS", "300") or 300)
MAX_TOKENS = int(os.environ.get("MINIMAX_MAX_TOKENS", "8000") or 8000)
TEMPERATURE: float | None = None
_raw_temperature = os.environ.get("MINIMAX_TEMPERATURE", "0.7").strip()
if _raw_temperature:
    TEMPERATURE = float(_raw_temperature)
ENABLE_THINKING = os.environ.get("MINIMAX_ENABLE_THINKING", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

TARGET_DEPLOYMENT = os.environ.get(
    "TARGET_DEPLOYMENT",
    default_model_for_backend("minimax_chat"),
)

_config_lock = threading.Lock()
tracker = TokenTracker()


def _chat_url() -> str:
    base = BASE_URL.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    return str(value)


def _usage_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _compat_message_from_payload(message: dict[str, Any], choice: dict[str, Any]) -> CompatAssistantMessage:
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    tool_calls: list[CompatToolCall] = []
    for index, tool_call in enumerate(message.get("tool_calls") or [], start=1):
        function = tool_call.get("function") or {}
        tool_calls.append(
            CompatToolCall(
                id=str(tool_call.get("id") or f"minimax_tool_{index}"),
                type=str(tool_call.get("type") or "function"),
                function=CompatToolFunction(
                    name=str(function.get("name") or ""),
                    arguments=str(function.get("arguments") or "{}"),
                ),
            )
        )
    return CompatAssistantMessage(
        content=content,
        tool_calls=tool_calls,
        metadata={
            "finish_reason": choice.get("finish_reason"),
            "choice0": _json_safe(choice),
        },
    )


def _post_chat_completion(payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(
        _chat_url(),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiniMax chat API returned HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"MiniMax chat API request failed: {e}") from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"MiniMax chat API returned non-JSON response: {raw[:1000]}") from e


def _chat_messages_impl(
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    retries: int,
    stage: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    deployment: str | None = None,
    timeout: float | None = None,
) -> tuple[Any, dict[str, int]]:
    payload: dict[str, Any] = {
        "model": deployment or TARGET_DEPLOYMENT,
        "messages": _json_safe(messages),
        "max_tokens": min(max_completion_tokens, MAX_TOKENS),
    }
    payload["chat_template_kwargs"] = {"enable_thinking": ENABLE_THINKING}
    if TEMPERATURE is not None:
        payload["temperature"] = TEMPERATURE
    if tools:
        payload["tools"] = _json_safe(tools)
        if tool_choice is not None:
            payload["tool_choice"] = _json_safe(tool_choice)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            data = _post_chat_completion(payload, timeout)
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"MiniMax chat API returned no choices: {data}")
            choice0 = choices[0]
            message = choice0.get("message") or {}
            text = message.get("content") or ""
            if not isinstance(text, str):
                text = json.dumps(text, ensure_ascii=False)
            usage_info = _usage_from_payload(data)
            tracker.record(stage, usage_info["prompt_tokens"], usage_info["completion_tokens"])
            if return_message:
                return _compat_message_from_payload(message, choice0), usage_info
            return text, usage_info
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"MiniMax chat call failed after {retries} retries: {last_err}")


def configure_minimax_chat(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | str | None = None,
    timeout_seconds: float | str | None = None,
    max_tokens: int | str | None = None,
    enable_thinking: bool | str | None = None,
) -> None:
    global BASE_URL, API_KEY, TEMPERATURE, TIMEOUT_SECONDS, MAX_TOKENS, ENABLE_THINKING
    with _config_lock:
        if base_url is not None:
            BASE_URL = str(base_url).strip() or BASE_URL
            os.environ["MINIMAX_BASE_URL"] = BASE_URL
        if api_key is not None:
            API_KEY = str(api_key).strip()
            os.environ["MINIMAX_API_KEY"] = API_KEY
        if temperature is not None:
            raw = str(temperature).strip()
            TEMPERATURE = float(raw) if raw else None
            os.environ["MINIMAX_TEMPERATURE"] = raw
        if timeout_seconds is not None:
            TIMEOUT_SECONDS = float(timeout_seconds)
            os.environ["MINIMAX_TIMEOUT_SECONDS"] = str(timeout_seconds)
        if max_tokens is not None:
            MAX_TOKENS = int(max_tokens)
            os.environ["MINIMAX_MAX_TOKENS"] = str(max_tokens)
        if enable_thinking is not None:
            if isinstance(enable_thinking, str):
                ENABLE_THINKING = enable_thinking.strip().lower() in {"1", "true", "yes", "on"}
            else:
                ENABLE_THINKING = bool(enable_thinking)
            os.environ["MINIMAX_ENABLE_THINKING"] = "true" if ENABLE_THINKING else "false"


def get_max_tokens() -> int:
    return MAX_TOKENS


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> tuple[str, dict[str, int]]:
    del reasoning_effort
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        timeout=timeout,
    )


def chat_target_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: float | None = None,
) -> tuple[Any, dict[str, int]]:
    del reasoning_effort
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    del effort


def set_target_deployment(deployment: str) -> None:
    global TARGET_DEPLOYMENT
    TARGET_DEPLOYMENT = deployment or default_model_for_backend("minimax_chat")
    os.environ["TARGET_DEPLOYMENT"] = TARGET_DEPLOYMENT
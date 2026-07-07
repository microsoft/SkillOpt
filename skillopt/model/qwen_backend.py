"""OpenAI-compatible Qwen chat backend for optimizer and target paths."""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class QwenChatConfig:
    base_url: str
    api_key: str
    timeout_seconds: float
    max_tokens: int
    temperature: float | None
    enable_thinking: bool
    deployment: str


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    return float(raw) if raw else None


def _parse_int(value: Any, default: int) -> int:
    if value is None:
        return default
    raw = str(value).strip()
    return int(raw) if raw else default


def _role_env(role: str, key: str, default: str) -> str:
    role_key = f"{role.upper()}_QWEN_CHAT_{key}"
    generic_key = f"QWEN_CHAT_{key}"
    return os.environ.get(role_key) or os.environ.get(generic_key) or default


def _initial_config(role: str) -> QwenChatConfig:
    role_upper = role.upper()
    deployment_env = "OPTIMIZER_DEPLOYMENT" if role == "optimizer" else "TARGET_DEPLOYMENT"
    return QwenChatConfig(
        base_url=_role_env(role, "BASE_URL", "http://localhost:8000/v1"),
        api_key=_role_env(role, "API_KEY", ""),
        timeout_seconds=float(_role_env(role, "TIMEOUT_SECONDS", "300") or 300),
        max_tokens=_parse_int(_role_env(role, "MAX_TOKENS", "8000"), 8000),
        temperature=_parse_optional_float(_role_env(role, "TEMPERATURE", "0.7")),
        enable_thinking=_parse_bool(_role_env(role, "ENABLE_THINKING", "false")),
        deployment=(
            os.environ.get(f"{role_upper}_QWEN_CHAT_MODEL")
            or os.environ.get("QWEN_CHAT_MODEL")
            or os.environ.get(deployment_env)
            or default_model_for_backend("qwen_chat")
        ),
    )


OPTIMIZER_CONFIG = _initial_config("optimizer")
TARGET_CONFIG = _initial_config("target")

_config_lock = threading.Lock()
tracker = TokenTracker()


def _chat_url(config: QwenChatConfig) -> str:
    base = config.base_url.rstrip("/")
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
                id=str(tool_call.get("id") or f"qwen_tool_{index}"),
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


def _post_chat_completion(
    payload: dict[str, Any],
    timeout: float | None,
    config: QwenChatConfig,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    req = urllib.request.Request(
        _chat_url(config),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or config.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen chat API returned HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Qwen chat API request failed: {e}") from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Qwen chat API returned non-JSON response: {raw[:1000]}") from e


def _chat_messages_impl(
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    retries: int,
    stage: str,
    *,
    role: str,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    deployment: str | None = None,
    timeout: float | None = None,
) -> tuple[Any, dict[str, int]]:
    config = OPTIMIZER_CONFIG if role == "optimizer" else TARGET_CONFIG
    payload: dict[str, Any] = {
        "model": deployment or config.deployment,
        "messages": _json_safe(messages),
        "max_tokens": min(max_completion_tokens, config.max_tokens),
    }
    # Always send an explicit value: omitting it leaves thinking mode to the
    # vLLM server / chat-template default, which varies across deployments (#90).
    payload["chat_template_kwargs"] = {"enable_thinking": bool(config.enable_thinking)}
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    if tools:
        payload["tools"] = _json_safe(tools)
        if tool_choice is not None:
            payload["tool_choice"] = _json_safe(tool_choice)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            data = _post_chat_completion(payload, timeout, config)
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"Qwen chat API returned no choices: {data}")
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
    raise RuntimeError(f"Qwen chat call failed after {retries} retries: {last_err}")


def configure_qwen_chat(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | str | None = None,
    timeout_seconds: float | str | None = None,
    max_tokens: int | str | None = None,
    enable_thinking: bool | str | None = None,
    optimizer_base_url: str | None = None,
    optimizer_api_key: str | None = None,
    optimizer_temperature: float | str | None = None,
    optimizer_timeout_seconds: float | str | None = None,
    optimizer_max_tokens: int | str | None = None,
    optimizer_enable_thinking: bool | str | None = None,
    target_base_url: str | None = None,
    target_api_key: str | None = None,
    target_temperature: float | str | None = None,
    target_timeout_seconds: float | str | None = None,
    target_max_tokens: int | str | None = None,
    target_enable_thinking: bool | str | None = None,
) -> None:
    with _config_lock:
        if base_url is not None:
            os.environ["QWEN_CHAT_BASE_URL"] = str(base_url).strip()
        if api_key is not None:
            os.environ["QWEN_CHAT_API_KEY"] = str(api_key).strip()
        if temperature is not None:
            os.environ["QWEN_CHAT_TEMPERATURE"] = str(temperature).strip()
        if timeout_seconds is not None:
            os.environ["QWEN_CHAT_TIMEOUT_SECONDS"] = str(timeout_seconds)
        if max_tokens is not None:
            os.environ["QWEN_CHAT_MAX_TOKENS"] = str(max_tokens)
        if enable_thinking is not None:
            os.environ["QWEN_CHAT_ENABLE_THINKING"] = (
                "true" if _parse_bool(enable_thinking) else "false"
            )
        _update_config(
            OPTIMIZER_CONFIG,
            "optimizer",
            base_url=optimizer_base_url if optimizer_base_url is not None else base_url,
            api_key=optimizer_api_key if optimizer_api_key is not None else api_key,
            temperature=(
                optimizer_temperature
                if optimizer_temperature is not None
                else temperature
            ),
            timeout_seconds=(
                optimizer_timeout_seconds
                if optimizer_timeout_seconds is not None
                else timeout_seconds
            ),
            max_tokens=optimizer_max_tokens if optimizer_max_tokens is not None else max_tokens,
            enable_thinking=(
                optimizer_enable_thinking
                if optimizer_enable_thinking is not None
                else enable_thinking
            ),
        )
        _update_config(
            TARGET_CONFIG,
            "target",
            base_url=target_base_url if target_base_url is not None else base_url,
            api_key=target_api_key if target_api_key is not None else api_key,
            temperature=target_temperature if target_temperature is not None else temperature,
            timeout_seconds=(
                target_timeout_seconds
                if target_timeout_seconds is not None
                else timeout_seconds
            ),
            max_tokens=target_max_tokens if target_max_tokens is not None else max_tokens,
            enable_thinking=(
                target_enable_thinking
                if target_enable_thinking is not None
                else enable_thinking
            ),
        )


def _update_config(
    config: QwenChatConfig,
    role: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | str | None = None,
    timeout_seconds: float | str | None = None,
    max_tokens: int | str | None = None,
    enable_thinking: bool | str | None = None,
) -> None:
    env_prefix = role.upper()
    if base_url is not None:
        config.base_url = str(base_url).strip() or config.base_url
        os.environ[f"{env_prefix}_QWEN_CHAT_BASE_URL"] = config.base_url
    if api_key is not None:
        config.api_key = str(api_key).strip()
        os.environ[f"{env_prefix}_QWEN_CHAT_API_KEY"] = config.api_key
    if temperature is not None:
        raw = str(temperature).strip()
        config.temperature = float(raw) if raw else None
        os.environ[f"{env_prefix}_QWEN_CHAT_TEMPERATURE"] = raw
    if timeout_seconds is not None:
        config.timeout_seconds = float(timeout_seconds)
        os.environ[f"{env_prefix}_QWEN_CHAT_TIMEOUT_SECONDS"] = str(timeout_seconds)
    if max_tokens is not None:
        config.max_tokens = int(max_tokens)
        os.environ[f"{env_prefix}_QWEN_CHAT_MAX_TOKENS"] = str(max_tokens)
    if enable_thinking is not None:
        config.enable_thinking = _parse_bool(enable_thinking)
        os.environ[f"{env_prefix}_QWEN_CHAT_ENABLE_THINKING"] = (
            "true" if config.enable_thinking else "false"
        )


def get_max_tokens() -> int:
    return TARGET_CONFIG.max_tokens


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
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
        role="optimizer",
        timeout=timeout,
    )


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
        role="target",
        timeout=timeout,
    )


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
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
        role="optimizer",
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
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
        role="target",
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
    TARGET_CONFIG.deployment = deployment or default_model_for_backend("qwen_chat")
    os.environ["TARGET_DEPLOYMENT"] = TARGET_CONFIG.deployment


def set_optimizer_deployment(deployment: str) -> None:
    OPTIMIZER_CONFIG.deployment = deployment or default_model_for_backend("qwen_chat")
    os.environ["OPTIMIZER_DEPLOYMENT"] = OPTIMIZER_CONFIG.deployment

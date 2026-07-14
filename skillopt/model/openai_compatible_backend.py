"""Generic OpenAI-compatible chat backend for optimizer and target paths.

This backend talks to *any* service that exposes an OpenAI-compatible
``/chat/completions`` endpoint through the official ``openai`` SDK. A single
implementation therefore covers a large family of providers, for example:

* DeepSeek           (``https://api.deepseek.com``)
* Groq               (``https://api.groq.com/openai/v1``)
* Together AI        (``https://api.together.xyz/v1``)
* Mistral / Fireworks / OpenRouter / Perplexity / xAI Grok
* Ollama             (``http://localhost:11434/v1``)
* vLLM / SGLang / TGI self-hosted servers
* LiteLLM proxy      (``http://localhost:4000``)
* Azure OpenAI and OpenAI itself

Unlike the Azure backend it never assumes Azure-specific auth or the Responses
API — it only needs a ``base_url`` and an ``api_key`` (some local servers accept
any key, so the key is optional and falls back to a harmless placeholder).

The module mirrors the callable surface of the other chat backends
(:mod:`skillopt.model.qwen_backend`, :mod:`skillopt.model.minimax_backend`) so
it can be selected as the optimizer and/or target backend and routed through
:mod:`skillopt.model`.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from skillopt.model.common import (
    TokenTracker,
    compat_message_from_chat_message,
    default_model_for_backend,
    usage_from_openai_usage,
)

BACKEND_NAME = "openai_compatible"

# A neutral, widely-available default. Real deployments should set the model
# explicitly (e.g. "deepseek-chat", "llama-3.3-70b-versatile", "qwen2.5:7b").
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    deployment: str
    timeout_seconds: float
    max_tokens: int
    temperature: float | None


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
    """Resolve a config value, preferring role-specific over shared env vars."""
    role_key = f"{role.upper()}_OPENAI_COMPATIBLE_{key}"
    generic_key = f"OPENAI_COMPATIBLE_{key}"
    return os.environ.get(role_key) or os.environ.get(generic_key) or default


def _initial_config(role: str) -> OpenAICompatibleConfig:
    role_upper = role.upper()
    deployment_env = "OPTIMIZER_DEPLOYMENT" if role == "optimizer" else "TARGET_DEPLOYMENT"
    return OpenAICompatibleConfig(
        base_url=_role_env(role, "BASE_URL", _DEFAULT_BASE_URL),
        api_key=_role_env(role, "API_KEY", ""),
        deployment=(
            os.environ.get(f"{role_upper}_OPENAI_COMPATIBLE_MODEL")
            or os.environ.get("OPENAI_COMPATIBLE_MODEL")
            or os.environ.get(deployment_env)
            or default_model_for_backend(BACKEND_NAME)
        ),
        timeout_seconds=float(_role_env(role, "TIMEOUT_SECONDS", "300") or 300),
        max_tokens=_parse_int(_role_env(role, "MAX_TOKENS", "8000"), 8000),
        temperature=_parse_optional_float(_role_env(role, "TEMPERATURE", "")),
    )


OPTIMIZER_CONFIG = _initial_config("optimizer")
TARGET_CONFIG = _initial_config("target")

_config_lock = threading.Lock()
_client_lock = threading.Lock()
tracker = TokenTracker()

_optimizer_client: OpenAI | None = None
_target_client: OpenAI | None = None


def _config_for(role: str) -> OpenAICompatibleConfig:
    return OPTIMIZER_CONFIG if role == "optimizer" else TARGET_CONFIG


def _build_client(config: OpenAICompatibleConfig) -> OpenAI:
    return OpenAI(
        base_url=config.base_url.rstrip("/") or _DEFAULT_BASE_URL,
        # Some OpenAI-compatible servers (Ollama, vLLM, local proxies) do not
        # require an API key. The SDK still expects a non-empty string, so fall
        # back to a harmless placeholder when none is configured.
        api_key=config.api_key or "dummy",
        timeout=config.timeout_seconds,
    )


def _get_client(role: str) -> OpenAI:
    global _optimizer_client, _target_client
    with _client_lock:
        if role == "optimizer":
            if _optimizer_client is None:
                _optimizer_client = _build_client(OPTIMIZER_CONFIG)
            return _optimizer_client
        if _target_client is None:
            _target_client = _build_client(TARGET_CONFIG)
        return _target_client


def _reset_clients() -> None:
    global _optimizer_client, _target_client
    with _client_lock:
        _optimizer_client = None
        _target_client = None


def count_tokens(text: str, model: str | None = None) -> int:
    """Best-effort token count for a string.

    Uses ``tiktoken`` when available (per-model encoding, falling back to the
    ``cl100k_base`` encoding). If ``tiktoken`` is not installed or fails — which
    is common for non-OpenAI models served through compatible APIs — it falls
    back to a character-based estimate of roughly four characters per token.
    """
    if not text:
        return 0
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-4o")
        except Exception:  # noqa: BLE001 - unknown/non-OpenAI model name
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:  # noqa: BLE001 - tiktoken missing or encoding failure
        # Rough heuristic: ~4 characters per token for English-like text.
        return max(1, (len(text) + 3) // 4)


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
    config = _config_for(role)
    client = _get_client(role)
    kwargs: dict[str, Any] = {
        "model": deployment or config.deployment,
        "messages": messages,
        # ``max_tokens`` (rather than ``max_completion_tokens``) is the field
        # understood by the broadest set of OpenAI-compatible providers.
        "max_tokens": min(max_completion_tokens, config.max_tokens),
    }
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
    if timeout is not None:
        kwargs["timeout"] = timeout

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            choices = getattr(resp, "choices", None) or []
            if not choices:
                raise RuntimeError(
                    f"OpenAI-compatible API returned no choices: {resp!r}"
                )
            message = choices[0].message
            text = message.content or ""
            usage_info = usage_from_openai_usage(getattr(resp, "usage", None))
            tracker.record(
                stage,
                usage_info["prompt_tokens"],
                usage_info["completion_tokens"],
            )
            if return_message:
                return compat_message_from_chat_message(message), usage_info
            return text, usage_info
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(
        f"OpenAI-compatible chat call failed after {retries} retries: {last_err}"
    )


# ── Public API (mirrors the other chat backends) ─────────────────────────────


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> tuple[str, dict[str, int]]:
    del reasoning_effort  # not forwarded — kept for a uniform signature
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
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
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
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


# ── Configuration / lifecycle ────────────────────────────────────────────────


def _update_config(
    config: OpenAICompatibleConfig,
    role: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    deployment: str | None = None,
    temperature: float | str | None = None,
    timeout_seconds: float | str | None = None,
    max_tokens: int | str | None = None,
) -> None:
    env_prefix = role.upper()
    if base_url is not None:
        config.base_url = str(base_url).strip() or config.base_url
        os.environ[f"{env_prefix}_OPENAI_COMPATIBLE_BASE_URL"] = config.base_url
    if api_key is not None:
        config.api_key = str(api_key).strip()
        os.environ[f"{env_prefix}_OPENAI_COMPATIBLE_API_KEY"] = config.api_key
    if deployment is not None:
        config.deployment = str(deployment).strip() or config.deployment
        os.environ[f"{env_prefix}_OPENAI_COMPATIBLE_MODEL"] = config.deployment
    if temperature is not None:
        raw = str(temperature).strip()
        config.temperature = float(raw) if raw else None
        os.environ[f"{env_prefix}_OPENAI_COMPATIBLE_TEMPERATURE"] = raw
    if timeout_seconds is not None:
        config.timeout_seconds = float(timeout_seconds)
        os.environ[f"{env_prefix}_OPENAI_COMPATIBLE_TIMEOUT_SECONDS"] = str(timeout_seconds)
    if max_tokens is not None:
        config.max_tokens = int(max_tokens)
        os.environ[f"{env_prefix}_OPENAI_COMPATIBLE_MAX_TOKENS"] = str(max_tokens)


def configure_openai_compatible(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float | str | None = None,
    timeout_seconds: float | str | None = None,
    max_tokens: int | str | None = None,
    optimizer_base_url: str | None = None,
    optimizer_api_key: str | None = None,
    optimizer_model: str | None = None,
    target_base_url: str | None = None,
    target_api_key: str | None = None,
    target_model: str | None = None,
) -> None:
    """Configure the generic OpenAI-compatible backend at runtime.

    Shared values apply to both the optimizer and target roles; the
    ``optimizer_*`` / ``target_*`` variants override them per role.
    """
    with _config_lock:
        if base_url is not None:
            os.environ["OPENAI_COMPATIBLE_BASE_URL"] = str(base_url).strip()
        if api_key is not None:
            os.environ["OPENAI_COMPATIBLE_API_KEY"] = str(api_key).strip()
        if model is not None:
            os.environ["OPENAI_COMPATIBLE_MODEL"] = str(model).strip()
        if temperature is not None:
            os.environ["OPENAI_COMPATIBLE_TEMPERATURE"] = str(temperature).strip()
        if timeout_seconds is not None:
            os.environ["OPENAI_COMPATIBLE_TIMEOUT_SECONDS"] = str(timeout_seconds)
        if max_tokens is not None:
            os.environ["OPENAI_COMPATIBLE_MAX_TOKENS"] = str(max_tokens)
        _update_config(
            OPTIMIZER_CONFIG,
            "optimizer",
            base_url=optimizer_base_url if optimizer_base_url is not None else base_url,
            api_key=optimizer_api_key if optimizer_api_key is not None else api_key,
            deployment=optimizer_model if optimizer_model is not None else model,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )
        _update_config(
            TARGET_CONFIG,
            "target",
            base_url=target_base_url if target_base_url is not None else base_url,
            api_key=target_api_key if target_api_key is not None else api_key,
            deployment=target_model if target_model is not None else model,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )
    _reset_clients()


def get_max_tokens() -> int:
    return TARGET_CONFIG.max_tokens


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    # Reasoning effort is provider-specific and not universally supported by
    # OpenAI-compatible endpoints, so it is intentionally a no-op here.
    del effort


def set_target_deployment(deployment: str) -> None:
    TARGET_CONFIG.deployment = deployment or default_model_for_backend(BACKEND_NAME)
    os.environ["TARGET_DEPLOYMENT"] = TARGET_CONFIG.deployment
    _reset_clients()


def set_optimizer_deployment(deployment: str) -> None:
    OPTIMIZER_CONFIG.deployment = deployment or default_model_for_backend(BACKEND_NAME)
    os.environ["OPTIMIZER_DEPLOYMENT"] = OPTIMIZER_CONFIG.deployment
    _reset_clients()

"""ReflACT model API with runtime backend selection for the target path."""

from __future__ import annotations

from typing import Any

from skillopt.model import azure_openai as _openai
from skillopt.model import claude_backend as _claude
from skillopt.model import claude_code_backend as _claude_code
from skillopt.model import codex_backend as _codex
from skillopt.model import copilot_backend as _copilot
from skillopt.model import minimax_backend as _minimax
from skillopt.model import openai_compatible_backend as _openai_compat
from skillopt.model import qwen_backend as _qwen
from skillopt.model.backend_config import (  # noqa: F401
    configure_claude_code_exec,
    configure_codex_exec,
    configure_codex_exec_from_config,
    configure_copilot_chat,
    configure_copilot_exec,
    configure_cursor_exec,
    get_claude_code_exec_config,
    get_codex_exec_config,
    get_copilot_chat_config,
    get_copilot_exec_config,
    get_cursor_exec_config,
    get_optimizer_backend,
    get_target_backend,
    is_optimizer_chat_backend,
    is_target_chat_backend,
    is_target_exec_backend,
    set_optimizer_backend,
    set_target_backend,
)
from skillopt.model.common import normalize_backend_name


def set_backend(name: str | None) -> str:
    """Backward-compatible global backend setter.

    Historically the codebase used one shared backend for both optimizer and
    target. Keep that entry point so older scripts continue to work, while
    mapping it onto the split optimizer/target backend model.
    """
    normalized = normalize_backend_name(name)
    if normalized in {"azure_openai", "openai_chat"}:
        set_optimizer_backend("openai_chat")
        set_target_backend("openai_chat")
        return "azure_openai"
    if normalized == "claude_chat":
        set_optimizer_backend("claude_chat")
        set_target_backend("claude_chat")
        return "claude_chat"
    if normalized in {"codex", "codex_exec"}:
        set_optimizer_backend("codex_exec")
        set_target_backend("codex_exec")
        return normalized
    if normalized == "claude_code_exec":
        # Both roles default to Claude Code so reflection sees the full
        # trajectory. A role pinned to a non-default value (e.g. minimax_chat)
        # still overrides; an explicit --optimizer_backend openai_chat does not,
        # because openai_chat is one of the base-config defaults.
        set_optimizer_backend("claude_code_exec")
        set_target_backend(normalized)
        return normalized
    if normalized == "cursor_exec":
        set_optimizer_backend("openai_chat")
        set_target_backend("cursor_exec")
        return "cursor_exec"
    if normalized == "copilot_chat":
        # The CLI-authenticated backend drives both roles without a separate
        # provider API key; inference still uses the Copilot cloud service.
        set_optimizer_backend("copilot_chat")
        set_target_backend("copilot_chat")
        return "copilot_chat"
    if normalized == "copilot_exec":
        set_optimizer_backend("openai_chat")
        set_target_backend("copilot_exec")
        return "copilot_exec"
    if normalized == "qwen_chat":
        set_optimizer_backend("openai_chat")
        set_target_backend("qwen_chat")
        return "qwen_chat"
    if normalized == "minimax_chat":
        set_optimizer_backend("openai_chat")
        set_target_backend("minimax_chat")
        return "minimax_chat"
    if normalized == "openai_compatible":
        set_optimizer_backend("openai_compatible")
        set_target_backend("openai_compatible")
        return "openai_compatible"
    raise ValueError(f"Unsupported legacy backend: {name!r}")


def get_backend_name() -> str:
    """Best-effort backward-compatible backend summary."""
    optimizer = get_optimizer_backend()
    target = get_target_backend()
    if optimizer == "claude_chat" and target == "claude_chat":
        return "claude_chat"
    if optimizer == "qwen_chat" and target == "qwen_chat":
        return "qwen_chat"
    if optimizer == "copilot_chat" and target == "copilot_chat":
        return "copilot_chat"
    if optimizer == "openai_chat" and target == "openai_chat":
        return "azure_openai"
    if optimizer == "codex_exec" and target == "codex_exec":
        return "codex"
    if optimizer == "openai_chat" and target == "qwen_chat":
        return "qwen_chat"
    if optimizer == "openai_chat" and target == "minimax_chat":
        return "minimax_chat"
    if optimizer == "openai_chat" and target == "cursor_exec":
        return "cursor_exec"
    if optimizer == "openai_chat" and target == "copilot_exec":
        return "copilot_exec"
    if optimizer == "openai_compatible" and target == "openai_compatible":
        return "openai_compatible"
    return f"{optimizer}+{target}"


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    if get_optimizer_backend() == "claude_chat":
        return _claude.chat_optimizer(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            timeout=timeout,
        )
    if get_optimizer_backend() == "copilot_chat":
        return _copilot.chat_optimizer(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    if get_optimizer_backend() == "qwen_chat":
        return _qwen.chat_optimizer(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    if get_optimizer_backend() == "minimax_chat":
        return _minimax.chat_optimizer(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    if get_optimizer_backend() == "openai_compatible":
        return _openai_compat.chat_optimizer(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    if get_optimizer_backend() == "codex_exec":
        return _codex.chat_optimizer(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            timeout=timeout,
        )
    if get_optimizer_backend() == "claude_code_exec":
        return _claude_code.chat_optimizer(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
    return _openai.chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    if get_target_backend() == "claude_chat":
        return _claude.chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            timeout=timeout,
        )
    if get_target_backend() == "qwen_chat":
        return _qwen.chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    if get_target_backend() == "minimax_chat":
        return _minimax.chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
        )
    if get_target_backend() == "openai_compatible":
        return _openai_compat.chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    if get_target_backend() == "copilot_chat":
        return _copilot.chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    if not is_target_chat_backend():
        raise NotImplementedError(
            "chat_target is only supported with target_backend=openai_chat, claude_chat, qwen_chat, minimax_chat, "
            "copilot_chat, or openai_compatible. Exec backends are handled in environment-specific rollout code."
        )
    return _openai.chat_target(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
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
    timeout: int | None = None,
) -> tuple[Any, dict]:
    if get_optimizer_backend() == "copilot_chat":
        return _copilot.chat_optimizer_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    if get_optimizer_backend() == "claude_chat":
        return _claude.chat_optimizer_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    if get_optimizer_backend() == "qwen_chat":
        return _qwen.chat_optimizer_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    if get_optimizer_backend() == "minimax_chat":
        return _minimax.chat_target_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    if get_optimizer_backend() == "openai_compatible":
        return _openai_compat.chat_optimizer_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    if get_optimizer_backend() == "codex_exec":
        return _codex.chat_optimizer_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    if get_optimizer_backend() == "claude_code_exec":
        return _claude_code.chat_optimizer_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
    return _openai.chat_optimizer_messages(
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
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
    timeout: int | None = None,
) -> tuple[Any, dict]:
    if get_target_backend() == "copilot_chat":
        return _copilot.chat_target_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    if get_target_backend() == "claude_chat":
        return _claude.chat_target_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    if get_target_backend() == "qwen_chat":
        return _qwen.chat_target_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    if get_target_backend() == "minimax_chat":
        return _minimax.chat_target_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
        )
    if get_target_backend() == "openai_compatible":
        return _openai_compat.chat_target_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    if not is_target_chat_backend():
        raise NotImplementedError(
            "chat_target_messages is only supported with target_backend=openai_chat, claude_chat, qwen_chat, "
            "minimax_chat, copilot_chat, or openai_compatible. Exec backends are handled in environment-specific "
            "rollout code."
        )
    return _openai.chat_target_messages(
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def chat_messages_with_deployment(
    deployment: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict]:
    return _openai.chat_messages_with_deployment(
        deployment=deployment,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def chat_with_deployment(
    deployment: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    return _openai.chat_with_deployment(
        deployment=deployment,
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def get_token_summary() -> dict:
    summary = _openai.get_token_summary()
    claude_summary = _claude.get_token_summary()
    for stage, values in claude_summary.items():
        if stage == "_total":
            continue
        if stage not in summary:
            summary[stage] = values
            continue
        summary[stage]["calls"] += values["calls"]
        summary[stage]["prompt_tokens"] += values["prompt_tokens"]
        summary[stage]["completion_tokens"] += values["completion_tokens"]
        summary[stage]["total_tokens"] += values["total_tokens"]
    claude_code_summary = _claude_code.get_token_summary()
    for stage, values in claude_code_summary.items():
        if stage == "_total":
            continue
        if stage not in summary:
            summary[stage] = values
            continue
        summary[stage]["calls"] += values["calls"]
        summary[stage]["prompt_tokens"] += values["prompt_tokens"]
        summary[stage]["completion_tokens"] += values["completion_tokens"]
        summary[stage]["total_tokens"] += values["total_tokens"]
    qwen_summary = _qwen.get_token_summary()
    for stage, values in qwen_summary.items():
        if stage == "_total":
            continue
        if stage not in summary:
            summary[stage] = values
            continue
        summary[stage]["calls"] += values["calls"]
        summary[stage]["prompt_tokens"] += values["prompt_tokens"]
        summary[stage]["completion_tokens"] += values["completion_tokens"]
        summary[stage]["total_tokens"] += values["total_tokens"]
    minimax_summary = _minimax.get_token_summary()
    for stage, values in minimax_summary.items():
        if stage == "_total":
            continue
        if stage not in summary:
            summary[stage] = values
            continue
        summary[stage]["calls"] += values["calls"]
        summary[stage]["prompt_tokens"] += values["prompt_tokens"]
        summary[stage]["completion_tokens"] += values["completion_tokens"]
        summary[stage]["total_tokens"] += values["total_tokens"]
    openai_compat_summary = _openai_compat.get_token_summary()
    for stage, values in openai_compat_summary.items():
        if stage == "_total":
            continue
        if stage not in summary:
            summary[stage] = values
            continue
        summary[stage]["calls"] += values["calls"]
        summary[stage]["prompt_tokens"] += values["prompt_tokens"]
        summary[stage]["completion_tokens"] += values["completion_tokens"]
        summary[stage]["total_tokens"] += values["total_tokens"]
    codex_summary = _codex.get_token_summary()
    for stage, values in codex_summary.items():
        if stage == "_total":
            continue
        if stage not in summary:
            summary[stage] = values
            continue
        summary[stage]["calls"] += values["calls"]
        summary[stage]["prompt_tokens"] += values["prompt_tokens"]
        summary[stage]["completion_tokens"] += values["completion_tokens"]
        summary[stage]["total_tokens"] += values["total_tokens"]
    copilot_summary = _copilot.get_token_summary()
    for stage, values in copilot_summary.items():
        if stage == "_total":
            continue
        if stage not in summary:
            summary[stage] = values
            continue
        summary[stage]["calls"] += values["calls"]
        summary[stage]["prompt_tokens"] += values["prompt_tokens"]
        summary[stage]["completion_tokens"] += values["completion_tokens"]
        summary[stage]["total_tokens"] += values["total_tokens"]
    total = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for stage, values in summary.items():
        if stage == "_total":
            continue
        total["calls"] += values["calls"]
        total["prompt_tokens"] += values["prompt_tokens"]
        total["completion_tokens"] += values["completion_tokens"]
        total["total_tokens"] += values["total_tokens"]
    summary["_total"] = total
    return summary


def reset_token_tracker() -> None:
    _openai.reset_token_tracker()
    _claude.reset_token_tracker()
    _claude_code.reset_token_tracker()
    _qwen.reset_token_tracker()
    _minimax.reset_token_tracker()
    _openai_compat.reset_token_tracker()
    _codex.reset_token_tracker()
    _copilot.reset_token_tracker()


def configure_azure_openai(
    *,
    endpoint: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    auth_mode: str | None = None,
    ad_scope: str | None = None,
    managed_identity_client_id: str | None = None,
    optimizer_endpoint: str | None = None,
    optimizer_api_version: str | None = None,
    optimizer_api_key: str | None = None,
    optimizer_auth_mode: str | None = None,
    optimizer_ad_scope: str | None = None,
    optimizer_managed_identity_client_id: str | None = None,
    target_endpoint: str | None = None,
    target_api_version: str | None = None,
    target_api_key: str | None = None,
    target_auth_mode: str | None = None,
    target_ad_scope: str | None = None,
    target_managed_identity_client_id: str | None = None,
) -> None:
    _openai.configure_azure_openai(
        endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,
        auth_mode=auth_mode,
        ad_scope=ad_scope,
        managed_identity_client_id=managed_identity_client_id,
        optimizer_endpoint=optimizer_endpoint,
        optimizer_api_version=optimizer_api_version,
        optimizer_api_key=optimizer_api_key,
        optimizer_auth_mode=optimizer_auth_mode,
        optimizer_ad_scope=optimizer_ad_scope,
        optimizer_managed_identity_client_id=optimizer_managed_identity_client_id,
        target_endpoint=target_endpoint,
        target_api_version=target_api_version,
        target_api_key=target_api_key,
        target_auth_mode=target_auth_mode,
        target_ad_scope=target_ad_scope,
        target_managed_identity_client_id=target_managed_identity_client_id,
    )


def configure_qwen_chat(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | str | None = None,
    timeout_seconds: float | str | None = None,
    max_tokens: int | str | None = None,
    enable_thinking: bool | str | None = None,
    use_max_completion_tokens: bool | str | None = None,
    optimizer_base_url: str | None = None,
    optimizer_api_key: str | None = None,
    optimizer_temperature: float | str | None = None,
    optimizer_timeout_seconds: float | str | None = None,
    optimizer_max_tokens: int | str | None = None,
    optimizer_enable_thinking: bool | str | None = None,
    optimizer_use_max_completion_tokens: bool | str | None = None,
    target_base_url: str | None = None,
    target_api_key: str | None = None,
    target_temperature: float | str | None = None,
    target_timeout_seconds: float | str | None = None,
    target_max_tokens: int | str | None = None,
    target_enable_thinking: bool | str | None = None,
    target_use_max_completion_tokens: bool | str | None = None,
) -> None:
    _qwen.configure_qwen_chat(
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        use_max_completion_tokens=use_max_completion_tokens,
        optimizer_base_url=optimizer_base_url,
        optimizer_api_key=optimizer_api_key,
        optimizer_temperature=optimizer_temperature,
        optimizer_timeout_seconds=optimizer_timeout_seconds,
        optimizer_max_tokens=optimizer_max_tokens,
        optimizer_enable_thinking=optimizer_enable_thinking,
        optimizer_use_max_completion_tokens=optimizer_use_max_completion_tokens,
        target_base_url=target_base_url,
        target_api_key=target_api_key,
        target_temperature=target_temperature,
        target_timeout_seconds=target_timeout_seconds,
        target_max_tokens=target_max_tokens,
        target_enable_thinking=target_enable_thinking,
        target_use_max_completion_tokens=target_use_max_completion_tokens,
    )


def configure_minimax_chat(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | str | None = None,
    timeout_seconds: float | str | None = None,
    max_tokens: int | str | None = None,
    enable_thinking: bool | str | None = None,
) -> None:
    _minimax.configure_minimax_chat(
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )


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
    _openai_compat.configure_openai_compatible(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        optimizer_base_url=optimizer_base_url,
        optimizer_api_key=optimizer_api_key,
        optimizer_model=optimizer_model,
        target_base_url=target_base_url,
        target_api_key=target_api_key,
        target_model=target_model,
    )


def set_reasoning_effort(effort: str | None) -> None:
    _openai.set_reasoning_effort(effort)
    _claude.set_reasoning_effort(effort)
    _claude_code.set_reasoning_effort(effort)
    _qwen.set_reasoning_effort(effort)
    _minimax.set_reasoning_effort(effort)
    _openai_compat.set_reasoning_effort(effort)
    _codex.set_reasoning_effort(effort)


def set_target_deployment(deployment: str) -> None:
    _openai.set_target_deployment(deployment)
    _claude.set_target_deployment(deployment)
    _claude_code.set_target_deployment(deployment)
    _qwen.set_target_deployment(deployment)
    _minimax.set_target_deployment(deployment)
    _openai_compat.set_target_deployment(deployment)
    _codex.set_target_deployment(deployment)


def set_optimizer_deployment(deployment: str) -> None:
    _openai.set_optimizer_deployment(deployment)
    _claude.set_optimizer_deployment(deployment)
    _claude_code.set_optimizer_deployment(deployment)
    _qwen.set_optimizer_deployment(deployment)
    _openai_compat.set_optimizer_deployment(deployment)
    _codex.set_optimizer_deployment(deployment)

"""ReflACT model API with runtime backend selection for the student path."""

from __future__ import annotations

from typing import Any

from skillopt.model import azure_openai as _openai
from skillopt.model import claude_backend as _claude
from skillopt.model.backend_config import (  # noqa: F401
    configure_claude_code_exec,
    configure_codex_exec,
    get_claude_code_exec_config,
    get_codex_exec_config,
    get_student_backend,
    get_teacher_backend,
    is_student_chat_backend,
    is_student_exec_backend,
    is_teacher_chat_backend,
    set_student_backend,
    set_teacher_backend,
)


def set_backend(name: str | None) -> str:
    """Backward-compatible global backend setter.

    Historically the codebase used one shared backend for both teacher and
    student. Keep that entry point so older scripts continue to work, while
    mapping it onto the split teacher/student backend model.
    """
    normalized = str(name or "azure_openai").strip().lower()
    if normalized in {"azure_openai", "openai_chat", "azure", "azure-openai"}:
        set_teacher_backend("openai_chat")
        set_student_backend("openai_chat")
        return "azure_openai"
    if normalized in {"claude", "claude_chat", "anthropic"}:
        set_teacher_backend("claude_chat")
        set_student_backend("claude_chat")
        return "claude_chat"
    if normalized == "codex":
        set_teacher_backend("openai_chat")
        set_student_backend("codex_exec")
        return "codex"
    if normalized in {"codex_exec", "claude_code_exec"}:
        set_teacher_backend("openai_chat")
        set_student_backend(normalized)
        return normalized
    raise ValueError(f"Unsupported legacy backend: {name!r}")


def get_backend_name() -> str:
    """Best-effort backward-compatible backend summary."""
    teacher = get_teacher_backend()
    student = get_student_backend()
    if teacher == "claude_chat" and student == "claude_chat":
        return "claude_chat"
    if teacher == "openai_chat" and student == "openai_chat":
        return "azure_openai"
    if teacher == "openai_chat" and student == "codex_exec":
        return "codex"
    return f"{teacher}+{student}"


def chat_teacher(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "teacher",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    if get_teacher_backend() == "claude_chat":
        return _claude.chat_teacher(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            timeout=timeout,
        )
    return _openai.chat_teacher(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def chat_student(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "student",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    if get_student_backend() == "claude_chat":
        return _claude.chat_student(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            timeout=timeout,
        )
    if not is_student_chat_backend():
        raise NotImplementedError(
            "chat_student is only supported with student_backend=openai_chat or claude_chat. "
            "Exec backends are handled in environment-specific rollout code."
        )
    return _openai.chat_student(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def chat_teacher_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "teacher",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict]:
    if get_teacher_backend() == "claude_chat":
        return _claude.chat_teacher_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    return _openai.chat_teacher_messages(
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


def chat_student_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "student",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict]:
    if get_student_backend() == "claude_chat":
        return _claude.chat_student_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            stage=stage,
            tools=tools,
            tool_choice=tool_choice,
            return_message=return_message,
            timeout=timeout,
        )
    if not is_student_chat_backend():
        raise NotImplementedError(
            "chat_student_messages is only supported with student_backend=openai_chat or claude_chat. "
            "Exec backends are handled in environment-specific rollout code."
        )
    return _openai.chat_student_messages(
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


def configure_azure_openai(
    *,
    endpoint: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    auth_mode: str | None = None,
    ad_scope: str | None = None,
    managed_identity_client_id: str | None = None,
    teacher_endpoint: str | None = None,
    teacher_api_version: str | None = None,
    teacher_api_key: str | None = None,
    teacher_auth_mode: str | None = None,
    teacher_ad_scope: str | None = None,
    teacher_managed_identity_client_id: str | None = None,
    student_endpoint: str | None = None,
    student_api_version: str | None = None,
    student_api_key: str | None = None,
    student_auth_mode: str | None = None,
    student_ad_scope: str | None = None,
    student_managed_identity_client_id: str | None = None,
) -> None:
    _openai.configure_azure_openai(
        endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,
        auth_mode=auth_mode,
        ad_scope=ad_scope,
        managed_identity_client_id=managed_identity_client_id,
        teacher_endpoint=teacher_endpoint,
        teacher_api_version=teacher_api_version,
        teacher_api_key=teacher_api_key,
        teacher_auth_mode=teacher_auth_mode,
        teacher_ad_scope=teacher_ad_scope,
        teacher_managed_identity_client_id=teacher_managed_identity_client_id,
        student_endpoint=student_endpoint,
        student_api_version=student_api_version,
        student_api_key=student_api_key,
        student_auth_mode=student_auth_mode,
        student_ad_scope=student_ad_scope,
        student_managed_identity_client_id=student_managed_identity_client_id,
    )


def set_reasoning_effort(effort: str | None) -> None:
    _openai.set_reasoning_effort(effort)
    _claude.set_reasoning_effort(effort)


def set_student_deployment(deployment: str) -> None:
    _openai.set_student_deployment(deployment)
    _claude.set_student_deployment(deployment)


def set_teacher_deployment(deployment: str) -> None:
    _openai.set_teacher_deployment(deployment)
    _claude.set_teacher_deployment(deployment)

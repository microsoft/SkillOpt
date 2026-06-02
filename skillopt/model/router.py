"""Runtime backend router for ReflACT model calls."""
from __future__ import annotations

import os
from typing import Any

from . import azure_openai, claude_backend, codex_backend
from .common import normalize_backend_name

_ACTIVE_BACKEND = normalize_backend_name(
    os.environ.get("REFLACT_MODEL_BACKEND", "azure_openai")
)


def _backend_module(name: str):
    if name == "azure_openai":
        return azure_openai
    if name == "codex":
        return codex_backend
    if name == "claude":
        return claude_backend
    raise ValueError(f"Unknown backend: {name!r}")


def _all_backend_modules() -> list[Any]:
    return [azure_openai, codex_backend, claude_backend]


def set_backend(name: str | None) -> str:
    """Select the active model backend for subsequent calls."""
    global _ACTIVE_BACKEND
    normalized = normalize_backend_name(name)
    if normalized not in {"azure_openai", "codex", "claude"}:
        valid = ", ".join(sorted({"azure_openai", "codex", "claude"}))
        raise ValueError(f"Unknown backend {name!r}. Expected one of: {valid}")
    _ACTIVE_BACKEND = normalized
    os.environ["REFLACT_MODEL_BACKEND"] = normalized
    return _ACTIVE_BACKEND


def get_backend_name() -> str:
    return _ACTIVE_BACKEND


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    return _backend_module(_ACTIVE_BACKEND).chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        timeout=timeout,
    )


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    return _backend_module(_ACTIVE_BACKEND).chat_target(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        timeout=timeout,
    )


def chat_with_deployment(
    deployment: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    return _backend_module(_ACTIVE_BACKEND).chat_with_deployment(
        deployment=deployment,
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        timeout=timeout,
    )


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict[str, int]]:
    return _backend_module(_ACTIVE_BACKEND).chat_optimizer_messages(
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
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
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict[str, int]]:
    return _backend_module(_ACTIVE_BACKEND).chat_target_messages(
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
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
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict[str, int]]:
    return _backend_module(_ACTIVE_BACKEND).chat_messages_with_deployment(
        deployment=deployment,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def get_token_summary() -> dict[str, dict[str, int]]:
    return _backend_module(_ACTIVE_BACKEND).get_token_summary()


def reset_token_tracker() -> None:
    _backend_module(_ACTIVE_BACKEND).reset_token_tracker()


def set_reasoning_effort(effort: str | None) -> None:
    for module in _all_backend_modules():
        module.set_reasoning_effort(effort)


def set_target_deployment(deployment: str) -> None:
    for module in _all_backend_modules():
        module.set_target_deployment(deployment)


def set_optimizer_deployment(deployment: str) -> None:
    for module in _all_backend_modules():
        module.set_optimizer_deployment(deployment)


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
    azure_openai.configure_azure_openai(
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

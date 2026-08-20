"""Claude Code CLI/SDK chat backend for ReflACT (optimizer role).

Runs Claude Code (the same CLI/SDK that powers the ``claude_code_exec`` target
backend) as a plain chat model for reflection.  This gives the optimizer access
to Claude's full context window, so minibatch trajectories are not truncated by
a narrower chat backend.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from skillopt.model import codex_backend as _codex
from skillopt.model.claude_backend import _build_prompt_from_messages
from skillopt.model.codex_harness import run_claude_code_chat
from skillopt.model.common import TokenTracker

OPTIMIZER_DEPLOYMENT = os.environ.get("OPTIMIZER_DEPLOYMENT", "claude-sonnet-4-6")
TARGET_DEPLOYMENT = os.environ.get("TARGET_DEPLOYMENT", "claude-sonnet-4-6")
REASONING_EFFORT: str | None = None
tracker = TokenTracker()


def _assistant_message_schema() -> dict[str, Any]:
    return _codex._assistant_message_schema()


def _compat_message_from_payload(
    payload: dict[str, Any],
    *,
    tool_choice: str | dict[str, Any] | None = None,
):
    return _codex._compat_message_from_payload(payload, tool_choice=tool_choice)


def _chat_messages_impl(
    model: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    retries: int,
    stage: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[Any, dict[str, int]]:
    del max_completion_tokens  # Claude Code does not expose a completion-token cap
    last_err = None
    structured_output = bool(tools) or return_message
    schema = _assistant_message_schema() if structured_output else None
    # An explicit per-call effort wins; otherwise fall back to the value set via
    # set_reasoning_effort (the `--reasoning_effort` / config path).
    effort = reasoning_effort if reasoning_effort is not None else REASONING_EFFORT

    for attempt in range(retries):
        try:
            system, prompt, attachments = _build_prompt_from_messages(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                structured_output=structured_output,
            )
            if attachments:
                raise RuntimeError(
                    "claude_code_exec backend does not support image attachments"
                )
            raw_text, usage_info = run_claude_code_chat(
                system=system,
                prompt=prompt,
                model=model,
                timeout=timeout,
                schema=schema,
                effort=effort,
            )
            tracker.record(
                stage,
                usage_info["prompt_tokens"],
                usage_info["completion_tokens"],
            )
            if not structured_output:
                return raw_text, usage_info
            payload = json.loads(raw_text)
            compat = _compat_message_from_payload(payload, tool_choice=tool_choice)
            return (compat if return_message else compat.content), usage_info
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"Claude Code call failed after {retries} retries: {last_err}")


def chat_with_model(
    model: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    timeout: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, int]]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _chat_messages_impl(
        model,
        messages,
        max_completion_tokens,
        retries,
        stage,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


def chat_messages_with_model(
    model: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[Any, dict[str, int]]:
    return _chat_messages_impl(
        model,
        messages,
        max_completion_tokens,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    timeout: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, int]]:
    return chat_with_model(
        model=OPTIMIZER_DEPLOYMENT,
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    timeout: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, int]]:
    return chat_with_model(
        model=TARGET_DEPLOYMENT,
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


def chat_with_deployment(
    deployment: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    timeout: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, int]]:
    return chat_with_model(
        model=deployment,
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
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
    reasoning_effort: str | None = None,
) -> tuple[Any, dict[str, int]]:
    return _chat_messages_impl(
        OPTIMIZER_DEPLOYMENT,
        messages,
        max_completion_tokens,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
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
    reasoning_effort: str | None = None,
) -> tuple[Any, dict[str, int]]:
    return _chat_messages_impl(
        TARGET_DEPLOYMENT,
        messages,
        max_completion_tokens,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
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
    reasoning_effort: str | None = None,
) -> tuple[Any, dict[str, int]]:
    return _chat_messages_impl(
        deployment,
        messages,
        max_completion_tokens,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    global REASONING_EFFORT
    REASONING_EFFORT = effort if effort else None


def set_target_deployment(deployment: str) -> None:
    global TARGET_DEPLOYMENT
    TARGET_DEPLOYMENT = deployment
    os.environ["TARGET_DEPLOYMENT"] = deployment


def set_optimizer_deployment(deployment: str) -> None:
    global OPTIMIZER_DEPLOYMENT
    OPTIMIZER_DEPLOYMENT = deployment
    os.environ["OPTIMIZER_DEPLOYMENT"] = deployment

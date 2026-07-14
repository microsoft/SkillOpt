"""Hermes CLI chat backend for SkillOpt.

Chama `hermes --profile <name> chat -q "<prompt>"` como target/optimizer.
Possui token tracker próprio (separado do Claude/OpenAI) para evitar
double-count. Profiles são mutáveis via set_target_deployment /
set_optimizer_deployment.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Any

from skillopt.model.common import (
    CompatAssistantMessage,
    TokenTracker,
    default_model_for_backend,
)

HERMES_BIN = os.environ.get("HERMES_BIN", "hermes")

# Profiles mutáveis — setters alteram estas variáveis
_target_profile: str = os.environ.get("HERMES_TARGET_PROFILE", "default")
_optimizer_profile: str = os.environ.get("HERMES_OPTIMIZER_PROFILE", "default")

# Token tracker próprio — não usa o global de common.py
_hermes_tracker = TokenTracker()


def _call_hermes(
    prompt: str,
    profile: str,
    *,
    retries: int = 3,
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    """Call hermes CLI and return (response_text, token_info).

    Retries on non-zero exit with exponential backoff.
    Sets ``last_call_error`` on the module on persistent failure.
    """
    cmd = [HERMES_BIN, "--profile", profile, "chat", "-q", prompt]
    last_err: Exception | None = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or 180,
                env={**os.environ, "HERMES_NO_COLOR": "1"},
            )
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 10))
            continue
        elapsed = time.time() - t0
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            last_err = RuntimeError(stderr or f"Hermes CLI exited with code {proc.returncode}")
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 10))
            continue
        text = (proc.stdout or "").strip()
        tokens_in = len(prompt) // 4
        tokens_out = len(text) // 4
        return text, {
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        }
    raise RuntimeError(
        f"Hermes CLI failed after {retries} retries: {last_err}"
    ) from last_err


# ── System + User (string) APIs ─────────────────────────────────────────────


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 3,
    stage: str = "optimizer",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    """Call Hermes as optimizer."""
    del max_completion_tokens
    prompt = _build_prompt(system, user)
    text, usage = _call_hermes(prompt, _optimizer_profile, retries=retries, timeout=timeout)
    _hermes_tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
    return text, usage


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 3,
    stage: str = "target",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    """Call Hermes as target."""
    del max_completion_tokens
    prompt = _build_prompt(system, user)
    text, usage = _call_hermes(prompt, _target_profile, retries=retries, timeout=timeout)
    _hermes_tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
    return text, usage


def chat_with_deployment(
    deployment: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 3,
    stage: str = "custom",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    """Call Hermes with a custom profile name as deployment."""
    del max_completion_tokens
    profile = deployment or _target_profile
    prompt = _build_prompt(system, user)
    text, usage = _call_hermes(prompt, profile, retries=retries, timeout=timeout)
    _hermes_tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
    return text, usage


# ── Message-based APIs (tool-using benchmarks) ───────────────────────────────


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Flatten a message list into a single prompt string.

    Includes tool definitions, tool calls, and tool results.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            content = "\n".join(texts)
        parts.append(f"<{role}>\n{content}")
        # Include tool_calls if present
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                parts.append(
                    f"  [tool_call: {fn.get('name', '')}]\n  args: {fn.get('arguments', '')}"
                )
        # Include tool_name + content for tool-role messages
        tool_name = msg.get("tool_name")
        if tool_name:
            parts.append(f"  [tool_result from: {tool_name}]")
    return "\n".join(parts)


def _build_prompt(system: str, user: str) -> str:
    """Build a prompt string from system + user messages."""
    parts: list[str] = []
    if system:
        parts.append(system)
    if user:
        parts.append(user)
    return "\n\n".join(parts)


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 3,
    stage: str = "optimizer",
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict[str, int]]:
    """Call Hermes with a list of messages.

    * ``tools`` / ``tool_choice`` are serialised into the prompt preamble.
    * ``retries`` is respected.
    * ``return_message=True`` returns a ``CompatAssistantMessage`` instead of raw text.
    """
    del max_completion_tokens
    prompt = _build_message_prompt(messages, tools=tools, tool_choice=tool_choice)
    text, usage = _call_hermes(
        prompt, _optimizer_profile, retries=retries, timeout=timeout
    )
    _hermes_tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
    if return_message:
        return CompatAssistantMessage(content=text), usage
    return text, usage


def chat_target_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 3,
    stage: str = "target",
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict[str, int]]:
    """Call Hermes as target with messages."""
    del max_completion_tokens
    prompt = _build_message_prompt(messages, tools=tools, tool_choice=tool_choice)
    text, usage = _call_hermes(
        prompt, _target_profile, retries=retries, timeout=timeout
    )
    _hermes_tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
    if return_message:
        return CompatAssistantMessage(content=text), usage
    return text, usage


def chat_messages_with_deployment(
    deployment: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 3,
    stage: str = "custom",
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict[str, int]]:
    """Call Hermes with a custom profile and messages."""
    del max_completion_tokens
    profile = deployment or _target_profile
    prompt = _build_message_prompt(messages, tools=tools, tool_choice=tool_choice)
    text, usage = _call_hermes(prompt, profile, retries=retries, timeout=timeout)
    _hermes_tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
    if return_message:
        return CompatAssistantMessage(content=text), usage
    return text, usage


def _build_message_prompt(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> str:
    """Build a flat prompt from messages + optional tool definitions."""
    parts: list[str] = []
    if tools:
        import json
        parts.append("# Available tools")
        for t in tools:
            parts.append(json.dumps(t, indent=2))
        if tool_choice:
            parts.append(f"# Tool choice: {tool_choice}")
        parts.append("")
    parts.append(_flatten_messages(messages))
    return "\n".join(parts)


# ── Deployment setters ───────────────────────────────────────────────────────


def set_target_deployment(deployment: str) -> None:
    """Set the Hermes profile used by ``chat_target`` and friends.

    ``deployment`` is interpreted as a Hermes profile name.
    """
    global _target_profile
    _target_profile = deployment or default_model_for_backend("hermes")
    os.environ["HERMES_TARGET_PROFILE"] = _target_profile


def set_optimizer_deployment(deployment: str) -> None:
    """Set the Hermes profile used by ``chat_optimizer``.

    ``deployment`` is interpreted as a Hermes profile name.
    """
    global _optimizer_profile
    _optimizer_profile = deployment or default_model_for_backend("hermes")
    os.environ["HERMES_OPTIMIZER_PROFILE"] = _optimizer_profile


# ── Token tracking ────────────────────────────────────────────────────────────


def get_token_summary() -> dict[str, dict[str, int]]:
    return _hermes_tracker.summary()


def reset_token_tracker() -> None:
    _hermes_tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    pass  # Not applicable for Hermes

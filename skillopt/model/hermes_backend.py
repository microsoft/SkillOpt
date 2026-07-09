"""Hermes CLI chat backend for SkillOpt.

Chama `hermes --profile <name> chat -q "<prompt>"` como target/optimizer.
Mais simples que claude_backend: sem tools, imagens, ou attachments.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

from skillopt.model.common import CompatAssistantMessage, CompatToolCall, CompatToolFunction, default_model_for_backend, tracker

HERMES_BIN = os.environ.get("HERMES_BIN", "hermes")
HERMES_TARGET_PROFILE = os.environ.get("HERMES_TARGET_PROFILE", "default")
HERMES_OPTIMIZER_PROFILE = os.environ.get("HERMES_OPTIMIZER_PROFILE", "default")

OPTIMIZER_DEPLOYMENT = os.environ.get("OPTIMIZER_DEPLOYMENT", "default")
TARGET_DEPLOYMENT = os.environ.get("TARGET_DEPLOYMENT", "default")


def _call_hermes(prompt: str, profile: str, timeout: int | None = None) -> tuple[str, dict[str, int]]:
    """Call hermes CLI and return (response_text, token_info)."""
    cmd = [HERMES_BIN, "--profile", profile, "chat", "-q", prompt]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout or 180,
        env={**os.environ, "HERMES_NO_COLOR": "1"},
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(stderr or f"Hermes CLI exited with code {proc.returncode}")

    text = (proc.stdout or "").strip()
    tokens_in = len(prompt) // 4
    tokens_out = len(text) // 4
    return text, {
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "total_tokens": tokens_in + tokens_out,
    }


def _build_prompt(system: str, user: str) -> str:
    """Build a prompt string from system + user messages."""
    parts = []
    if system:
        parts.append(system)
    if user:
        parts.append(user)
    return "\n\n".join(parts)


def chat_optimizer(system: str, user: str, max_completion_tokens: int = 16384, retries: int = 3, stage: str = "optimizer", timeout: int | None = None) -> tuple[str, dict[str, int]]:
    """Call Hermes as optimizer with profile=target."""
    del max_completion_tokens
    prompt = _build_prompt(system, user)
    last_err = None
    for attempt in range(retries):
        try:
            text, usage = _call_hermes(prompt, HERMES_OPTIMIZER_PROFILE, timeout=timeout)
            tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
            return text, usage
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"Hermes optimizer backend failed after {retries} retries: {last_err}")


def chat_target(system: str, user: str, max_completion_tokens: int = 16384, retries: int = 3, stage: str = "target", timeout: int | None = None) -> tuple[str, dict[str, int]]:
    """Call Hermes as target with profile=target."""
    del max_completion_tokens
    prompt = _build_prompt(system, user)
    last_err = None
    for attempt in range(retries):
        try:
            text, usage = _call_hermes(prompt, HERMES_TARGET_PROFILE, timeout=timeout)
            tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
            return text, usage
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"Hermes target backend failed after {retries} retries: {last_err}")


def chat_with_deployment(deployment: str, system: str, user: str, max_completion_tokens: int = 16384, retries: int = 3, stage: str = "custom", timeout: int | None = None) -> tuple[str, dict[str, int]]:
    """Call Hermes with a custom profile name as deployment."""
    del max_completion_tokens
    profile = deployment or HERMES_TARGET_PROFILE
    prompt = _build_prompt(system, user)
    last_err = None
    for attempt in range(retries):
        try:
            text, usage = _call_hermes(prompt, profile, timeout=timeout)
            tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
            return text, usage
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"Hermes backend (deployment={deployment}) failed after {retries} retries: {last_err}")


# ── Message-based variants (needed for tool-using benchmarks like spreadsheetbench) ──

def chat_optimizer_messages(messages: list[dict[str, Any]], max_completion_tokens: int = 16384, retries: int = 3, stage: str = "optimizer", *, tools: list[dict[str, Any]] | None = None, tool_choice: str | dict[str, Any] | None = None, return_message: bool = False, timeout: int | None = None) -> tuple[Any, dict[str, int]]:
    """Simplified: flatten messages to prompt text."""
    del max_completion_tokens, tools, tool_choice, return_message
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = "\n".join(texts)
        parts.append(f"<{role}>\n{content}")
    prompt = "\n".join(parts)
    text, usage = _call_hermes(prompt, HERMES_OPTIMIZER_PROFILE, timeout=timeout)
    tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
    return text, usage


def chat_target_messages(messages: list[dict[str, Any]], max_completion_tokens: int = 16384, retries: int = 3, stage: str = "target", *, tools: list[dict[str, Any]] | None = None, tool_choice: str | dict[str, Any] | None = None, return_message: bool = False, timeout: int | None = None) -> tuple[Any, dict[str, int]]:
    """Simplified: flatten messages to prompt text."""
    del max_completion_tokens, tools, tool_choice, return_message
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = "\n".join(texts)
        parts.append(f"<{role}>\n{content}")
    prompt = "\n".join(parts)
    text, usage = _call_hermes(prompt, HERMES_TARGET_PROFILE, timeout=timeout)
    tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
    return text, usage


def chat_messages_with_deployment(deployment: str, messages: list[dict[str, Any]], max_completion_tokens: int = 16384, retries: int = 3, stage: str = "custom", *, tools: list[dict[str, Any]] | None = None, tool_choice: str | dict[str, Any] | None = None, return_message: bool = False, timeout: int | None = None) -> tuple[Any, dict[str, int]]:
    """Simplified: flatten messages to prompt text."""
    del max_completion_tokens, tools, tool_choice, return_message
    profile = deployment or HERMES_TARGET_PROFILE
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = "\n".join(texts)
        parts.append(f"<{role}>\n{content}")
    prompt = "\n".join(parts)
    text, usage = _call_hermes(prompt, profile, timeout=timeout)
    tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
    return text, usage


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    pass  # Not applicable for Hermes


def set_target_deployment(deployment: str) -> None:
    global TARGET_DEPLOYMENT
    TARGET_DEPLOYMENT = deployment or default_model_for_backend("hermes")
    os.environ["TARGET_DEPLOYMENT"] = TARGET_DEPLOYMENT


def set_optimizer_deployment(deployment: str) -> None:
    global OPTIMIZER_DEPLOYMENT
    OPTIMIZER_DEPLOYMENT = deployment or default_model_for_backend("hermes")
    os.environ["OPTIMIZER_DEPLOYMENT"] = OPTIMIZER_DEPLOYMENT

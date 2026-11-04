"""OpenCode CLI backend (``opencode_exec``): the optimizer chat calls and shared helpers.

Each call runs ``opencode run --standalone --format json`` with a private
config home. Its global ``permissions`` list starts with deny-all and is the
only gate: the agent entry carries no rules (OpenCode 2.x drops a per-agent
``permissions`` list, though it does apply a per-agent ``permission`` map). The private
config carries none of the user's MCP servers, and the ``OPENCODE_DISABLE_*``
switches keep project ``AGENTS.md`` files and Claude Code rules and skills out.
The user's login is reused: only ``XDG_CONFIG_HOME`` moves, the data directory
with the credentials stays where it is.

The optimizer runs in an empty directory with every tool denied: it reads
rollout trajectories, which may hold untrusted text, so it must never act.
Tool calls through the chat API are rejected explicitly. The exec target lives
in ``codex_harness.run_opencode_exec`` and uses the helpers below.

Targets OpenCode 2.x (``--standalone``, ``permissions`` lists).
"""
from __future__ import annotations

import functools
import json
import os
import re
import secrets
import subprocess
import tempfile
import time
from typing import Any

from skillopt.model.backend_config import get_opencode_exec_config
from skillopt.model.common import TokenTracker, default_model_for_backend

OPTIMIZER_DEPLOYMENT = os.environ.get("OPENCODE_EXEC_OPTIMIZER_MODEL") or default_model_for_backend("opencode_exec")
TARGET_DEPLOYMENT = OPTIMIZER_DEPLOYMENT

REASONING_EFFORT: str | None = None
tracker = TokenTracker()

DENY_ALL = {"action": "*", "resource": "*", "effect": "deny"}
READ_TOOLS = ("read", "grep", "glob", "list")
# Variables that would point the child at a config other than the private one.
_CONFIG_ENV = ("OPENCODE_CONFIG", "OPENCODE_CONFIG_DIR", "OPENCODE_CONFIG_CONTENT")
_STDERR_TAIL = 400


class OpenCodeExecError(RuntimeError):
    """An OpenCode call failed; the message is safe to show and log."""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content or "")


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """System messages become the agent prompt; the rest is the stdin prompt."""
    system: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        text = _content_text(message.get("content", "")).strip()
        if role == "system":
            if text:
                system.append(text)
        else:
            turns.append((role, text))
    if len(turns) == 1 and turns[0][0] == "user":
        return "\n\n".join(system), turns[0][1]
    history = "\n\n".join(f"[{role}]\n{text}" for role, text in turns)
    return "\n\n".join(system), history + "\n\nAnswer the latest user message."


def permission_rules(
    *,
    read: bool = False,
    edit: bool = False,
    shell: bool = False,
    external_dirs: list[str] | None = None,
) -> list[dict[str, str]]:
    """Deny-all first, then only what the run needs. Later rules win in OpenCode."""
    rules = [dict(DENY_ALL)]
    actions = (READ_TOOLS if read else ()) + (("edit",) if edit else ()) + (("shell",) if shell else ())
    rules += [{"action": action, "resource": "*", "effect": "allow"} for action in actions]
    rules += [
        {"action": "external_directory", "resource": os.path.join(path, "*"), "effect": "allow"}
        for path in external_dirs or []
    ]
    return rules


def write_private_config(root: str, *, agent: str, permissions: list[dict[str, str]], system: str = "") -> str:
    """Write ``<root>/config/opencode/opencode.json``; return the config home for ``XDG_CONFIG_HOME``."""
    config_home = os.path.join(root, "config")
    os.makedirs(os.path.join(config_home, "opencode"), exist_ok=True)
    agent_cfg: dict[str, Any] = {"mode": "primary"}
    if system:
        agent_cfg["prompt"] = system
    with open(os.path.join(config_home, "opencode", "opencode.json"), "w", encoding="utf-8") as f:
        json.dump({"permissions": permissions, "agent": {agent: agent_cfg}}, f, ensure_ascii=False)
    return config_home


@functools.lru_cache(maxsize=8)
def _cli_version(path: str) -> str:
    try:
        proc = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        # Not runnable: the call itself reports that with its own error.
        return ""
    return (proc.stdout or proc.stderr or "").strip()


def require_supported_cli(path: str) -> None:
    """Refuse OpenCode 1.x: the flags and config format here are OpenCode 2.x."""
    version = _cli_version(path)
    match = re.search(r"(\d+)\.\d+", version)
    if match and int(match.group(1)) < 2:
        raise OpenCodeExecError(
            f"opencode_exec needs OpenCode 2.x (--standalone, permissions lists); {path} reports {version!r}"
        )


def new_agent_name(role: str) -> str:
    return f"skillopt-{role}-{secrets.token_hex(8)}"


def parse_events(stdout: str, *, allow_tools: bool) -> tuple[str, dict[str, int], str, list[dict[str, Any]]]:
    """Final text, usage, an error message and tool steps from ``--format json`` output."""
    texts: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    error = ""
    steps: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            return "", usage, "OpenCode returned malformed JSONL", steps
        if not isinstance(event, dict):
            return "", usage, "OpenCode returned malformed JSONL", steps
        kind = event.get("type")
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        if kind == "error":
            detail = event.get("error") if isinstance(event.get("error"), dict) else {}
            error = str(detail.get("message") or detail.get("type") or "OpenCode reported an error")
        elif kind == "text" and part.get("type") == "text":
            texts.append(str(part.get("text", "")))
        elif kind == "tool_use":
            if not allow_tools:
                # Tools are denied; a tool event means the isolation did not hold.
                return "", usage, "OpenCode attempted a tool call although every tool is denied", steps
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            steps.append(
                {
                    "tool": str(part.get("tool", "")),
                    "status": str(state.get("status", "")),
                    "input": state.get("input"),
                    "output": state.get("output") if state.get("status") != "error" else state.get("error"),
                }
            )
        elif kind == "step_finish":
            tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
            prompt = int(tokens.get("input", 0) or 0)
            completion = int(tokens.get("output", 0) or 0) + int(tokens.get("reasoning", 0) or 0)
            usage["prompt_tokens"] += prompt
            usage["completion_tokens"] += completion
            usage["total_tokens"] += prompt + completion
    return "".join(texts).strip(), usage, error, steps


def child_env(config_home: str, work_dir: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in _CONFIG_ENV and key != "OLDPWD"}
    env.update(
        {
            # OpenCode takes its project directory from PWD, not the process cwd.
            "PWD": work_dir,
            "XDG_CONFIG_HOME": config_home,
            "NO_COLOR": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_SHARE": "1",
            # Project AGENTS.md up to the git root, and Claude Code rules and skills.
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_CONFIG_PROJECT_DISABLE": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
        }
    )
    return env


def _run_once(model: str, system: str, prompt: str, timeout: int | None) -> tuple[str, dict[str, int]]:
    config = get_opencode_exec_config()
    agent = new_agent_name("optimizer")
    with tempfile.TemporaryDirectory(prefix="skillopt_opencode_") as root:
        config_home = write_private_config(root, agent=agent, permissions=permission_rules(), system=system)
        work = os.path.join(root, "work")
        os.makedirs(work)

        command = [str(config["path"]), "run", "--standalone", "--format", "json", "--agent", agent]
        if model:
            command.extend(["-m", model])
        try:
            proc = subprocess.run(
                command,
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
                cwd=work,
                env=child_env(config_home, work),
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise OpenCodeExecError(f"OpenCode timed out after {timeout}s") from None
        except OSError as exc:
            raise OpenCodeExecError(f"OpenCode could not be started: {exc.strerror or exc}") from None

    text, usage, error, _steps = parse_events(proc.stdout or "", allow_tools=False)
    if error:
        raise OpenCodeExecError(error)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-_STDERR_TAIL:]
        raise OpenCodeExecError(f"OpenCode exited {proc.returncode}" + (f": {tail}" if tail else ""))
    if not text:
        raise OpenCodeExecError("OpenCode returned an empty final message")
    return text, usage


def _chat(
    model: str,
    messages: list[dict[str, Any]],
    retries: int,
    stage: str,
    timeout: int | None,
) -> tuple[str, dict[str, int]]:
    require_supported_cli(str(get_opencode_exec_config()["path"]))
    system, prompt = _split_messages(messages)
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            text, usage = _run_once(model, system, prompt, timeout)
            tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])
            return text, usage
        except OpenCodeExecError as exc:
            last_err = exc
        if attempt + 1 < retries:
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenCode call failed after {max(1, retries)} attempt(s): {last_err}")


def chat_with_model(
    model: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    del max_completion_tokens
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _chat(model, messages, retries, stage, timeout)


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    return chat_with_model(OPTIMIZER_DEPLOYMENT, system, user, max_completion_tokens, retries, stage, timeout)


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
    del max_completion_tokens, tool_choice
    if tools or return_message:
        raise NotImplementedError("opencode_exec is a text-only optimizer backend; tool calls are not supported")
    return _chat(OPTIMIZER_DEPLOYMENT, messages, retries, stage, timeout)


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    # OpenCode picks reasoning per model variant (``provider/model#variant``);
    # the shared setting is kept for parity and not translated.
    global REASONING_EFFORT
    REASONING_EFFORT = effort if effort else None


def set_optimizer_deployment(deployment: str) -> None:
    global OPTIMIZER_DEPLOYMENT
    OPTIMIZER_DEPLOYMENT = deployment


def set_target_deployment(deployment: str) -> None:
    global TARGET_DEPLOYMENT
    TARGET_DEPLOYMENT = deployment

"""Helpers for running exec backends as the target harness."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import threading
import traceback
from typing import Any

from skillopt.model.backend_config import (
    get_claude_code_exec_config,
    get_codex_exec_config,
    get_target_backend,
)


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "final_response": {
            "type": "string",
            "description": "The exact final answer text to return, preserving required <answer>...</answer> tags.",
        },
        "final_answer": {
            "type": "string",
            "description": "The concise answer value without explanation, if separable.",
        },
    },
    "required": ["final_response", "final_answer"],
    "additionalProperties": False,
}


def render_skill_md(
    skill_content: str,
    *,
    name: str = "skillopt-target",
    description: str = "Dynamic ReflACT skill for the current benchmark task.",
    preamble: str = "",
) -> str:
    body = skill_content.strip() or "No additional dynamic guidance was provided for this task."
    chunks = [
        "---",
        f'name: "{name}"',
        f'description: "{description}"',
        "---",
        "",
        "# ReflACT Target Skill",
        "",
    ]
    if preamble.strip():
        chunks.append(preamble.strip())
        chunks.append("")
    chunks.extend([
        "## Dynamic Guidance",
        "",
        body,
        "",
    ])
    return "\n".join(chunks)


def prepare_workspace(
    *,
    work_dir: str,
    skill_md: str,
    task_text: str = "",
    task_filename: str = "task.md",
    images: list[str] | None = None,
    extra_files: dict[str, str] | None = None,
    copy_files: list[tuple[str, str]] | None = None,
    link_dirs: list[tuple[str, str]] | None = None,
) -> tuple[str, str]:
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(os.path.join(work_dir, ".agents", "skills", "skillopt-target"), exist_ok=True)

    skill_path = os.path.join(work_dir, ".agents", "skills", "skillopt-target", "SKILL.md")
    with open(skill_path, "w", encoding="utf-8") as f:
        f.write(skill_md)

    task_path = os.path.join(work_dir, task_filename)
    if task_text:
        with open(task_path, "w", encoding="utf-8") as f:
            f.write(task_text)

    if extra_files:
        for rel_path, content in extra_files.items():
            full_path = os.path.join(work_dir, rel_path)
            parent = os.path.dirname(full_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

    if copy_files:
        for src, rel_dst in copy_files:
            dst = os.path.join(work_dir, rel_dst)
            parent = os.path.dirname(dst)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.copy2(src, dst)

    if link_dirs:
        for src, rel_dst in link_dirs:
            dst = os.path.join(work_dir, rel_dst)
            parent = os.path.dirname(dst)
            if parent:
                os.makedirs(parent, exist_ok=True)
            os.symlink(os.path.abspath(src), dst)

    attachment_lines: list[str] = []
    if images:
        attachments_dir = os.path.join(work_dir, "attachments")
        os.makedirs(attachments_dir, exist_ok=True)
        for index, image in enumerate(images, 1):
            if not os.path.exists(image):
                raise FileNotFoundError(image)
            src = os.path.abspath(image)
            base = os.path.basename(src) or f"image_{index}"
            dst_name = f"{index:02d}_{base}"
            dst = os.path.join(attachments_dir, dst_name)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
            rel_dst = os.path.relpath(dst, work_dir)
            attachment_lines.append(f"- `{rel_dst}` (source: `{src}`)")

    if attachment_lines:
        with open(os.path.join(work_dir, "ATTACHMENTS.md"), "w", encoding="utf-8") as f:
            f.write(
                "# Attachments\n\n"
                "Use these local files when the task refers to attached images or documents.\n\n"
                + "\n".join(attachment_lines)
                + "\n"
            )

    return skill_path, task_path


def _build_codex_trace_summary(raw: str, response: str) -> str:
    lines = [ln.rstrip() for ln in (raw or "").splitlines()]

    def _find(prefix: str) -> str:
        for ln in lines:
            if ln.startswith(prefix):
                return ln[len(prefix):].strip()
        return ""

    sandbox = _find("sandbox: ")
    reasoning = _find("reasoning effort: ")
    task_read = "unknown"
    skill_read = "unknown"
    exec_errors: list[str] = []
    tokens_used = ""

    for idx, ln in enumerate(lines):
        if ln.startswith("exec"):
            cmd = lines[idx + 1] if idx + 1 < len(lines) else ""
            outcome = lines[idx + 2] if idx + 2 < len(lines) else ""
            joined = f"{cmd}\n{outcome}"
            if "task.md" in joined:
                if "succeeded" in outcome:
                    task_read = "success"
                elif "failed" in outcome or "ERROR" in outcome:
                    task_read = "failed"
            if "SKILL.md" in joined:
                if "succeeded" in outcome:
                    skill_read = "success"
                elif "failed" in outcome or "ERROR" in outcome:
                    skill_read = "failed"
        if ln.startswith("ERROR:"):
            exec_errors.append(ln[len("ERROR:"):].strip())
        if ln == "tokens used" and idx + 1 < len(lines):
            tokens_used = lines[idx + 1].strip()

    match = re.search(r"<answer>\s*([A-E])\s*</answer>", response or "", re.IGNORECASE)
    if match:
        answer_format = "well_formed"
        answer_label = match.group(1).upper()
    elif "<answer>" in (response or "").lower():
        answer_format = "tagged_nonlabel"
        answer_label = ""
    elif (response or "").strip():
        answer_format = "plain_text"
        answer_label = ""
    else:
        answer_format = "missing"
        answer_label = ""

    parts = ["Codex Trace Summary"]
    if sandbox:
        parts.append(f"- sandbox: {sandbox}")
    if reasoning:
        parts.append(f"- reasoning: {reasoning}")
    parts.append(f"- read task.md: {task_read}")
    parts.append(f"- read SKILL.md: {skill_read}")
    if exec_errors:
        parts.append(f"- shell/tool errors: {' | '.join(exec_errors[:3])}")
    else:
        parts.append("- shell/tool errors: none")
    parts.append(f"- final answer format: {answer_format}")
    parts.append(f"- final answer label: {answer_label or '(none)'}")
    if tokens_used:
        parts.append(f"- tokens used: {tokens_used}")
    return "\n".join(parts)


def _build_claude_trace_summary(raw: str, response: str) -> str:
    answer_format = "missing"
    if "<answer>" in (response or "").lower():
        answer_format = "tagged"
    elif (response or "").strip():
        answer_format = "plain_text"
    errors: list[str] = []
    for ln in (raw or "").splitlines():
        if "error" in ln.lower() or "traceback" in ln.lower():
            errors.append(ln.strip())
        if len(errors) >= 3:
            break
    parts = ["Claude Code Trace Summary", f"- final answer format: {answer_format}"]
    parts.append(f"- final response chars: {len(response or '')}")
    parts.append(f"- errors: {' | '.join(errors) if errors else 'none'}")
    return "\n".join(parts)


def _persist_artifacts(
    *,
    work_dir: str,
    raw: str,
    response: str,
    prefix: str,
    summary_builder,
) -> None:
    pred_dir = os.path.dirname(work_dir.rstrip(os.sep))
    raw_path = os.path.join(pred_dir, f"{prefix}_raw.txt")
    summary_path = os.path.join(pred_dir, f"{prefix}_trace_summary.txt")

    combined_raw = raw
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            prev = f.read()
        combined_raw = f"{prev}\n\n===== TURN BREAK =====\n\n{raw}" if prev.strip() else raw

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(combined_raw)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_builder(combined_raw, response))


def _persist_codex_artifacts(work_dir: str, raw: str, response: str) -> None:
    _persist_artifacts(
        work_dir=work_dir,
        raw=raw,
        response=response,
        prefix="codex",
        summary_builder=_build_codex_trace_summary,
    )


def _persist_claude_artifacts(work_dir: str, raw: str, response: str) -> None:
    _persist_artifacts(
        work_dir=work_dir,
        raw=raw,
        response=response,
        prefix="claude",
        summary_builder=_build_claude_trace_summary,
    )


def parse_codex_raw(raw: str) -> dict:
    """Parse raw Codex CLI output into step sections.

    Returns a dict with:
    - ``steps``: ordered sections beginning at the first ``user/codex/exec`` marker
    - ``trace_body``: raw trace starting at the first marker
    """
    lines = (raw or "").splitlines()
    markers = {"user", "codex", "exec"}
    first_step_line: int | None = None
    for idx, line in enumerate(lines):
        if line in markers:
            first_step_line = idx
            break
    if first_step_line is None:
        return {"steps": [], "trace_body": ""}

    steps: list[dict] = []
    current: dict | None = None
    for idx in range(first_step_line, len(lines)):
        line = lines[idx]
        if line in markers:
            if current is not None:
                current["end_line"] = idx
                current["content"] = "\n".join(current["content_lines"]).strip()
                current.pop("content_lines", None)
                steps.append(current)
            current = {
                "index": len(steps) + 1,
                "type": line,
                "start_line": idx,
                "content_lines": [],
            }
            continue
        if current is not None:
            current["content_lines"].append(line)
    if current is not None:
        current["end_line"] = len(lines)
        current["content"] = "\n".join(current["content_lines"]).strip()
        current.pop("content_lines", None)
        steps.append(current)

    trace_body = "\n".join(lines[first_step_line:]).strip()
    return {"steps": steps, "trace_body": trace_body}


def format_codex_trace_steps(raw: str, *, max_chars: int = 4000) -> str:
    """Render parsed Codex trace into numbered compact steps for optimizer prompts."""
    parsed = parse_codex_raw(raw)
    steps = parsed["steps"]
    if not steps:
        return ""

    rendered: list[str] = []
    for step in steps:
        summary = ""
        content = str(step.get("content") or "").strip()
        if step["type"] == "exec":
            body_lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            cmd = body_lines[0] if body_lines else ""
            status = ""
            for ln in body_lines[1:]:
                low = ln.lower()
                if "succeeded in" in low or "failed in" in low or "timed out" in low or low.startswith("error"):
                    status = ln
                    break
            summary = cmd
            if status:
                summary = f"{summary} | {status}" if summary else status
        else:
            summary = " ".join(content.splitlines())
        summary = summary[:500] if summary else "(empty)"
        rendered.append(f"[{step['index']}] {step['type']}: {summary}")

    text = "\n".join(rendered)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[trace steps truncated]..."
    return text


def extract_codex_trace_prefix(raw: str, *, after_step: int) -> str:
    """Return raw trace body up to and including ``after_step``.

    ``after_step <= 0`` yields an empty string.
    """
    if after_step <= 0:
        return ""
    parsed = parse_codex_raw(raw)
    steps = parsed["steps"]
    if not steps:
        return ""
    clamped = min(after_step, len(steps))
    lines = parsed["trace_body"].splitlines()
    end_line = int(steps[clamped - 1]["end_line"]) - int(steps[0]["start_line"])
    return "\n".join(lines[:end_line]).strip()


_DENIED_DATA_DIR_NAMES = {"officeqa_split", "sealqa_split"}


def _normalize_tools(allowed_tools: list[str] | str | None) -> str:
    if allowed_tools is None:
        return ""
    if isinstance(allowed_tools, str):
        return ",".join(part.strip() for part in allowed_tools.split(",") if part.strip())
    return ",".join(str(tool).strip() for tool in allowed_tools if str(tool).strip())


def _tools_list(allowed_tools: list[str] | str | None) -> list[str]:
    tools = _normalize_tools(allowed_tools)
    return [part.strip() for part in tools.split(",") if part.strip()]


def _validate_exec_path(path: str) -> str:
    resolved = os.path.realpath(os.path.abspath(path))
    parts = set(resolved.split(os.sep))
    denied = parts & _DENIED_DATA_DIR_NAMES
    if denied:
        raise ValueError(f"Refusing to expose denied data directory to exec backend: {', '.join(sorted(denied))}")
    return resolved


def _validated_add_dirs(work_dir: str, data_dirs: list[str] | None, images: list[str] | None) -> list[str]:
    add_dirs = [_validate_exec_path(work_dir)]
    for data_dir in data_dirs or []:
        add_dirs.append(_validate_exec_path(data_dir))
    for image in images or []:
        add_dirs.append(_validate_exec_path(os.path.dirname(image) or work_dir))
    deduped: list[str] = []
    for path in add_dirs:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _sdk_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in {"1", "true", "yes", "on", "sdk"}:
        return "sdk"
    if mode in {"0", "false", "no", "off", "cli"}:
        return "cli"
    return "auto"


def _claude_effort(value: Any) -> str:
    effort = str(value or "medium").strip().lower()
    if effort in {"", "none", "off"}:
        return ""
    if effort == "xhigh":
        return "max"
    if effort not in {"low", "medium", "high", "max"}:
        return "medium"
    return effort


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            box["exception"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()
    if "exception" in box:
        raise box["exception"]
    return box.get("result")


def _exec_prompt(prompt: str, *, allow_file_edits: bool = False) -> str:
    edit_instruction = (
        "You may modify files in the workspace when the task asks you to create an artifact. "
        if allow_file_edits
        else "Do not modify files. "
    )
    return (
        "Use the workspace files to solve the task. Read task.md and the skill at "
        ".agents/skills/skillopt-target/SKILL.md before answering. "
        "If ATTACHMENTS.md exists, read it and inspect the listed local files. "
        "Do not call a Skill tool; the ReflACT guidance is a local markdown file. "
        f"Do not ask for permission. {edit_instruction}"
        "Return only the final answer text, keeping any required <answer>...</answer> tags exactly.\n\n"
        f"{_normalize_target_exec_prompt(prompt)}"
    )


def _retry_prompt(prompt: str, attempt: int) -> str:
    if attempt <= 0:
        return prompt
    return (
        f"{prompt}\n\n"
        "Previous execution returned an empty final response. Re-read task.md and "
        ".agents/skills/skillopt-target/SKILL.md. If ATTACHMENTS.md exists, use the listed files. "
        "Then produce the final answer inside <answer>...</answer>."
    )


def _normalize_target_exec_prompt(prompt: str) -> str:
    """Avoid wording that makes Claude Code call an unregistered Skill tool."""
    text = prompt or ""
    replacements = {
        "Use the `skillopt-target` skill available in this workspace.": (
            "Read `.agents/skills/skillopt-target/SKILL.md` directly; do not call a Skill tool."
        ),
        "- Use the local `skillopt-target` skill before writing code.": (
            "- Read `.agents/skills/skillopt-target/SKILL.md` before writing code; do not call a Skill tool."
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict = json.loads(json.dumps(schema))
    strict["additionalProperties"] = False
    properties = strict.get("properties") or {}
    strict["required"] = list(properties.keys())
    return strict


def _structured_response(data: Any) -> tuple[str, str]:
    if not isinstance(data, dict):
        return "", f"Structured output was not an object: {type(data).__name__}"
    final_response = str(data.get("final_response") or "").strip()
    final_answer = str(data.get("final_answer") or "").strip()
    if final_response:
        return final_response, ""
    if final_answer:
        if "<answer>" in final_answer.lower():
            return final_answer, ""
        return f"<answer>{final_answer}</answer>", ""
    return "", "Structured output did not contain a final response."


def _extract_claude_structured_output(messages: list[Any]) -> Any:
    """Claude Code SDK can finish with error_during_execution after StructuredOutput."""
    for msg in reversed(messages):
        structured = getattr(msg, "structured_output", None)
        if isinstance(structured, dict):
            return structured

        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in reversed(content):
            name = getattr(item, "name", None)
            payload = getattr(item, "input", None)
            if isinstance(item, dict):
                name = item.get("name", name)
                payload = item.get("input", payload)
            if name == "StructuredOutput" and isinstance(payload, dict):
                return payload
    return None


def _raw_exception(label: str, exc: BaseException) -> str:
    return _json_dumps({
        "backend": label,
        "is_error": True,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    })


def _run_claude_code_sdk_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    allowed_tools: list[str] | str | None = None,
    permission_mode: str | None = None,
    allow_file_edits: bool = False,
) -> tuple[str, str]:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    async def _query() -> tuple[str, str]:
        system_prompt: dict[str, Any] = {
            "type": "preset",
            "preset": "claude_code",
            "append": (
                "Use the workspace files to solve the task. Read task.md and the skill at "
                ".agents/skills/skillopt-target/SKILL.md before answering. "
                "If ATTACHMENTS.md exists, read it and inspect the listed local files. "
                "Do not call a Skill tool; the ReflACT guidance is a local markdown file. "
                + (
                    "You may modify files in the workspace when the task asks you to create an artifact. "
                    if allow_file_edits
                    else "Do not modify files. "
                )
                + "Return structured output whose final_response preserves required <answer>...</answer> tags."
            ),
        }
        kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "output_format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            "allowed_tools": _tools_list(allowed_tools) or ["Read", "Bash"],
            "cwd": str(work_dir),
            "permission_mode": permission_mode or "bypassPermissions",
            "add_dirs": _validated_add_dirs(work_dir, data_dirs, images),
            "max_buffer_size": 8 * 1024 * 1024,
        }
        config = get_claude_code_exec_config()
        effort = _claude_effort(config.get("effort"))
        if effort:
            kwargs["effort"] = effort
        max_thinking_tokens = int(config.get("max_thinking_tokens", 0) or 0)
        if max_thinking_tokens > 0:
            kwargs["max_thinking_tokens"] = max_thinking_tokens
        options = ClaudeAgentOptions(**kwargs)
        if model:
            options.model = model.split("/", 1)[1] if model.startswith("anthropic/") else model

        messages = []
        async with ClaudeSDKClient(options) as client:
            await client.query(_normalize_target_exec_prompt(prompt))
            messages = [msg async for msg in client.receive_response()]
        last = messages[-1] if messages else None
        raw_structured_output = _extract_claude_structured_output(messages)
        response, parse_error = _structured_response(raw_structured_output)
        first = messages[0] if messages else None
        first_data = getattr(first, "data", {}) if first is not None else {}
        terminal_is_error = bool(getattr(last, "is_error", False)) if last is not None else False
        raw = _json_dumps({
            "backend": "claude_code_sdk",
            "uuid": first_data.get("uuid", "") if isinstance(first_data, dict) else "",
            "session_id": getattr(last, "session_id", "") if last is not None else "",
            "model": first_data.get("model", model) if isinstance(first_data, dict) else model,
            "tools": first_data.get("tools", _tools_list(allowed_tools)) if isinstance(first_data, dict) else _tools_list(allowed_tools),
            "duration_ms": getattr(last, "duration_ms", 0) if last is not None else 0,
            "total_cost_usd": getattr(last, "total_cost_usd", 0.0) if last is not None else 0.0,
            "num_turns": getattr(last, "num_turns", 0) if last is not None else 0,
            "usage": getattr(last, "usage", {}) if last is not None else {},
            "result": getattr(last, "result", "") if last is not None else "",
            "is_error": bool(parse_error) or (terminal_is_error and not response.strip()),
            "terminal_is_error": terminal_is_error,
            "parse_error": parse_error,
            "raw_structured_output": raw_structured_output,
            "messages": messages,
        })
        return response, raw

    return _run_async(asyncio.wait_for(_query(), timeout=timeout))


def _run_claude_code_cli_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    allowed_tools: list[str] | str | None = None,
    permission_mode: str | None = None,
    allow_file_edits: bool = False,
) -> tuple[str, str]:
    config = get_claude_code_exec_config()
    tools = "Read,Bash" if allowed_tools is None else _normalize_tools(allowed_tools)
    cmd = [
        str(config["path"]),
        "-p",
        "--output-format",
        "text",
        "--permission-mode",
        permission_mode or "bypassPermissions",
        "--add-dir",
        work_dir,
        "--tools",
        tools,
        "--allowedTools",
        tools,
    ]
    if config.get("profile"):
        cmd.extend(["--settings", '{"env":{"CLAUDE_CODE_USE_BEDROCK":"0"}}'])
        cmd.extend(["--append-system-prompt", f"Profile: {config['profile']}"])
    if model:
        cmd.extend(["--model", model])
    effort = _claude_effort(config.get("effort"))
    if effort:
        cmd.extend(["--effort", effort])
    max_thinking_tokens = int(config.get("max_thinking_tokens", 0) or 0)
    if max_thinking_tokens > 0:
        cmd.extend(["--max-thinking-tokens", str(max_thinking_tokens)])
    for data_dir in data_dirs or []:
        cmd.extend(["--add-dir", _validate_exec_path(data_dir)])
    if images:
        for image in images:
            cmd.extend(["--add-dir", _validate_exec_path(os.path.dirname(image) or work_dir)])
    cmd.extend(["--", _exec_prompt(prompt, allow_file_edits=allow_file_edits)])

    try:
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raw = stdout
        if stderr:
            raw = f"{raw}\n[stderr]\n{stderr}" if raw else stderr
        return "", raw

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    raw = stdout
    if stderr:
        raw = f"{raw}\n[stderr]\n{stderr}" if raw else stderr
    response = stdout.strip()
    if proc.returncode != 0 and not response:
        return "", raw
    return response, raw


def run_claude_code_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    allowed_tools: list[str] | str | None = None,
    permission_mode: str | None = None,
    allow_file_edits: bool = False,
) -> tuple[str, str]:
    config = get_claude_code_exec_config()
    mode = _sdk_mode(config.get("use_sdk"))
    retries = int(config.get("empty_response_retries", 0) or 0)
    last_response = ""
    all_raw: list[str] = []

    for attempt in range(retries + 1):
        attempt_prompt = _retry_prompt(prompt, attempt)
        if mode != "cli":
            try:
                response, raw = _run_claude_code_sdk_exec(
                    work_dir=work_dir,
                    prompt=attempt_prompt,
                    model=model,
                    timeout=timeout,
                    images=images,
                    data_dirs=data_dirs,
                    allowed_tools=allowed_tools,
                    permission_mode=permission_mode,
                    allow_file_edits=allow_file_edits,
                )
                all_raw.append(f"===== CLAUDE SDK ATTEMPT {attempt + 1} =====\n{raw}")
                if response.strip():
                    combined = "\n\n".join(all_raw)
                    _persist_claude_artifacts(work_dir, combined, response)
                    return response, combined
            except (ImportError, ModuleNotFoundError) as exc:
                raw = _raw_exception("claude_code_sdk", exc)
                all_raw.append(f"===== CLAUDE SDK ATTEMPT {attempt + 1} =====\n{raw}")
                if mode == "sdk":
                    _persist_claude_artifacts(work_dir, "\n\n".join(all_raw), "")
                    raise
            except Exception as exc:  # noqa: BLE001
                raw = _raw_exception("claude_code_sdk", exc)
                all_raw.append(f"===== CLAUDE SDK ATTEMPT {attempt + 1} =====\n{raw}")
                if mode == "sdk" and attempt >= retries:
                    _persist_claude_artifacts(work_dir, "\n\n".join(all_raw), "")
                    raise
        if mode != "sdk":
            response, raw = _run_claude_code_cli_exec(
                work_dir=work_dir,
                prompt=attempt_prompt,
                model=model,
                timeout=timeout,
                images=images,
                data_dirs=data_dirs,
                allowed_tools=allowed_tools,
                permission_mode=permission_mode,
                allow_file_edits=allow_file_edits,
            )
            all_raw.append(f"===== CLAUDE CLI ATTEMPT {attempt + 1} =====\n{raw}")
            last_response = response
            if response.strip():
                combined = "\n\n".join(all_raw)
                _persist_claude_artifacts(work_dir, combined, response)
                return response, combined

    combined = "\n\n".join(all_raw)
    _persist_claude_artifacts(work_dir, combined, last_response)
    return last_response, combined


def _run_codex_sdk_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    allow_file_edits: bool = False,
) -> tuple[str, str]:
    from openai_codex_sdk import Codex

    for data_dir in data_dirs or []:
        _validate_exec_path(data_dir)
    for image in images or []:
        _validate_exec_path(os.path.dirname(image) or work_dir)

    async def _query() -> tuple[str, str]:
        config = get_codex_exec_config()
        reasoning_effort = str(config.get("reasoning_effort", "") or "").strip()
        thread_options: dict[str, Any] = {
            "working_directory": work_dir,
            "skip_git_repo_check": True,
            "sandbox_mode": str(config.get("sandbox") or "workspace-write"),
            "network_access_enabled": bool(config.get("network_access", False)),
            "web_search_enabled": bool(config.get("web_search", False)),
            "approval_policy": str(config.get("approval_policy") or "never"),
        }
        if model:
            thread_options["model"] = model
        if data_dirs:
            thread_options["additional_directories"] = data_dirs
        if reasoning_effort and reasoning_effort != "none":
            thread_options["model_reasoning_effort"] = reasoning_effort

        codex_options: dict[str, Any] = {"env": os.environ.copy()}
        codex_path = str(config.get("path") or "").strip()
        if codex_path:
            codex_options["codexPathOverride"] = codex_path
        codex = Codex(codex_options)
        thread = codex.start_thread(thread_options)
        turn = await thread.run(
            _exec_prompt(prompt, allow_file_edits=allow_file_edits),
            {"output_schema": _strict_schema(ANSWER_SCHEMA)},
        )
        result_text = str(getattr(turn, "final_response", "") or "")
        parsed: Any = None
        parse_error = ""
        response = ""
        if result_text.strip():
            try:
                parsed = json.loads(result_text)
                response, parse_error = _structured_response(parsed)
            except Exception as exc:  # noqa: BLE001
                parse_error = f"{type(exc).__name__}: {exc}"
        else:
            parse_error = "No response from Codex SDK (final_response is empty)."
        raw = _json_dumps({
            "backend": "codex_sdk",
            "id": getattr(turn, "id", ""),
            "thread_id": getattr(turn, "thread_id", ""),
            "model": model,
            "thread_options": thread_options,
            "final_response": result_text,
            "raw_structured_output": parsed,
            "parse_error": parse_error,
            "is_error": bool(parse_error),
            "items": getattr(turn, "items", []),
        })
        return response, raw

    return _run_async(asyncio.wait_for(_query(), timeout=timeout))


def _run_codex_cli_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    sandbox: str | None = None,
    full_auto: bool | None = None,
    allow_file_edits: bool = False,
) -> tuple[str, str]:
    config = get_codex_exec_config()
    last_message_path = os.path.join(work_dir, "codex_last_message.txt")
    cmd = [
        str(config["path"]),
        "exec",
        "--skip-git-repo-check",
        "--color",
        "never",
        "-C",
        work_dir,
    ]
    if config.get("profile"):
        cmd.extend(["-p", str(config["profile"])])
    reasoning_effort = str(config.get("reasoning_effort", "")).strip()
    if reasoning_effort:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    actual_full_auto = bool(config.get("full_auto", True)) if full_auto is None else bool(full_auto)
    actual_sandbox = str(sandbox or config["sandbox"])
    if actual_full_auto:
        cmd.append("--full-auto")
    else:
        cmd.extend(["--sandbox", actual_sandbox])
    if model:
        cmd.extend(["-m", model])
    for data_dir in data_dirs or []:
        _validate_exec_path(data_dir)
    for image in images or []:
        _validate_exec_path(os.path.dirname(image) or work_dir)
        cmd.extend(["-i", image])
    cmd.extend(["--output-last-message", last_message_path, _exec_prompt(prompt, allow_file_edits=allow_file_edits)])

    try:
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raw = stdout
        if stderr:
            raw = f"{raw}\n[stderr]\n{stderr}" if raw else stderr
        _persist_codex_artifacts(work_dir, raw, "")
        raise
    try:
        from skillopt.model import azure_openai as _openai
        _openai.tracker.record("rollout", 0, 0)
    except Exception:
        pass
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    last_message = ""
    if os.path.exists(last_message_path):
        with open(last_message_path, encoding="utf-8") as f:
            last_message = f.read()
    raw = stdout
    if stderr:
        raw = f"{raw}\n[stderr]\n{stderr}" if raw else stderr
    if proc.returncode != 0:
        _persist_codex_artifacts(work_dir, raw, last_message)
        detail = (stderr or stdout).strip()
        raise RuntimeError(
            f"codex exec failed with exit code {proc.returncode}: {detail[:4000]}"
        )
    return last_message, raw


def run_codex_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    sandbox: str | None = None,
    full_auto: bool | None = None,
    allow_file_edits: bool = False,
) -> tuple[str, str]:
    config = get_codex_exec_config()
    mode = _sdk_mode(config.get("use_sdk"))
    retries = int(config.get("empty_response_retries", 0) or 0)
    last_response = ""
    all_raw: list[str] = []

    for attempt in range(retries + 1):
        attempt_prompt = _retry_prompt(prompt, attempt)
        if mode != "cli":
            try:
                response, raw = _run_codex_sdk_exec(
                    work_dir=work_dir,
                    prompt=attempt_prompt,
                    model=model,
                    timeout=timeout,
                    images=images,
                    data_dirs=data_dirs,
                    allow_file_edits=allow_file_edits,
                )
                all_raw.append(f"===== CODEX SDK ATTEMPT {attempt + 1} =====\n{raw}")
                if response.strip():
                    combined = "\n\n".join(all_raw)
                    _persist_codex_artifacts(work_dir, combined, response)
                    return response, combined
            except (ImportError, ModuleNotFoundError) as exc:
                raw = _raw_exception("codex_sdk", exc)
                all_raw.append(f"===== CODEX SDK ATTEMPT {attempt + 1} =====\n{raw}")
                if mode == "sdk":
                    _persist_codex_artifacts(work_dir, "\n\n".join(all_raw), "")
                    raise
            except Exception as exc:  # noqa: BLE001
                raw = _raw_exception("codex_sdk", exc)
                all_raw.append(f"===== CODEX SDK ATTEMPT {attempt + 1} =====\n{raw}")
                if mode == "sdk" and attempt >= retries:
                    _persist_codex_artifacts(work_dir, "\n\n".join(all_raw), "")
                    raise
        if mode != "sdk":
            response, raw = _run_codex_cli_exec(
                work_dir=work_dir,
                prompt=attempt_prompt,
                model=model,
                timeout=timeout,
                images=images,
                data_dirs=data_dirs,
                sandbox=sandbox,
                full_auto=full_auto,
                allow_file_edits=allow_file_edits,
            )
            all_raw.append(f"===== CODEX CLI ATTEMPT {attempt + 1} =====\n{raw}")
            last_response = response
            if response.strip():
                combined = "\n\n".join(all_raw)
                _persist_codex_artifacts(work_dir, combined, response)
                return response, combined

    combined = "\n\n".join(all_raw)
    _persist_codex_artifacts(work_dir, combined, last_response)
    return last_response, combined


def run_target_exec(
    *,
    work_dir: str,
    prompt: str,
    model: str,
    timeout: int,
    images: list[str] | None = None,
    data_dirs: list[str] | None = None,
    allowed_tools: list[str] | str | None = None,
    permission_mode: str | None = None,
    sandbox: str | None = None,
    full_auto: bool | None = None,
    allow_file_edits: bool = False,
) -> tuple[str, str]:
    backend = get_target_backend()
    if backend == "codex_exec":
        return run_codex_exec(
            work_dir=work_dir,
            prompt=prompt,
            model=model,
            timeout=timeout,
            images=images,
            data_dirs=data_dirs,
            sandbox=sandbox,
            full_auto=full_auto,
            allow_file_edits=allow_file_edits,
        )
    if backend == "claude_code_exec":
        return run_claude_code_exec(
            work_dir=work_dir,
            prompt=prompt,
            model=model,
            timeout=timeout,
            images=images,
            data_dirs=data_dirs,
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
            allow_file_edits=allow_file_edits,
        )
    raise ValueError(f"Unsupported exec backend: {backend}")

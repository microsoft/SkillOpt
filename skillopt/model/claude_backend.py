"""Claude CLI chat backend for ReflACT."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any
from urllib.parse import unquote, urlparse

from skillopt.model.common import CompatAssistantMessage, CompatToolCall, CompatToolFunction, default_model_for_backend, tracker

CLAUDE_BIN = os.environ.get("CLAUDE_CLI_BIN", "claude")
CLAUDE_PERMISSION_MODE = os.environ.get("CLAUDE_PERMISSION_MODE", "dontAsk")
CLAUDE_SETTING_SOURCES = os.environ.get("CLAUDE_SETTING_SOURCES", "user,project")
CLAUDE_ALLOW_ATTACHMENT_READ = os.environ.get("CLAUDE_ALLOW_ATTACHMENT_READ", "1").strip().lower() not in {"0", "false", "no"}

OPTIMIZER_DEPLOYMENT = os.environ.get("OPTIMIZER_DEPLOYMENT", "claude-sonnet-4-6")
TARGET_DEPLOYMENT = os.environ.get("TARGET_DEPLOYMENT", "claude-sonnet-4-6")
REASONING_EFFORT: str | None = None
_VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def _parse_data_uri(url: str) -> tuple[bytes, str]:
    header, data = url.split(",", 1)
    mime = header[5:].split(";", 1)[0] or "image/png"
    return base64.b64decode(data), mime


def _content_to_text(content: Any, attachments: list[dict[str, Any]], *, image_counter: int) -> tuple[str, int]:
    if isinstance(content, str):
        return content, image_counter
    if not isinstance(content, list):
        return str(content), image_counter
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            parts.append(str(item.get("text", "")))
            continue
        if item_type != "image_url":
            continue
        image_counter += 1
        label = f"[Attached image {image_counter}]"
        parts.append(label)
        image_url = item.get("image_url", {}) or {}
        url = str(image_url.get("url", "") or "")
        if not url:
            continue
        if url.startswith("data:") and ";base64," in url:
            data, mime = _parse_data_uri(url)
            attachments.append({"bytes": data, "mime": mime, "label": label})
            continue
        if url.startswith("file://"):
            parsed = urlparse(url)
            path = unquote(parsed.path)
            if path:
                attachments.append({"path": path, "label": label})
            continue
        if os.path.exists(url):
            attachments.append({"path": url, "label": label})
    return "".join(parts), image_counter


def _simplify_tool_schemas(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    simplified: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function", tool)
        simplified.append({
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        })
    return simplified


def _build_prompt_from_messages(messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, tool_choice: str | dict[str, Any] | None = None, structured_output: bool = False) -> tuple[str, str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    history_parts: list[str] = []
    attachments: list[dict[str, Any]] = []
    image_counter = 0

    def _history_line(label: str, body: str) -> str:
        stripped = body.strip()
        if not stripped:
            return f"- {label}:"
        indented = stripped.replace("\n", "\n  ")
        return f"- {label}: {indented}"

    for message in messages:
        role = str(message.get("role", "user"))
        text, image_counter = _content_to_text(message.get("content", ""), attachments, image_counter=image_counter)
        if role == "system":
            if text.strip():
                system_parts.append(text.strip())
            continue
        if role == "assistant":
            block = _history_line("Assistant", text)
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                simplified_calls = []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {}) or {}
                    simplified_calls.append({
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    })
                block += "\n  Compatibility tool requests:\n" + json.dumps(simplified_calls, ensure_ascii=False, indent=2)
            history_parts.append(block)
            continue
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id", "") or "")
            history_parts.append(_history_line(f"Tool result (tool_call_id={tool_call_id})", text))
            continue
        history_parts.append(_history_line(role.capitalize(), text))

    prompt_parts: list[str] = []
    if tools:
        simplified_tools = _simplify_tool_schemas(tools)
        prompt_parts.append("Available compatibility tools:\n" + json.dumps(simplified_tools, ensure_ascii=False, indent=2))
        prompt_parts.append("Do not execute these compatibility tools yourself. If you need one, request it in `tool_calls`. Each `arguments` field must be a JSON string.")
        if tool_choice == "required":
            prompt_parts.append("Tool choice policy: you must request at least one compatibility tool.")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function = tool_choice.get("function", {}) or {}
            prompt_parts.append(f"Tool choice policy: you must request the compatibility tool `{function.get('name', '')}`.")
    history_text = "\n".join(part for part in history_parts if part).strip()
    if history_text:
        prompt_parts.append("History:\n" + history_text)
    if structured_output:
        prompt_parts.append("Return only JSON matching the provided schema.")
        if tools:
            prompt_parts.append("Set `content` to the assistant-visible reply. Set `tool_calls` to an empty array when no compatibility tool is needed.")
    else:
        prompt_parts.append("Answer the latest user request.")
    return "\n\n".join(part for part in system_parts if part).strip(), "\n\n".join(prompt_parts), attachments


def _copy_attachments_to_temp(attachments: list[dict[str, Any]], temp_dir: str) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for index, attachment in enumerate(attachments, 1):
        source_path = attachment.get("path")
        if source_path:
            source_path = str(source_path)
            source_suffix = os.path.splitext(source_path)[1]
            target_path = os.path.join(temp_dir, f"image_{index}{source_suffix or '.bin'}")
            shutil.copyfile(source_path, target_path)
            copied.append({"path": target_path, "label": str(attachment.get("label", ""))})
            continue
        mime = str(attachment.get("mime", "image/png"))
        suffix = mimetypes.guess_extension(mime) or ".png"
        target_path = os.path.join(temp_dir, f"image_{index}{suffix}")
        with open(target_path, "wb") as f:
            f.write(attachment.get("bytes", b"") or b"")
        copied.append({"path": target_path, "label": str(attachment.get("label", ""))})
    return copied


def _append_attachment_instructions(prompt: str, copied_attachments: list[dict[str, str]]) -> str:
    if not copied_attachments or not CLAUDE_ALLOW_ATTACHMENT_READ:
        return prompt
    lines = [
        "Attached image files:",
        *[f"- {item['label'] or f'Attached image {index}'}: {item['path']}" for index, item in enumerate(copied_attachments, 1)],
        "If you need to inspect an attached image, you may use the built-in `Read` tool on those listed paths only. Do not use built-in tools for any other purpose.",
    ]
    return prompt.rstrip() + "\n\n" + "\n".join(lines)


def _usage_from_result(result_event: dict[str, Any] | None) -> dict[str, int]:
    usage = (result_event or {}).get("usage", {}) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _extract_result(event_stream: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    result_event = None
    for event in reversed(event_stream):
        if event.get("type") == "result":
            result_event = event
            break
    if result_event is None:
        raise RuntimeError("Claude backend did not return a result event.")
    content = result_event.get("result") or result_event.get("content") or ""
    return str(content), result_event


def _check_claude_error(stderr_text: str, model: str) -> None:
    lowered = stderr_text.lower()
    if "invalid api key" in lowered or "authentication" in lowered or "login" in lowered:
        raise RuntimeError("Claude CLI is not logged in. Run `claude auth login` (or start `claude` and use `/login`) first.")
    if "unknown model" in lowered or "not available" in lowered or "invalid model" in lowered:
        default_model = default_model_for_backend("claude")
        raise RuntimeError(f"Claude backend tried to use model {model!r}, but your current Claude CLI/account rejected it. Try an available Claude model such as {default_model!r}.")


def _normalize_reasoning_effort(effort: str | None) -> str | None:
    normalized = str(effort or "").strip().lower()
    if not normalized or normalized == "off":
        return None
    if normalized in _VALID_EFFORTS:
        return normalized
    return None


def _assistant_message_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "string"},
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["content", "tool_calls"],
        "additionalProperties": False,
    }


def _assistant_message_schema_wrapper() -> str:
    return json.dumps(_assistant_message_schema(), ensure_ascii=False)


def _run_claude_print(*, system: str, prompt: str, model: str, tools: list[dict[str, Any]] | None, tool_choice: str | dict[str, Any] | None, return_message: bool, timeout: int | None, attachments: list[dict[str, Any]] | None = None) -> tuple[str, dict[str, Any], dict[str, int]]:
    effort = _normalize_reasoning_effort(REASONING_EFFORT)
    with tempfile.TemporaryDirectory(prefix="skillopt_claude_") as temp_dir:
        copied_attachments = _copy_attachments_to_temp(attachments or [], temp_dir)
        prompt_for_cli = _append_attachment_instructions(prompt, copied_attachments)
        cmd = [CLAUDE_BIN, "-p", "--output-format", "json", "--permission-mode", CLAUDE_PERMISSION_MODE, "--add-dir", temp_dir]
        if model:
            cmd.extend(["--model", model])
        if CLAUDE_SETTING_SOURCES:
            cmd.extend(["--setting-sources", CLAUDE_SETTING_SOURCES])
        if system:
            # Write the system prompt to a file, not argv: here the skill being
            # optimized IS the system prompt, and SkillOpt grows it over training,
            # so past ~30 KB it would re-hit the Windows argv cap (WinError 206).
            # The CLI reads it via --append-system-prompt-file.
            system_path = os.path.join(temp_dir, "system_prompt.txt")
            with open(system_path, "w", encoding="utf-8") as system_fh:
                system_fh.write(system)
            cmd.extend(["--append-system-prompt-file", system_path])
        if effort:
            cmd.extend(["--effort", effort])
        structured_output = bool(return_message)
        if structured_output:
            cmd.extend(["--schema", _assistant_message_schema_wrapper()])
        # Feed the prompt via stdin (and the system prompt via a file, above), not
        # argv: on Windows the whole command line is capped at ~32 KB and large
        # optimizer prompts / grown skills overflow it → [WinError 206]. Pin UTF-8
        # so a zh-CN default codepage (cp936) can't raise UnicodeEncodeError on
        # emoji / non-GBK glyphs before the CLI even starts.
        proc = subprocess.run(cmd, input=prompt_for_cli, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout or 300, cwd=temp_dir)
        stderr_text = (proc.stderr or "").strip()
        if proc.returncode != 0:
            _check_claude_error(stderr_text, model)
            raise RuntimeError(stderr_text or f"Claude CLI exited with code {proc.returncode}")
        stream = []
        for raw_line in (proc.stdout or "").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                stream.append(json.loads(raw_line))
            except json.JSONDecodeError:
                continue
        raw_text, result_event = _extract_result(stream)
        usage_info = _usage_from_result(result_event)
        return raw_text, result_event or {}, usage_info


def _compat_message_from_payload(payload: Any) -> CompatAssistantMessage:
    if not isinstance(payload, dict):
        return CompatAssistantMessage(content=str(payload or ""), tool_calls=[])
    content = str(payload.get("content", "") or "")
    tool_calls: list[CompatToolCall] = []
    for index, tool_call in enumerate(payload.get("tool_calls", []) or [], start=1):
        name = str(tool_call.get("name", "") or "")
        arguments = str(tool_call.get("arguments", "{}") or "{}")
        tool_calls.append(CompatToolCall(id=f"claude_tool_{index}", function=CompatToolFunction(name=name, arguments=arguments)))
    return CompatAssistantMessage(content=content, tool_calls=tool_calls)


def _call_messages(messages: list[dict[str, Any]], max_completion_tokens: int, retries: int, stage: str, *, tools: list[dict[str, Any]] | None = None, tool_choice: str | dict[str, Any] | None = None, return_message: bool = False, deployment: str | None = None, timeout: int | None = None) -> tuple[Any, dict[str, int]]:
    del max_completion_tokens
    system, prompt, attachments = _build_prompt_from_messages(messages, tools=tools, tool_choice=tool_choice, structured_output=return_message)
    model = deployment or TARGET_DEPLOYMENT
    last_err = None
    for attempt in range(retries):
        try:
            raw_text, payload, usage_info = _run_claude_print(system=system, prompt=prompt, model=model, tools=tools, tool_choice=tool_choice, return_message=return_message, timeout=timeout, attachments=attachments)
            tracker.record(stage, usage_info["prompt_tokens"], usage_info["completion_tokens"])
            if return_message:
                return _compat_message_from_payload(payload.get("result", payload)), usage_info
            return raw_text, usage_info
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Claude backend failed after {retries} retries: {last_err}")


def chat_optimizer(system: str, user: str, max_completion_tokens: int = 16384, retries: int = 5, stage: str = "optimizer", timeout: int | None = None) -> tuple[str, dict[str, int]]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _call_messages(messages, max_completion_tokens, retries, stage, deployment=OPTIMIZER_DEPLOYMENT, timeout=timeout)


def chat_target(system: str, user: str, max_completion_tokens: int = 16384, retries: int = 5, stage: str = "target", timeout: int | None = None) -> tuple[str, dict[str, int]]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _call_messages(messages, max_completion_tokens, retries, stage, deployment=TARGET_DEPLOYMENT, timeout=timeout)


def chat_with_deployment(deployment: str, system: str, user: str, max_completion_tokens: int = 16384, retries: int = 5, stage: str = "custom", timeout: int | None = None) -> tuple[str, dict[str, int]]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _call_messages(messages, max_completion_tokens, retries, stage, deployment=deployment, timeout=timeout)


def chat_optimizer_messages(messages: list[dict[str, Any]], max_completion_tokens: int = 16384, retries: int = 5, stage: str = "optimizer", *, tools: list[dict[str, Any]] | None = None, tool_choice: str | dict[str, Any] | None = None, return_message: bool = False, timeout: int | None = None) -> tuple[Any, dict[str, int]]:
    return _call_messages(messages, max_completion_tokens, retries, stage, tools=tools, tool_choice=tool_choice, return_message=return_message, deployment=OPTIMIZER_DEPLOYMENT, timeout=timeout)


def chat_target_messages(messages: list[dict[str, Any]], max_completion_tokens: int = 16384, retries: int = 5, stage: str = "target", *, tools: list[dict[str, Any]] | None = None, tool_choice: str | dict[str, Any] | None = None, return_message: bool = False, timeout: int | None = None) -> tuple[Any, dict[str, int]]:
    return _call_messages(messages, max_completion_tokens, retries, stage, tools=tools, tool_choice=tool_choice, return_message=return_message, deployment=TARGET_DEPLOYMENT, timeout=timeout)


def chat_messages_with_deployment(deployment: str, messages: list[dict[str, Any]], max_completion_tokens: int = 16384, retries: int = 5, stage: str = "custom", *, tools: list[dict[str, Any]] | None = None, tool_choice: str | dict[str, Any] | None = None, return_message: bool = False, timeout: int | None = None) -> tuple[Any, dict[str, int]]:
    return _call_messages(messages, max_completion_tokens, retries, stage, tools=tools, tool_choice=tool_choice, return_message=return_message, deployment=deployment, timeout=timeout)


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    global REASONING_EFFORT
    REASONING_EFFORT = effort if effort else None


def set_target_deployment(deployment: str) -> None:
    global TARGET_DEPLOYMENT
    TARGET_DEPLOYMENT = deployment or default_model_for_backend("claude")
    os.environ["TARGET_DEPLOYMENT"] = TARGET_DEPLOYMENT


def set_optimizer_deployment(deployment: str) -> None:
    global OPTIMIZER_DEPLOYMENT
    OPTIMIZER_DEPLOYMENT = deployment or default_model_for_backend("claude")
    os.environ["OPTIMIZER_DEPLOYMENT"] = OPTIMIZER_DEPLOYMENT

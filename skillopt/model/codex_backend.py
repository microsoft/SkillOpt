"""Codex CLI backend for ReflACT."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import tempfile
import time
import uuid
from typing import Any
from urllib.parse import unquote, urlparse

from skillopt.model.common import (
    CompatAssistantMessage,
    CompatToolCall,
    CompatToolFunction,
    tracker,
)


CODEX_BIN = os.environ.get("CODEX_CLI_BIN", "codex")
CODEX_PROFILE = os.environ.get("CODEX_PROFILE", "review")
CODEX_SANDBOX_MODE = os.environ.get("CODEX_SANDBOX_MODE", "read-only")

OPTIMIZER_DEPLOYMENT = os.environ.get("OPTIMIZER_DEPLOYMENT", "gpt-4o")
TARGET_DEPLOYMENT = os.environ.get("TARGET_DEPLOYMENT", "gpt-4o")

REASONING_EFFORT: str | None = None


def _default_working_directory() -> str:
    return os.environ.get("CODEX_WORKING_DIRECTORY", os.getcwd())


def _parse_data_uri(url: str) -> tuple[bytes, str]:
    header, data = url.split(",", 1)
    mime = header[5:].split(";", 1)[0] or "image/png"
    return base64.b64decode(data), mime


def _content_to_text(
    content: Any,
    attachments: list[dict[str, Any]],
    *,
    image_counter: int,
) -> tuple[str, int]:
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
            attachments.append({"bytes": data, "mime": mime})
            continue
        if url.startswith("file://"):
            parsed = urlparse(url)
            path = unquote(parsed.path)
            if path:
                attachments.append({"path": path})
            continue
        if os.path.exists(url):
            attachments.append({"path": url})

    return "".join(parts), image_counter


def _simplify_tool_schemas(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    simplified: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function", tool)
        simplified.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
        )
    return simplified


def _build_prompt_from_messages(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    structured_output: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
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
        text, image_counter = _content_to_text(
            message.get("content", ""),
            attachments,
            image_counter=image_counter,
        )

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
                    simplified_calls.append(
                        {
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", "{}"),
                        }
                    )
                block += (
                    "\n  Compatibility tool requests:\n"
                    + json.dumps(simplified_calls, ensure_ascii=False, indent=2)
                )
            history_parts.append(block)
            continue

        if role == "tool":
            tool_call_id = str(message.get("tool_call_id", "") or "")
            label = f"Tool result (tool_call_id={tool_call_id})"
            history_parts.append(_history_line(label, text))
            continue

        history_parts.append(_history_line(role.capitalize(), text))

    prompt_parts: list[str] = []

    system_text = "\n\n".join(part for part in system_parts if part).strip()
    if system_text:
        prompt_parts.append(system_text)

    if tools:
        simplified_tools = _simplify_tool_schemas(tools)
        prompt_parts.append(
            "Available compatibility tools:\n"
            + json.dumps(simplified_tools, ensure_ascii=False, indent=2)
        )
        prompt_parts.append(
            "Do not execute these tools yourself. If you need one, request it in "
            "`tool_calls`. Each `arguments` field must be a JSON string."
        )

        if tool_choice == "required":
            prompt_parts.append(
                "Tool choice policy: you must request at least one compatibility tool."
            )
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function = tool_choice.get("function", {}) or {}
            prompt_parts.append(
                "Tool choice policy: you must request the compatibility tool "
                f"`{function.get('name', '')}`."
            )

    history_text = "\n".join(part for part in history_parts if part).strip()
    if history_text:
        prompt_parts.append("History:\n" + history_text)

    if structured_output:
        prompt_parts.append("Return only JSON matching the provided schema.")
        if tools:
            prompt_parts.append(
                "Set `content` to the assistant-visible reply. Set `tool_calls` to "
                "an empty array when no tool is needed."
            )
    else:
        prompt_parts.append("Answer the latest user request.")

    return "\n\n".join(prompt_parts), attachments


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


def _materialize_attachments(
    attachments: list[dict[str, Any]],
    temp_dir: str,
) -> list[str]:
    image_paths: list[str] = []
    for index, attachment in enumerate(attachments, 1):
        path = attachment.get("path")
        if path:
            image_paths.append(str(path))
            continue

        mime = str(attachment.get("mime", "image/png"))
        suffix = mimetypes.guess_extension(mime) or ".png"
        image_path = os.path.join(temp_dir, f"image_{index}{suffix}")
        with open(image_path, "wb") as f:
            f.write(attachment.get("bytes", b""))
        image_paths.append(image_path)
    return image_paths


def _usage_from_event(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _extract_error(stdout: str, stderr: str) -> str:
    for raw_line in reversed(stdout.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "turn.failed":
            error = payload.get("error", {}) or {}
            return str(error.get("message", "") or "Codex turn failed")
        if payload.get("type") == "error":
            return str(payload.get("message", "") or "Codex execution failed")
    return stderr.strip() or stdout.strip() or "Codex execution failed"


def _run_codex_exec(
    *,
    model: str,
    prompt: str,
    attachments: list[dict[str, Any]],
    output_schema: dict[str, Any] | None,
    timeout: int | None,
) -> tuple[str, dict[str, int]]:
    with tempfile.TemporaryDirectory(prefix="skillopt_codex_") as temp_dir:
        output_path = os.path.join(temp_dir, "last_message.txt")
        image_paths = _materialize_attachments(attachments, temp_dir)

        command = [
            CODEX_BIN,
            "exec",
            "--json",
            "--ephemeral",
            "--profile",
            CODEX_PROFILE,
            "-c",
            "approval_policy=\"never\"",
            "--sandbox",
            CODEX_SANDBOX_MODE,
            "--skip-git-repo-check",
            "--cd",
            _default_working_directory(),
            "--model",
            model,
            "--output-last-message",
            output_path,
        ]

        if REASONING_EFFORT:
            command.extend(["-c", f"model_reasoning_effort={json.dumps(REASONING_EFFORT)}"])

        schema_path = None
        if output_schema is not None:
            schema_path = os.path.join(temp_dir, "schema.json")
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(output_schema, f, ensure_ascii=False)
            command.extend(["--output-schema", schema_path])

        for image_path in image_paths:
            command.extend(["--image", image_path])

        command.append("-")

        proc = subprocess.run(
            command,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )

        usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        fallback_text = ""
        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "item.completed":
                item = payload.get("item", {}) or {}
                if item.get("type") == "agent_message":
                    fallback_text = str(item.get("text", "") or fallback_text)
            if payload.get("type") == "turn.completed":
                usage_info = _usage_from_event(payload.get("usage"))

        last_message = ""
        if os.path.exists(output_path):
            with open(output_path, encoding="utf-8") as f:
                last_message = f.read().strip()
        if not last_message:
            last_message = fallback_text.strip()

        if proc.returncode != 0:
            raise RuntimeError(_extract_error(proc.stdout, proc.stderr))
        if not last_message:
            raise RuntimeError("Codex returned an empty final message")
        return last_message, usage_info


def _tool_name_from_choice(tool_choice: str | dict[str, Any] | None) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") != "function":
        return None
    function = tool_choice.get("function", {}) or {}
    return str(function.get("name", "") or "") or None


def _compat_message_from_payload(
    payload: dict[str, Any],
    *,
    tool_choice: str | dict[str, Any] | None = None,
) -> CompatAssistantMessage:
    content = str(payload.get("content", "") or "")
    tool_calls: list[CompatToolCall] = []
    for index, raw_tool_call in enumerate(payload.get("tool_calls", []) or [], 1):
        if not isinstance(raw_tool_call, dict):
            continue
        name = str(raw_tool_call.get("name", "") or "")
        arguments = raw_tool_call.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append(
            CompatToolCall(
                id=f"tool_{index}_{uuid.uuid4().hex[:12]}",
                function=CompatToolFunction(name=name, arguments=arguments),
            )
        )

    if tool_choice == "required" and not tool_calls:
        raise RuntimeError("Codex response did not request a tool under tool_choice='required'")

    required_name = _tool_name_from_choice(tool_choice)
    if required_name and all(
        tool_call.function.name != required_name for tool_call in tool_calls
    ):
        raise RuntimeError(
            f"Codex response did not request the required tool {required_name!r}"
        )

    return CompatAssistantMessage(content=content, tool_calls=tool_calls)


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
) -> tuple[Any, dict[str, int]]:
    del max_completion_tokens
    last_err = None
    structured_output = bool(tools) or return_message

    for attempt in range(retries):
        try:
            prompt, attachments = _build_prompt_from_messages(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                structured_output=structured_output,
            )
            raw_text, usage_info = _run_codex_exec(
                model=model,
                prompt=prompt,
                attachments=attachments,
                output_schema=_assistant_message_schema() if structured_output else None,
                timeout=timeout,
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
        except subprocess.TimeoutExpired as exc:
            last_err = RuntimeError(f"Codex CLI timed out after {timeout}s") if timeout else exc
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"Codex call failed after {retries} retries: {last_err}")


def chat_with_model(
    model: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    timeout: int | None = None,
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
    )


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    return chat_with_model(
        model=OPTIMIZER_DEPLOYMENT,
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
    return chat_with_model(
        model=deployment,
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
    return chat_with_model(
        model=TARGET_DEPLOYMENT,
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
    )


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_target_deployment(deployment: str) -> None:
    global TARGET_DEPLOYMENT
    TARGET_DEPLOYMENT = deployment
    os.environ["TARGET_DEPLOYMENT"] = deployment


def set_reasoning_effort(effort: str | None) -> None:
    global REASONING_EFFORT
    REASONING_EFFORT = effort if effort else None


def set_optimizer_deployment(deployment: str) -> None:
    global OPTIMIZER_DEPLOYMENT
    OPTIMIZER_DEPLOYMENT = deployment
    os.environ["OPTIMIZER_DEPLOYMENT"] = deployment

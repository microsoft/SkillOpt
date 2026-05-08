"""ReflACT Model backend — Azure OpenAI wrapper with token tracking.

Provides teacher/student dual-deployment chat functions and a global
TokenTracker for per-stage cost accounting. Previously llm/azure_openai.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any
from openai import AzureOpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

ENDPOINT = os.environ.get(
    "AZURE_OPENAI_ENDPOINT",
    "https://t2vgoaigpt4o3.openai.azure.com/",
)
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
API_KEY = os.environ.get(
    "AZURE_OPENAI_API_KEY",
    "",
)
AUTH_MODE = os.environ.get("AZURE_OPENAI_AUTH_MODE", "azure_cli").strip().lower()
AD_SCOPE = os.environ.get(
    "AZURE_OPENAI_AD_SCOPE",
    "https://cognitiveservices.azure.com/.default",
)
MANAGED_IDENTITY_CLIENT_ID = os.environ.get(
    "AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID",
    "",
).strip()

TEACHER_ENDPOINT = (
    os.environ.get("TEACHER_AZURE_OPENAI_ENDPOINT")
    or os.environ.get("AZURE_OPENAI_TEACHER_ENDPOINT")
    or ENDPOINT
)
STUDENT_ENDPOINT = (
    os.environ.get("STUDENT_AZURE_OPENAI_ENDPOINT")
    or os.environ.get("AZURE_OPENAI_STUDENT_ENDPOINT")
    or ENDPOINT
)
TEACHER_API_VERSION = (
    os.environ.get("TEACHER_AZURE_OPENAI_API_VERSION")
    or os.environ.get("AZURE_OPENAI_TEACHER_API_VERSION")
    or API_VERSION
)
STUDENT_API_VERSION = (
    os.environ.get("STUDENT_AZURE_OPENAI_API_VERSION")
    or os.environ.get("AZURE_OPENAI_STUDENT_API_VERSION")
    or API_VERSION
)
TEACHER_API_KEY = (
    os.environ.get("TEACHER_AZURE_OPENAI_API_KEY")
    or os.environ.get("AZURE_OPENAI_TEACHER_API_KEY")
    or API_KEY
)
STUDENT_API_KEY = (
    os.environ.get("STUDENT_AZURE_OPENAI_API_KEY")
    or os.environ.get("AZURE_OPENAI_STUDENT_API_KEY")
    or API_KEY
)
TEACHER_AUTH_MODE = (
    os.environ.get("TEACHER_AZURE_OPENAI_AUTH_MODE")
    or os.environ.get("AZURE_OPENAI_TEACHER_AUTH_MODE")
    or AUTH_MODE
).strip().lower()
STUDENT_AUTH_MODE = (
    os.environ.get("STUDENT_AZURE_OPENAI_AUTH_MODE")
    or os.environ.get("AZURE_OPENAI_STUDENT_AUTH_MODE")
    or AUTH_MODE
).strip().lower()
TEACHER_AD_SCOPE = (
    os.environ.get("TEACHER_AZURE_OPENAI_AD_SCOPE")
    or os.environ.get("AZURE_OPENAI_TEACHER_AD_SCOPE")
    or AD_SCOPE
)
STUDENT_AD_SCOPE = (
    os.environ.get("STUDENT_AZURE_OPENAI_AD_SCOPE")
    or os.environ.get("AZURE_OPENAI_STUDENT_AD_SCOPE")
    or AD_SCOPE
)
TEACHER_MANAGED_IDENTITY_CLIENT_ID = (
    os.environ.get("TEACHER_AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID")
    or os.environ.get("AZURE_OPENAI_TEACHER_MANAGED_IDENTITY_CLIENT_ID")
    or MANAGED_IDENTITY_CLIENT_ID
).strip()
STUDENT_MANAGED_IDENTITY_CLIENT_ID = (
    os.environ.get("STUDENT_AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID")
    or os.environ.get("AZURE_OPENAI_STUDENT_MANAGED_IDENTITY_CLIENT_ID")
    or MANAGED_IDENTITY_CLIENT_ID
).strip()

TEACHER_DEPLOYMENT = os.environ.get("TEACHER_DEPLOYMENT", "gpt-5.5")
STUDENT_DEPLOYMENT = os.environ.get("STUDENT_DEPLOYMENT", "gpt-5.5")

REASONING_EFFORT: str | None = None

_AZ_CLI_TOKEN_CACHE: dict[str, dict[str, Any]] = {}

# Deployments that require Responses API
_RESPONSES_API_MODELS = {
    "gpt-5.3-codex", "gpt-5.1-codex", "gpt-5.2-codex",
    "gpt-5-codex", "codex-mini", "gpt-5.4-pro",
}


# ── Token Tracker ─────────────────────────────────────────────────────────────

class TokenTracker:
    """Thread-safe per-stage token counter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}

    def record(
        self, stage: str, prompt_tokens: int, completion_tokens: int,
    ) -> None:
        with self._lock:
            if stage not in self._data:
                self._data[stage] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            d = self._data[stage]
            d["calls"] += 1
            d["prompt_tokens"] += prompt_tokens
            d["completion_tokens"] += completion_tokens

    def summary(self) -> dict:
        with self._lock:
            out: dict = {}
            total_p = total_c = total_calls = 0
            for stage, d in sorted(self._data.items()):
                out[stage] = {
                    "calls": d["calls"],
                    "prompt_tokens": d["prompt_tokens"],
                    "completion_tokens": d["completion_tokens"],
                    "total_tokens": d["prompt_tokens"] + d["completion_tokens"],
                }
                total_p += d["prompt_tokens"]
                total_c += d["completion_tokens"]
                total_calls += d["calls"]
            out["_total"] = {
                "calls": total_calls,
                "prompt_tokens": total_p,
                "completion_tokens": total_c,
                "total_tokens": total_p + total_c,
            }
            return out

    def reset(self) -> None:
        with self._lock:
            self._data.clear()

    def stage_snapshot(self, stage: str) -> dict:
        """Return a copy of one stage's counters (or zeros if not tracked yet)."""
        with self._lock:
            d = self._data.get(stage, {})
            return {
                "calls": d.get("calls", 0),
                "prompt_tokens": d.get("prompt_tokens", 0),
                "completion_tokens": d.get("completion_tokens", 0),
                "total_tokens": d.get("prompt_tokens", 0) + d.get("completion_tokens", 0),
            }


tracker = TokenTracker()


# ── Client management ─────────────────────────────────────────────────────────

_teacher_client: AzureOpenAI | None = None
_student_client: AzureOpenAI | None = None
_teacher_lock = threading.Lock()
_student_lock = threading.Lock()


def _role_config(role: str) -> dict[str, str]:
    if role == "teacher":
        return {
            "endpoint": TEACHER_ENDPOINT,
            "api_version": TEACHER_API_VERSION,
            "api_key": TEACHER_API_KEY,
            "auth_mode": TEACHER_AUTH_MODE,
            "ad_scope": TEACHER_AD_SCOPE,
            "managed_identity_client_id": TEACHER_MANAGED_IDENTITY_CLIENT_ID,
        }
    if role == "student":
        return {
            "endpoint": STUDENT_ENDPOINT,
            "api_version": STUDENT_API_VERSION,
            "api_key": STUDENT_API_KEY,
            "auth_mode": STUDENT_AUTH_MODE,
            "ad_scope": STUDENT_AD_SCOPE,
            "managed_identity_client_id": STUDENT_MANAGED_IDENTITY_CLIENT_ID,
        }
    raise ValueError(f"Unknown Azure OpenAI client role: {role!r}")


def _make_token_provider(
    auth_mode: str,
    ad_scope: str,
    managed_identity_client_id: str,
):
    try:
        from azure.identity import (  # type: ignore[import-not-found]
            AzureCliCredential,
            ManagedIdentityCredential,
            get_bearer_token_provider,
        )
    except ImportError as e:
        if auth_mode == "azure_cli":
            return _make_azure_cli_token_provider(ad_scope)
        raise ImportError(
            "Azure AD auth requires azure-identity. Install it with `pip install azure-identity`."
        ) from e

    if auth_mode in {"managed_identity", "aad", "azure_ad"}:
        if managed_identity_client_id:
            credential = ManagedIdentityCredential(client_id=managed_identity_client_id)
        else:
            credential = ManagedIdentityCredential()
    elif auth_mode == "azure_cli":
        credential = AzureCliCredential()
    else:
        raise ValueError(
            "Unsupported Azure OpenAI auth mode "
            f"{auth_mode!r}; expected api_key, managed_identity, azure_ad, aad, or azure_cli."
        )
    return get_bearer_token_provider(credential, ad_scope)


def _make_azure_cli_token_provider(ad_scope: str):
    """Return an Azure CLI token provider compatible with AzureOpenAI.

    This fallback avoids requiring azure-identity in environments where `az`
    is already logged in. The SDK calls this provider whenever it needs a
    bearer token.
    """

    resource = ad_scope.removesuffix("/.default")

    def _provider() -> str:
        now = int(time.time())
        cache = _AZ_CLI_TOKEN_CACHE.setdefault(resource, {"token": "", "expires_on": 0})
        cached = str(cache.get("token") or "")
        expires_on = int(cache.get("expires_on") or 0)
        if cached and expires_on - now > 300:
            return cached

        raw = subprocess.check_output(
            [
                "az",
                "account",
                "get-access-token",
                "--resource",
                resource,
                "-o",
                "json",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        payload = json.loads(raw)
        token = str(payload["accessToken"])
        cache["token"] = token
        cache["expires_on"] = int(payload.get("expires_on") or now + 3000)
        return token

    return _provider


def _make_client(role: str) -> AzureOpenAI:
    cfg = _role_config(role)
    auth_mode = cfg["auth_mode"]
    if auth_mode in {"api_key", "key"}:
        if not cfg["api_key"]:
            raise ValueError(
                f"Azure OpenAI API key is not configured for {role}. "
                "Set model.azure_openai_api_key in the config or export AZURE_OPENAI_API_KEY."
            )
        return AzureOpenAI(
            api_version=cfg["api_version"],
            azure_endpoint=cfg["endpoint"],
            api_key=cfg["api_key"],
        )
    return AzureOpenAI(
        api_version=cfg["api_version"],
        azure_endpoint=cfg["endpoint"],
        azure_ad_token_provider=_make_token_provider(
            auth_mode,
            cfg["ad_scope"],
            cfg["managed_identity_client_id"],
        ),
    )


def get_teacher_client() -> AzureOpenAI:
    global _teacher_client
    with _teacher_lock:
        if _teacher_client is None:
            _teacher_client = _make_client("teacher")
        return _teacher_client


def get_student_client() -> AzureOpenAI:
    global _student_client
    with _student_lock:
        if _student_client is None:
            _student_client = _make_client("student")
        return _student_client


def _needs_responses_api(deployment: str) -> bool:
    dep = deployment.lower()
    return any(dep == m or dep.startswith(m + "-") for m in _RESPONSES_API_MODELS)


# ── Core chat function ────────────────────────────────────────────────────────

def _chat_impl(
    client: AzureOpenAI,
    deployment: str,
    system: str,
    user: str,
    max_completion_tokens: int,
    retries: int,
    stage: str,
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    """Call LLM, track tokens, return (text, usage_dict)."""
    last_err = None
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for attempt in range(retries):
        try:
            if _needs_responses_api(deployment):
                kwargs: dict[str, Any] = {
                    "model": deployment,
                    "instructions": system,
                    "input": [{"role": "user", "content": user}],
                    "max_output_tokens": max_completion_tokens,
                }
                actual_effort = reasoning_effort or REASONING_EFFORT
                if actual_effort:
                    kwargs["reasoning"] = {"effort": actual_effort}
                if timeout is not None:
                    kwargs["timeout"] = timeout
                resp = client.responses.create(**kwargs)
                text = getattr(resp, "output_text", None) or ""
                if not text:
                    for item in getattr(resp, "output", None) or []:
                        for part in getattr(item, "content", []):
                            if getattr(part, "type", "") == "output_text":
                                text = part.text or ""
                if hasattr(resp, "usage") and resp.usage:
                    usage_info = {
                        "prompt_tokens": getattr(resp.usage, "input_tokens", 0) or 0,
                        "completion_tokens": getattr(resp.usage, "output_tokens", 0) or 0,
                        "total_tokens": (
                            (getattr(resp.usage, "input_tokens", 0) or 0)
                            + (getattr(resp.usage, "output_tokens", 0) or 0)
                        ),
                    }
            else:
                kwargs: dict[str, Any] = dict(
                    model=deployment,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_completion_tokens=max_completion_tokens,
                )
                actual_effort = reasoning_effort or REASONING_EFFORT
                if actual_effort is not None:
                    kwargs["reasoning_effort"] = actual_effort
                if timeout is not None:
                    kwargs["timeout"] = timeout
                resp = client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                if resp.usage:
                    usage_info = {
                        "prompt_tokens": resp.usage.prompt_tokens or 0,
                        "completion_tokens": resp.usage.completion_tokens or 0,
                        "total_tokens": resp.usage.total_tokens or 0,
                    }

            tracker.record(
                stage,
                usage_info["prompt_tokens"],
                usage_info["completion_tokens"],
            )
            return text, usage_info

        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep = min(2 ** attempt, 30)
            time.sleep(sleep)

    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


def _chat_messages_impl(
    client: AzureOpenAI,
    deployment: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    retries: int,
    stage: str,
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict]:
    """Call the model with a pre-built message list."""
    last_err = None
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for attempt in range(retries):
        try:
            if _needs_responses_api(deployment):
                input_items, instructions = _messages_to_responses_input(messages)
                kwargs: dict[str, Any] = {
                    "model": deployment,
                    "input": input_items,
                    "max_output_tokens": max_completion_tokens,
                }
                if instructions:
                    kwargs["instructions"] = instructions
                actual_effort = reasoning_effort or REASONING_EFFORT
                if actual_effort:
                    kwargs["reasoning"] = {"effort": actual_effort}
                if tools:
                    kwargs["tools"] = [_chat_tool_to_responses_tool(tool) for tool in tools]
                    if tool_choice is not None:
                        kwargs["tool_choice"] = tool_choice
                if timeout is not None:
                    kwargs["timeout"] = timeout
                resp = client.responses.create(**kwargs)
                message, text = _responses_to_chat_message(resp)
                if hasattr(resp, "usage") and resp.usage:
                    usage_info = {
                        "prompt_tokens": getattr(resp.usage, "input_tokens", 0) or 0,
                        "completion_tokens": getattr(resp.usage, "output_tokens", 0) or 0,
                        "total_tokens": (
                            (getattr(resp.usage, "input_tokens", 0) or 0)
                            + (getattr(resp.usage, "output_tokens", 0) or 0)
                        ),
                    }
            else:
                kwargs = dict(
                    model=deployment,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                )
                actual_effort = reasoning_effort or REASONING_EFFORT
                if actual_effort is not None:
                    kwargs["reasoning_effort"] = actual_effort
                if tools:
                    kwargs["tools"] = tools
                    if tool_choice is not None:
                        kwargs["tool_choice"] = tool_choice
                if timeout is not None:
                    kwargs["timeout"] = timeout
                resp = client.chat.completions.create(**kwargs)
                message = resp.choices[0].message
                text = message.content or ""
                if resp.usage:
                    usage_info = {
                        "prompt_tokens": resp.usage.prompt_tokens or 0,
                        "completion_tokens": resp.usage.completion_tokens or 0,
                        "total_tokens": resp.usage.total_tokens or 0,
                    }
            tracker.record(
                stage,
                usage_info["prompt_tokens"],
                usage_info["completion_tokens"],
            )
            return (message if return_message else text), usage_info
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep = min(2 ** attempt, 30)
            time.sleep(sleep)

    raise RuntimeError(f"LLM message call failed after {retries} retries: {last_err}")


def _chat_tool_to_responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chat Completions function tool to Responses API format."""
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        fn = tool["function"]
        return {
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }
    return tool


def _messages_to_responses_input(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Convert chat-style messages, including tool results, to Responses input."""
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            if content:
                instructions.append(str(content))
            continue
        if role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": str(message.get("tool_call_id", "")),
                "output": str(content),
            })
            continue
        if role == "assistant":
            if content:
                input_items.append({"role": "assistant", "content": str(content)})
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function", {}) or {}
                input_items.append({
                    "type": "function_call",
                    "call_id": str(tool_call.get("id", "")),
                    "name": str(function.get("name", "")),
                    "arguments": str(function.get("arguments", "{}") or "{}"),
                })
            continue
        if role in {"user", "developer"}:
            input_items.append({"role": "user", "content": str(content)})
    return input_items, "\n\n".join(instructions)


def _responses_to_chat_message(resp: Any) -> tuple[Any, str]:
    """Convert Responses output into the subset of Chat message API we use."""
    text = getattr(resp, "output_text", None) or ""
    tool_calls: list[dict[str, Any]] = []
    for item in getattr(resp, "output", None) or []:
        item_type = getattr(item, "type", "")
        if item_type == "function_call":
            tool_calls.append({
                "id": getattr(item, "call_id", "") or getattr(item, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(item, "name", ""),
                    "arguments": getattr(item, "arguments", "") or "{}",
                },
            })
        elif item_type == "message" and not text:
            content_parts = getattr(item, "content", []) or []
            for part in content_parts:
                if getattr(part, "type", "") == "output_text":
                    text += getattr(part, "text", "") or ""
    return SimpleNamespace(content=text, tool_calls=tool_calls), text


# ── Public API ────────────────────────────────────────────────────────────────

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
    global ENDPOINT, API_VERSION, API_KEY, AUTH_MODE, AD_SCOPE, MANAGED_IDENTITY_CLIENT_ID
    global TEACHER_ENDPOINT, TEACHER_API_VERSION, TEACHER_API_KEY, TEACHER_AUTH_MODE
    global TEACHER_AD_SCOPE, TEACHER_MANAGED_IDENTITY_CLIENT_ID
    global STUDENT_ENDPOINT, STUDENT_API_VERSION, STUDENT_API_KEY, STUDENT_AUTH_MODE
    global STUDENT_AD_SCOPE, STUDENT_MANAGED_IDENTITY_CLIENT_ID
    global _teacher_client, _student_client

    def _clean(value: str | None, *, lower: bool = False) -> str | None:
        if value is None:
            return None
        str_value = str(value).strip()
        if not str_value:
            return None
        if lower:
            str_value = str_value.lower()
        return str_value

    def _set(global_name: str, value: str | None, env_key: str) -> None:
        if value is None:
            return
        globals()[global_name] = value
        os.environ[env_key] = value

    shared_endpoint = _clean(endpoint)
    shared_api_version = _clean(api_version)
    shared_api_key = _clean(api_key)
    shared_auth_mode = _clean(auth_mode, lower=True)
    shared_ad_scope = _clean(ad_scope)
    shared_managed_identity_client_id = _clean(managed_identity_client_id)

    _set("ENDPOINT", shared_endpoint, "AZURE_OPENAI_ENDPOINT")
    _set("API_VERSION", shared_api_version, "AZURE_OPENAI_API_VERSION")
    _set("API_KEY", shared_api_key, "AZURE_OPENAI_API_KEY")
    _set("AUTH_MODE", shared_auth_mode, "AZURE_OPENAI_AUTH_MODE")
    _set("AD_SCOPE", shared_ad_scope, "AZURE_OPENAI_AD_SCOPE")
    _set(
        "MANAGED_IDENTITY_CLIENT_ID",
        shared_managed_identity_client_id,
        "AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID",
    )

    resolved_teacher_endpoint = _clean(teacher_endpoint) or shared_endpoint
    resolved_teacher_api_version = _clean(teacher_api_version) or shared_api_version
    resolved_teacher_api_key = _clean(teacher_api_key) or shared_api_key
    resolved_teacher_auth_mode = _clean(teacher_auth_mode, lower=True) or shared_auth_mode
    resolved_teacher_ad_scope = _clean(teacher_ad_scope) or shared_ad_scope
    resolved_teacher_mi = (
        _clean(teacher_managed_identity_client_id)
        or shared_managed_identity_client_id
    )
    resolved_student_endpoint = _clean(student_endpoint) or shared_endpoint
    resolved_student_api_version = _clean(student_api_version) or shared_api_version
    resolved_student_api_key = _clean(student_api_key) or shared_api_key
    resolved_student_auth_mode = _clean(student_auth_mode, lower=True) or shared_auth_mode
    resolved_student_ad_scope = _clean(student_ad_scope) or shared_ad_scope
    resolved_student_mi = (
        _clean(student_managed_identity_client_id)
        or shared_managed_identity_client_id
    )

    _set("TEACHER_ENDPOINT", resolved_teacher_endpoint, "TEACHER_AZURE_OPENAI_ENDPOINT")
    _set(
        "TEACHER_API_VERSION",
        resolved_teacher_api_version,
        "TEACHER_AZURE_OPENAI_API_VERSION",
    )
    _set("TEACHER_API_KEY", resolved_teacher_api_key, "TEACHER_AZURE_OPENAI_API_KEY")
    _set("TEACHER_AUTH_MODE", resolved_teacher_auth_mode, "TEACHER_AZURE_OPENAI_AUTH_MODE")
    _set("TEACHER_AD_SCOPE", resolved_teacher_ad_scope, "TEACHER_AZURE_OPENAI_AD_SCOPE")
    _set(
        "TEACHER_MANAGED_IDENTITY_CLIENT_ID",
        resolved_teacher_mi,
        "TEACHER_AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID",
    )
    _set("STUDENT_ENDPOINT", resolved_student_endpoint, "STUDENT_AZURE_OPENAI_ENDPOINT")
    _set(
        "STUDENT_API_VERSION",
        resolved_student_api_version,
        "STUDENT_AZURE_OPENAI_API_VERSION",
    )
    _set("STUDENT_API_KEY", resolved_student_api_key, "STUDENT_AZURE_OPENAI_API_KEY")
    _set("STUDENT_AUTH_MODE", resolved_student_auth_mode, "STUDENT_AZURE_OPENAI_AUTH_MODE")
    _set("STUDENT_AD_SCOPE", resolved_student_ad_scope, "STUDENT_AZURE_OPENAI_AD_SCOPE")
    _set(
        "STUDENT_MANAGED_IDENTITY_CLIENT_ID",
        resolved_student_mi,
        "STUDENT_AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID",
    )

    with _teacher_lock:
        _teacher_client = None
    with _student_lock:
        _student_client = None


def chat_teacher(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "teacher",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    """Call the teacher model.  Returns (response_text, usage_dict)."""
    return _chat_impl(
        get_teacher_client(), TEACHER_DEPLOYMENT,
        system, user, max_completion_tokens, retries, stage, reasoning_effort, timeout,
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
    """Call an arbitrary deployment using the shared Azure client."""
    return _chat_impl(
        get_teacher_client(),
        deployment,
        system,
        user,
        max_completion_tokens,
        retries,
        stage,
        reasoning_effort,
        timeout,
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
    """Call the student model.  Returns (response_text, usage_dict)."""
    return _chat_impl(
        get_student_client(), STUDENT_DEPLOYMENT,
        system, user, max_completion_tokens, retries, stage, reasoning_effort, timeout,
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
    """Call the teacher model with a pre-built chat message list."""
    return _chat_messages_impl(
        get_teacher_client(),
        TEACHER_DEPLOYMENT,
        messages,
        max_completion_tokens,
        retries,
        stage,
        reasoning_effort,
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
    """Call an arbitrary deployment with a pre-built chat message list."""
    return _chat_messages_impl(
        get_teacher_client(),
        deployment,
        messages,
        max_completion_tokens,
        retries,
        stage,
        reasoning_effort,
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
    """Call the student model with a pre-built chat message list."""
    return _chat_messages_impl(
        get_student_client(),
        STUDENT_DEPLOYMENT,
        messages,
        max_completion_tokens,
        retries,
        stage,
        reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def get_token_summary() -> dict:
    """Return per-stage and total token usage."""
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_student_deployment(deployment: str) -> None:
    """Change student deployment at runtime."""
    global _student_client, STUDENT_DEPLOYMENT
    STUDENT_DEPLOYMENT = deployment
    os.environ["STUDENT_DEPLOYMENT"] = deployment
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = deployment
    with _student_lock:
        _student_client = None
    try:
        import llm_client as _legacy
        _legacy.DEPLOYMENT = deployment
        _legacy._client = None
    except Exception:
        pass


def set_reasoning_effort(effort: str | None) -> None:
    """Set reasoning effort for all LLM calls. None = off."""
    global REASONING_EFFORT
    REASONING_EFFORT = effort if effort else None


def get_reasoning_effort() -> str | None:
    """Return the process-wide reasoning effort for direct Azure client users."""
    return REASONING_EFFORT


def set_teacher_deployment(deployment: str) -> None:
    """Change teacher deployment at runtime."""
    global _teacher_client, TEACHER_DEPLOYMENT
    TEACHER_DEPLOYMENT = deployment
    os.environ["TEACHER_DEPLOYMENT"] = deployment
    with _teacher_lock:
        _teacher_client = None

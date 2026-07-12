"""Tests for the OpenAI-compatible Qwen chat backend."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from collections.abc import Iterator
from dataclasses import fields
from typing import Any

import pytest

from skillopt.envs.searchqa.evaluator import extract_answer

_QWEN_CONFIG_ENV_KEYS = (
    "BASE_URL",
    "API_KEY",
    "TEMPERATURE",
    "TIMEOUT_SECONDS",
    "MAX_TOKENS",
    "ENABLE_THINKING",
    "USE_MAX_COMPLETION_TOKENS",
)
_ENV_KEYS = ("OPTIMIZER_BACKEND", "TARGET_BACKEND") + tuple(
    f"{prefix}QWEN_CHAT_{key}" for prefix in ("", "OPTIMIZER_", "TARGET_") for key in _QWEN_CONFIG_ENV_KEYS
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _UrlopenRecorder:
    def __init__(self, content: str = "<answer>yes</answer>") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def __call__(self, request: Any, timeout: float | None = None) -> _FakeResponse:
        request_data = request.data.decode("utf-8")
        self.calls.append(
            {
                "payload": json.loads(request_data),
                "timeout": timeout,
            }
        )
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": self.content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            }
        )


class _OpenAIClientStub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def _install_openai_stub() -> None:
    if "openai" in sys.modules or importlib.util.find_spec("openai") is not None:
        return
    openai_stub = types.ModuleType("openai")
    openai_stub.AzureOpenAI = _OpenAIClientStub
    openai_stub.OpenAI = _OpenAIClientStub
    sys.modules["openai"] = openai_stub


def _import_model_modules() -> tuple[Any, Any, Any]:
    _install_openai_stub()
    import skillopt.model as model_module
    from skillopt.model import backend_config, qwen_backend

    return model_module, backend_config, qwen_backend


def _snapshot_config(config: Any) -> dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def _restore_config(config: Any, snapshot: dict[str, Any]) -> None:
    for key, value in snapshot.items():
        setattr(config, key, value)


@pytest.fixture(autouse=True)
def isolate_qwen_state() -> Iterator[tuple[Any, Any]]:
    model_module, backend_config, qwen_backend = _import_model_modules()
    optimizer_config = _snapshot_config(qwen_backend.OPTIMIZER_CONFIG)
    target_config = _snapshot_config(qwen_backend.TARGET_CONFIG)
    optimizer_backend = backend_config.get_optimizer_backend()
    target_backend = backend_config.get_target_backend()
    env = {key: os.environ.get(key) for key in _ENV_KEYS}
    qwen_backend.reset_token_tracker()
    yield model_module, qwen_backend
    qwen_backend.reset_token_tracker()
    _restore_config(qwen_backend.OPTIMIZER_CONFIG, optimizer_config)
    _restore_config(qwen_backend.TARGET_CONFIG, target_config)
    backend_config.set_optimizer_backend(optimizer_backend)
    backend_config.set_target_backend(target_backend)
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _use_qwen_target(model_module: Any, qwen_backend: Any, enable_thinking: bool) -> None:
    model_module.set_target_backend("qwen_chat")
    qwen_backend.TARGET_CONFIG.base_url = "http://qwen.example/v1"
    qwen_backend.TARGET_CONFIG.api_key = ""
    qwen_backend.TARGET_CONFIG.timeout_seconds = 300.0
    qwen_backend.TARGET_CONFIG.max_tokens = 8000
    qwen_backend.TARGET_CONFIG.temperature = None
    qwen_backend.TARGET_CONFIG.enable_thinking = enable_thinking
    qwen_backend.TARGET_CONFIG.deployment = "qwen-test"


def _record_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    qwen_backend: Any,
    content: str = "<answer>yes</answer>",
) -> _UrlopenRecorder:
    recorder = _UrlopenRecorder(content)
    monkeypatch.setattr(qwen_backend.urllib.request, "urlopen", recorder)
    return recorder


def test_chat_target_omits_chat_template_kwargs_when_thinking_disabled(
    monkeypatch: pytest.MonkeyPatch,
    isolate_qwen_state: tuple[Any, Any],
) -> None:
    model_module, qwen_backend = isolate_qwen_state
    _use_qwen_target(model_module, qwen_backend, enable_thinking=False)
    recorder = _record_urlopen(monkeypatch, qwen_backend)

    text, usage = model_module.chat_target(
        "system",
        "user",
        max_completion_tokens=128,
        retries=1,
        timeout=10.0,
    )

    assert text == "<answer>yes</answer>"
    assert usage["total_tokens"] == 3
    assert "chat_template_kwargs" not in recorder.calls[0]["payload"]
    assert recorder.calls[0]["timeout"] == 10.0


def test_chat_target_includes_chat_template_kwargs_when_thinking_enabled(
    monkeypatch: pytest.MonkeyPatch,
    isolate_qwen_state: tuple[Any, Any],
) -> None:
    model_module, qwen_backend = isolate_qwen_state
    _use_qwen_target(model_module, qwen_backend, enable_thinking=True)
    content = "<think>working</think>\n<answer>yes</answer>"
    recorder = _record_urlopen(monkeypatch, qwen_backend, content=content)

    text, _ = model_module.chat_target(
        "system",
        "user",
        max_completion_tokens=128,
        retries=1,
    )

    assert recorder.calls[0]["payload"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert extract_answer(text) == "yes"


def test_chat_target_messages_forwards_timeout_to_qwen_backend(
    monkeypatch: pytest.MonkeyPatch,
    isolate_qwen_state: tuple[Any, Any],
) -> None:
    model_module, qwen_backend = isolate_qwen_state
    _use_qwen_target(model_module, qwen_backend, enable_thinking=False)
    recorder = _record_urlopen(monkeypatch, qwen_backend)

    text, _ = model_module.chat_target_messages(
        [{"role": "user", "content": "question"}],
        max_completion_tokens=128,
        retries=1,
        timeout=10.0,
    )

    assert text == "<answer>yes</answer>"
    assert recorder.calls[0]["timeout"] == 10.0


def test_configure_qwen_chat_runtime_toggle_controls_payload(
    monkeypatch: pytest.MonkeyPatch,
    isolate_qwen_state: tuple[Any, Any],
) -> None:
    model_module, qwen_backend = isolate_qwen_state
    _use_qwen_target(model_module, qwen_backend, enable_thinking=False)
    recorder = _record_urlopen(monkeypatch, qwen_backend)

    model_module.configure_qwen_chat(enable_thinking=True)
    model_module.chat_target("system", "user", max_completion_tokens=128, retries=1)
    model_module.configure_qwen_chat(enable_thinking=False)
    model_module.chat_target("system", "user", max_completion_tokens=128, retries=1)

    assert recorder.calls[0]["payload"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "chat_template_kwargs" not in recorder.calls[1]["payload"]


def test_chat_target_uses_max_tokens_by_default(
    monkeypatch: pytest.MonkeyPatch,
    isolate_qwen_state: tuple[Any, Any],
) -> None:
    model_module, qwen_backend = isolate_qwen_state
    _use_qwen_target(model_module, qwen_backend, enable_thinking=False)
    recorder = _record_urlopen(monkeypatch, qwen_backend)

    model_module.chat_target("system", "user", max_completion_tokens=128, retries=1)

    payload = recorder.calls[0]["payload"]
    assert payload["max_tokens"] == 128
    assert "max_completion_tokens" not in payload


def test_chat_target_uses_max_completion_tokens_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    isolate_qwen_state: tuple[Any, Any],
) -> None:
    model_module, qwen_backend = isolate_qwen_state
    _use_qwen_target(model_module, qwen_backend, enable_thinking=False)
    qwen_backend.TARGET_CONFIG.use_max_completion_tokens = True
    recorder = _record_urlopen(monkeypatch, qwen_backend)

    model_module.chat_target("system", "user", max_completion_tokens=128, retries=1)

    payload = recorder.calls[0]["payload"]
    assert payload["max_completion_tokens"] == 128
    assert "max_tokens" not in payload


def test_configure_qwen_chat_toggles_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
    isolate_qwen_state: tuple[Any, Any],
) -> None:
    model_module, qwen_backend = isolate_qwen_state
    _use_qwen_target(model_module, qwen_backend, enable_thinking=False)
    recorder = _record_urlopen(monkeypatch, qwen_backend)

    model_module.configure_qwen_chat(target_use_max_completion_tokens=True)
    model_module.chat_target("system", "user", max_completion_tokens=128, retries=1)
    model_module.configure_qwen_chat(target_use_max_completion_tokens=False)
    model_module.chat_target("system", "user", max_completion_tokens=128, retries=1)

    assert "max_completion_tokens" in recorder.calls[0]["payload"]
    assert "max_tokens" in recorder.calls[1]["payload"]


def test_temperature_omitted_when_env_is_blank(
    monkeypatch: pytest.MonkeyPatch,
    isolate_qwen_state: tuple[Any, Any],
) -> None:
    _model_module, qwen_backend = isolate_qwen_state
    # Explicit blank means "omit", not "fall back to 0.7".
    monkeypatch.setenv("TARGET_QWEN_CHAT_TEMPERATURE", "")
    assert qwen_backend._resolve_temperature("target") is None
    monkeypatch.setenv("TARGET_QWEN_CHAT_TEMPERATURE", "off")
    assert qwen_backend._resolve_temperature("target") is None


def test_temperature_resolves_float_and_default(
    monkeypatch: pytest.MonkeyPatch,
    isolate_qwen_state: tuple[Any, Any],
) -> None:
    _model_module, qwen_backend = isolate_qwen_state
    monkeypatch.delenv("QWEN_CHAT_TEMPERATURE", raising=False)
    monkeypatch.delenv("TARGET_QWEN_CHAT_TEMPERATURE", raising=False)
    assert qwen_backend._resolve_temperature("target") == 0.7
    monkeypatch.setenv("TARGET_QWEN_CHAT_TEMPERATURE", "0.2")
    assert qwen_backend._resolve_temperature("target") == 0.2

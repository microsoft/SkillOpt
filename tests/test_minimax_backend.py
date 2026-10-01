"""Tests for the OpenAI-compatible MiniMax chat backend."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _UrlopenRecorder:
    def __init__(self, content: str = "answer") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def __call__(self, request: Any, timeout: float | None = None) -> _FakeResponse:
        self.calls.append(
            {
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return _FakeResponse(
            {
                "choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
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


@pytest.fixture()
def minimax_backend() -> Iterator[Any]:
    _install_openai_stub()
    from skillopt.model import minimax_backend as backend

    snapshot = {
        "ENABLE_THINKING": backend.ENABLE_THINKING,
        "TARGET_DEPLOYMENT": backend.TARGET_DEPLOYMENT,
        "OPTIMIZER_DEPLOYMENT": backend.OPTIMIZER_DEPLOYMENT,
        "API_KEY": backend.API_KEY,
        "BASE_URL": backend.BASE_URL,
    }
    backend.reset_token_tracker()
    yield backend
    backend.reset_token_tracker()
    for key, value in snapshot.items():
        setattr(backend, key, value)


def _record_urlopen(monkeypatch: pytest.MonkeyPatch, backend: Any) -> _UrlopenRecorder:
    recorder = _UrlopenRecorder()
    monkeypatch.setattr(backend.urllib.request, "urlopen", recorder)
    return recorder


def test_default_deployment_is_current_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillopt.model.common import default_model_for_backend

    _install_openai_stub()
    monkeypatch.delenv("TARGET_DEPLOYMENT", raising=False)
    monkeypatch.delenv("OPTIMIZER_DEPLOYMENT", raising=False)

    from skillopt.model import minimax_backend as backend
    module = importlib.reload(backend)

    assert default_model_for_backend("minimax_chat") == "MiniMax-M3"
    assert module.TARGET_DEPLOYMENT == "MiniMax-M3"
    assert module.OPTIMIZER_DEPLOYMENT == "MiniMax-M3"


def test_always_on_model_sends_adaptive_not_disabled(monkeypatch: pytest.MonkeyPatch, minimax_backend: Any) -> None:
    """M2.x cannot turn thinking off, so never claim it is disabled.

    MiniMax documents that the M2 family accepts ``{"type": "disabled"}`` but
    keeps thinking on anyway. Sending "disabled" would record a request that
    does not match what the model actually does.
    """
    minimax_backend.ENABLE_THINKING = False
    minimax_backend.TARGET_DEPLOYMENT = "MiniMax-M2.7"
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("system", "user", retries=1)

    payload = recorder.calls[0]["payload"]
    assert payload["model"] == "MiniMax-M2.7"
    assert payload["thinking"] == {"type": "adaptive"}


def test_adaptive_model_respects_disabled_flag(monkeypatch: pytest.MonkeyPatch, minimax_backend: Any) -> None:
    minimax_backend.ENABLE_THINKING = False
    minimax_backend.TARGET_DEPLOYMENT = "MiniMax-M3"
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("system", "user", retries=1)

    payload = recorder.calls[0]["payload"]
    assert payload["model"] == "MiniMax-M3"
    assert payload["thinking"] == {"type": "disabled"}


def test_adaptive_model_respects_enabled_flag(monkeypatch: pytest.MonkeyPatch, minimax_backend: Any) -> None:
    minimax_backend.ENABLE_THINKING = True
    minimax_backend.TARGET_DEPLOYMENT = "MiniMax-M3"
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("system", "user", retries=1)

    assert recorder.calls[0]["payload"]["thinking"] == {"type": "adaptive"}


def test_unsupported_chat_template_kwargs_is_never_sent(monkeypatch: pytest.MonkeyPatch, minimax_backend: Any) -> None:
    """Guards the original regression.

    ``chat_template_kwargs.enable_thinking`` is a Qwen/HuggingFace-serving
    convention. It appears nowhere in MiniMax's OpenAI-compatible reference, so
    the endpoint ignores it -- meaning thinking silently stayed at the server
    default no matter what the flag said.
    """
    minimax_backend.ENABLE_THINKING = False
    minimax_backend.TARGET_DEPLOYMENT = "MiniMax-M3"
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("system", "user", retries=1)

    assert "chat_template_kwargs" not in recorder.calls[0]["payload"]


def test_unknown_deployment_defaults_to_adaptive(monkeypatch: pytest.MonkeyPatch, minimax_backend: Any) -> None:
    """An unrecognized model follows the documented API default (thinking on)."""
    minimax_backend.ENABLE_THINKING = True
    minimax_backend.TARGET_DEPLOYMENT = "MiniMax-Future-9"
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("system", "user", retries=1)

    assert recorder.calls[0]["payload"]["thinking"] == {"type": "adaptive"}


def test_chat_optimizer_and_target_use_respective_deployments(
    monkeypatch: pytest.MonkeyPatch, minimax_backend: Any
) -> None:
    minimax_backend.TARGET_DEPLOYMENT = "MiniMax-Target-Model"
    minimax_backend.OPTIMIZER_DEPLOYMENT = "MiniMax-Optimizer-Model"
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("sys_target", "user_target", retries=1)
    minimax_backend.chat_optimizer("sys_opt", "user_opt", retries=1)
    minimax_backend.chat_target_messages([{"role": "user", "content": "msg_target"}], retries=1)
    minimax_backend.chat_optimizer_messages([{"role": "user", "content": "msg_opt"}], retries=1)

    assert recorder.calls[0]["payload"]["model"] == "MiniMax-Target-Model"
    assert recorder.calls[1]["payload"]["model"] == "MiniMax-Optimizer-Model"
    assert recorder.calls[2]["payload"]["model"] == "MiniMax-Target-Model"
    assert recorder.calls[3]["payload"]["model"] == "MiniMax-Optimizer-Model"


def test_set_optimizer_and_target_deployment(minimax_backend: Any) -> None:
    minimax_backend.set_target_deployment("MiniMax-New-Target")
    assert minimax_backend.TARGET_DEPLOYMENT == "MiniMax-New-Target"
    assert os.environ.get("TARGET_DEPLOYMENT") == "MiniMax-New-Target"

    minimax_backend.set_optimizer_deployment("MiniMax-New-Optimizer")
    assert minimax_backend.OPTIMIZER_DEPLOYMENT == "MiniMax-New-Optimizer"
    assert os.environ.get("OPTIMIZER_DEPLOYMENT") == "MiniMax-New-Optimizer"


def test_timeout_forwarded_to_urlopen(monkeypatch: pytest.MonkeyPatch, minimax_backend: Any) -> None:
    recorder = _record_urlopen(monkeypatch, minimax_backend)

    minimax_backend.chat_target("system", "user", retries=1, timeout=42.5)
    minimax_backend.chat_optimizer("system", "user", retries=1, timeout=55.0)
    minimax_backend.chat_target_messages([{"role": "user", "content": "hi"}], retries=1, timeout=60.0)
    minimax_backend.chat_optimizer_messages([{"role": "user", "content": "hi"}], retries=1, timeout=75.0)

    assert recorder.calls[0]["timeout"] == 42.5
    assert recorder.calls[1]["timeout"] == 55.0
    assert recorder.calls[2]["timeout"] == 60.0
    assert recorder.calls[3]["timeout"] == 75.0


def test_fresh_import_optimizer_calls_without_setter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: calling chat_optimizer or chat_optimizer_messages without calling

    set_optimizer_deployment() on a fresh import must not raise NameError for
    OPTIMIZER_DEPLOYMENT.
    """
    _install_openai_stub()
    monkeypatch.delenv("OPTIMIZER_DEPLOYMENT", raising=False)
    monkeypatch.delenv("TARGET_DEPLOYMENT", raising=False)

    from skillopt.model import minimax_backend as backend

    module = importlib.reload(backend)
    recorder = _record_urlopen(monkeypatch, module)

    text, usage = module.chat_optimizer("system prompt", "user query", retries=1)
    assert text == "answer"
    assert recorder.calls[0]["payload"]["model"] == "MiniMax-M3"

    msg, usage_msg = module.chat_optimizer_messages([{"role": "user", "content": "user query"}], retries=1)
    assert msg == "answer"
    assert recorder.calls[1]["payload"]["model"] == "MiniMax-M3"


def test_fresh_import_respects_optimizer_deployment_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_openai_stub()
    monkeypatch.setenv("OPTIMIZER_DEPLOYMENT", "MiniMax-Env-Optimizer")
    monkeypatch.setenv("TARGET_DEPLOYMENT", "MiniMax-Env-Target")

    from skillopt.model import minimax_backend as backend

    module = importlib.reload(backend)
    recorder = _record_urlopen(monkeypatch, module)

    module.chat_optimizer("system prompt", "user query", retries=1)
    module.chat_optimizer_messages([{"role": "user", "content": "user query"}], retries=1)
    module.chat_target("system prompt", "user query", retries=1)
    module.chat_target_messages([{"role": "user", "content": "user query"}], retries=1)

    assert recorder.calls[0]["payload"]["model"] == "MiniMax-Env-Optimizer"
    assert recorder.calls[1]["payload"]["model"] == "MiniMax-Env-Optimizer"
    assert recorder.calls[2]["payload"]["model"] == "MiniMax-Env-Target"
    assert recorder.calls[3]["payload"]["model"] == "MiniMax-Env-Target"


def test_model_dispatcher_chat_optimizer_messages_minimax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_openai_stub()
    import skillopt.model as model
    from skillopt.model import backend_config
    from skillopt.model import minimax_backend as backend

    module = importlib.reload(backend)
    recorder = _record_urlopen(monkeypatch, module)

    backend_config.set_optimizer_backend("minimax_chat")
    model.set_optimizer_deployment("MiniMax-Custom-Opt")

    res, _ = model.chat_optimizer("system", "user", retries=1, timeout=99)
    assert res == "answer"
    assert recorder.calls[0]["payload"]["model"] == "MiniMax-Custom-Opt"
    assert recorder.calls[0]["timeout"] == 99

    res_msg, _ = model.chat_optimizer_messages([{"role": "user", "content": "test"}], retries=1, timeout=88)
    assert res_msg == "answer"
    assert recorder.calls[1]["payload"]["model"] == "MiniMax-Custom-Opt"
    assert recorder.calls[1]["timeout"] == 88

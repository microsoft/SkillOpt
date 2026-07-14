"""Tests for the generic OpenAI-compatible model backend."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import skillopt.model as model
from skillopt.model import backend_config
from skillopt.model import openai_compatible_backend as backend


class _CompletionRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=[])
        usage = SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _Client:
    def __init__(self, recorder: _CompletionRecorder) -> None:
        self.chat = SimpleNamespace(completions=recorder)


@pytest.fixture(autouse=True)
def isolate_backend_state(monkeypatch: pytest.MonkeyPatch):
    optimizer_backend = backend_config.get_optimizer_backend()
    target_backend = backend_config.get_target_backend()
    optimizer_config = vars(backend.OPTIMIZER_CONFIG).copy()
    target_config = vars(backend.TARGET_CONFIG).copy()
    backend.reset_token_tracker()
    yield
    backend.reset_token_tracker()
    vars(backend.OPTIMIZER_CONFIG).update(optimizer_config)
    vars(backend.TARGET_CONFIG).update(target_config)
    backend_config.set_optimizer_backend(optimizer_backend)
    backend_config.set_target_backend(target_backend)
    backend._reset_clients()


def test_configure_preserves_role_specific_values() -> None:
    model.configure_openai_compatible(
        base_url="https://shared.example/v1",
        api_key="shared-key",
        model="shared-model",
        optimizer_base_url="https://optimizer.example/v1",
        optimizer_api_key="optimizer-key",
        optimizer_model="optimizer-model",
        target_base_url="https://target.example/v1",
        target_api_key="target-key",
        target_model="target-model",
    )

    assert backend.OPTIMIZER_CONFIG.base_url == "https://optimizer.example/v1"
    assert backend.OPTIMIZER_CONFIG.api_key == "optimizer-key"
    assert backend.OPTIMIZER_CONFIG.deployment == "optimizer-model"
    assert backend.TARGET_CONFIG.base_url == "https://target.example/v1"
    assert backend.TARGET_CONFIG.api_key == "target-key"
    assert backend.TARGET_CONFIG.deployment == "target-model"


def test_optimizer_and_target_route_to_their_own_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer_calls = _CompletionRecorder()
    target_calls = _CompletionRecorder()
    monkeypatch.setattr(
        backend,
        "_get_client",
        lambda role: _Client(optimizer_calls if role == "optimizer" else target_calls),
    )
    model.set_optimizer_backend("openai_compatible")
    model.set_target_backend("openai_compatible")
    backend.OPTIMIZER_CONFIG.deployment = "optimizer-model"
    backend.TARGET_CONFIG.deployment = "target-model"

    model.chat_optimizer("system", "user", retries=1)
    model.chat_target_messages([{"role": "user", "content": "question"}], retries=1)

    assert optimizer_calls.calls[0]["model"] == "optimizer-model"
    assert target_calls.calls[0]["model"] == "target-model"


def test_client_creation_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[str] = []

    def build(config: backend.OpenAICompatibleConfig) -> _Client:
        builds.append(config.deployment)
        return _Client(_CompletionRecorder())

    backend._reset_clients()
    monkeypatch.setattr(backend, "_build_client", build)
    assert builds == []

    backend._get_client("optimizer")
    backend._get_client("optimizer")

    assert builds == [backend.OPTIMIZER_CONFIG.deployment]


def test_combined_token_summary_counts_each_backend_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def make(stage: str) -> dict[str, dict[str, int]]:
        return {
            stage: {
                "calls": 1,
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
            "_total": {
                "calls": 1,
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        }

    monkeypatch.setattr(model._openai, "get_token_summary", lambda: make("azure"))
    monkeypatch.setattr(model._claude, "get_token_summary", lambda: make("claude"))
    monkeypatch.setattr(model._qwen, "get_token_summary", lambda: make("qwen"))
    monkeypatch.setattr(model._minimax, "get_token_summary", lambda: make("minimax"))
    monkeypatch.setattr(model._openai_compat, "get_token_summary", lambda: make("openai_compatible"))

    combined = model.get_token_summary()

    assert set(combined) - {"_total"} == {"azure", "claude", "qwen", "minimax", "openai_compatible"}
    expected_stage_total = {
        "calls": 1,
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    for stage in {"azure", "claude", "qwen", "minimax", "openai_compatible"}:
        assert combined[stage] == expected_stage_total
    assert combined["_total"] == {
        "calls": 5,
        "prompt_tokens": 10,
        "completion_tokens": 15,
        "total_tokens": 25,
    }

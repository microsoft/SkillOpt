"""Tests for model backend routing."""
from __future__ import annotations

import skillopt.model as model


def test_minimax_optimizer_backend_routes_to_minimax(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_minimax_optimizer(**kwargs):
        calls.append(("minimax", kwargs["stage"]))
        return "minimax-ok", {"total_tokens": 1}

    def fake_openai_optimizer(**kwargs):  # pragma: no cover - should not be called
        calls.append(("openai", kwargs["stage"]))
        return "openai-wrong", {"total_tokens": 1}

    original_backend = model.get_optimizer_backend()
    try:
        monkeypatch.setattr(model._minimax, "chat_optimizer", fake_minimax_optimizer)
        monkeypatch.setattr(model._openai, "chat_optimizer", fake_openai_optimizer)
        model.set_optimizer_backend("minimax_chat")

        text, usage = model.chat_optimizer("system", "user", stage="optimizer-test")

        assert text == "minimax-ok"
        assert usage == {"total_tokens": 1}
        assert calls == [("minimax", "optimizer-test")]
    finally:
        model.set_optimizer_backend(original_backend)


def test_minimax_optimizer_messages_backend_routes_to_minimax(monkeypatch) -> None:
    calls: list[str] = []

    def fake_minimax_messages(**kwargs):
        calls.append(kwargs["stage"])
        return "minimax-messages-ok", {"total_tokens": 2}

    original_backend = model.get_optimizer_backend()
    try:
        monkeypatch.setattr(model._minimax, "chat_optimizer_messages", fake_minimax_messages)
        model.set_optimizer_backend("minimax_chat")

        text, usage = model.chat_optimizer_messages(
            [{"role": "user", "content": "hello"}],
            stage="messages-test",
        )

        assert text == "minimax-messages-ok"
        assert usage == {"total_tokens": 2}
        assert calls == ["messages-test"]
    finally:
        model.set_optimizer_backend(original_backend)

from __future__ import annotations

import skillopt.model as model
import skillopt.model.codex_harness as codex_harness


def test_codex_optimizer_backend_routes_chat_optimizer(monkeypatch):
    calls = {}

    def fake_chat_optimizer(**kwargs):
        calls.update(kwargs)
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(model._codex, "chat_optimizer", fake_chat_optimizer)
    model.set_optimizer_backend("codex")
    try:
        response, usage = model.chat_optimizer(system="system", user="user", stage="test")
    finally:
        model.set_optimizer_backend("openai_chat")

    assert response == "ok"
    assert usage["prompt_tokens"] == 1
    assert calls["system"] == "system"
    assert calls["user"] == "user"
    assert calls["stage"] == "test"


def test_codex_optimizer_deployment_is_configurable():
    original = model._codex.OPTIMIZER_DEPLOYMENT
    try:
        model.set_optimizer_deployment("gpt-5.5")
        assert model._codex.OPTIMIZER_DEPLOYMENT == "gpt-5.5"
    finally:
        model.set_optimizer_deployment(original)


def test_codex_target_backend_alias_routes_to_exec_backend():
    original = model.get_target_backend()
    try:
        model.set_target_backend("codex")
        assert model.get_target_backend() == "codex_exec"
        assert model.is_target_exec_backend() is True
    finally:
        model.set_target_backend(original)


def test_codex_target_exec_receives_file_edit_mode(monkeypatch, tmp_path):
    calls = {}

    def fake_sdk_exec(**kwargs):
        raise ModuleNotFoundError("openai_codex_sdk")

    def fake_cli_exec(**kwargs):
        calls.update(kwargs)
        return "done", "raw"

    monkeypatch.setattr(codex_harness, "_run_codex_sdk_exec", fake_sdk_exec)
    monkeypatch.setattr(codex_harness, "_run_codex_cli_exec", fake_cli_exec)

    response, raw = codex_harness.run_codex_exec(
        work_dir=str(tmp_path),
        prompt="Build a Vue/Vite preview.",
        model="gpt-test",
        timeout=1,
        allow_file_edits=True,
    )

    assert response == "done"
    assert raw.startswith("===== CODEX")
    assert calls["allow_file_edits"] is True

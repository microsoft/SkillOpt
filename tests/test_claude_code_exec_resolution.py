"""Config-resolution regressions for the claude_code_exec backend (issue #233).

Route B: ``--backend claude_code_exec`` defaults only the *target* to Claude
Code; the optimizer keeps its configured backend (openai_chat by default).
Opting the optimizer in via ``--optimizer_backend claude_code_exec`` must then
normalize its model to the Claude default rather than leaving ``gpt-5.5``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.train as train_script

_ROOT = Path(__file__).resolve().parents[1]


def _train_cfg(monkeypatch, *extra_argv: str) -> dict:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillopt-train",
            "--config",
            str(_ROOT / "configs" / "searchqa" / "default.yaml"),
            *extra_argv,
        ],
    )
    return train_script.load_config(train_script.parse_args())


def test_train_claude_code_exec_defaults_target_only(monkeypatch) -> None:
    cfg = _train_cfg(monkeypatch, "--backend", "claude_code_exec")

    assert cfg["optimizer_backend"] == "openai_chat"
    assert cfg["optimizer_model"] == "gpt-5.5"
    assert cfg["target_backend"] == "claude_code_exec"
    assert cfg["target_model"] == "claude-sonnet-4-6"


def test_train_claude_code_exec_optimizer_opt_in_normalizes_model(monkeypatch) -> None:
    cfg = _train_cfg(
        monkeypatch,
        "--backend",
        "claude_code_exec",
        "--optimizer_backend",
        "claude_code_exec",
    )

    assert cfg["optimizer_backend"] == "claude_code_exec"
    assert cfg["optimizer_model"] == "claude-sonnet-4-6"
    assert cfg["target_backend"] == "claude_code_exec"
    assert cfg["target_model"] == "claude-sonnet-4-6"


class _StopAfterResolution(Exception):
    pass


def _run_eval_resolution(
    monkeypatch, tmp_path, *, optimizer_backend: str | None
) -> dict:
    import scripts.eval_only as eval_script

    skill_path = tmp_path / "skill.md"
    skill_path.write_text("# Test skill\n", encoding="utf-8")

    cfg = {
        "model": {
            "backend": "azure_openai",
            "optimizer": "gpt-5.5",
            "target": "gpt-5.5",
            "optimizer_backend": "openai_chat",
            "target_backend": "openai_chat",
        },
        "env": {"out_root": str(tmp_path / "out")},
    }
    args = SimpleNamespace(
        config="unused.yaml",
        skill=str(skill_path),
        split=None,
        cfg_options=[],
        backend="claude_code_exec",
        optimizer_backend=optimizer_backend,
    )
    monkeypatch.setattr(eval_script, "parse_args", lambda: args)
    monkeypatch.setattr("skillopt.config.load_config", lambda *a, **kw: cfg)

    observed: dict[str, str] = {}

    def capture(name):
        def _fn(value, *a, **k):
            observed[name] = value

        return _fn

    monkeypatch.setattr(eval_script, "configure_azure_openai", lambda **kw: None)
    monkeypatch.setattr(eval_script, "set_optimizer_backend", capture("optimizer_backend"))
    monkeypatch.setattr(eval_script, "set_target_backend", capture("target_backend"))
    monkeypatch.setattr(eval_script, "set_optimizer_deployment", capture("optimizer_model"))
    monkeypatch.setattr(eval_script, "set_target_deployment", capture("target_model"))

    def stop(*a, **k):
        raise _StopAfterResolution

    monkeypatch.setattr(eval_script, "configure_codex_exec_from_config", stop)

    with pytest.raises(_StopAfterResolution):
        eval_script.main()

    return observed


def test_eval_claude_code_exec_defaults_target_only(monkeypatch, tmp_path) -> None:
    observed = _run_eval_resolution(monkeypatch, tmp_path, optimizer_backend=None)

    assert observed == {
        "optimizer_backend": "openai_chat",
        "target_backend": "claude_code_exec",
        "optimizer_model": "gpt-5.5",
        "target_model": "claude-sonnet-4-6",
    }


def test_eval_claude_code_exec_optimizer_opt_in_normalizes_model(
    monkeypatch, tmp_path
) -> None:
    observed = _run_eval_resolution(
        monkeypatch, tmp_path, optimizer_backend="claude_code_exec"
    )

    assert observed == {
        "optimizer_backend": "claude_code_exec",
        "target_backend": "claude_code_exec",
        "optimizer_model": "claude-sonnet-4-6",
        "target_model": "claude-sonnet-4-6",
    }

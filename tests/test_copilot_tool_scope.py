"""Tests for Copilot CLI tool-scope reduction.

The sleep engine's Copilot backend previously launched the CLI with
``--allow-all-tools``, granting the model unrestricted tool access. It now
passes ``--allowed-tools`` scoped to ``COPILOT_ALLOWED_TOOLS`` (default
``Bash``). These tests capture the constructed argv without spawning the CLI.
"""
from __future__ import annotations

from skillopt_sleep import backend as backend_mod
from skillopt_sleep.backend import CopilotCliBackend


def _make_backend() -> CopilotCliBackend:
    b = CopilotCliBackend.__new__(CopilotCliBackend)
    b.copilot_path = "copilot"
    b.full_env = False
    b.model = ""
    b.copilot_home = ""
    b.timeout = 10
    return b


def _capture_argv(monkeypatch) -> list[str]:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        raise RuntimeError("stop before spawn")

    monkeypatch.setattr(backend_mod.subprocess, "run", fake_run)
    _make_backend()._call("hello")
    return captured["cmd"]


def test_allow_all_tools_removed(monkeypatch) -> None:
    cmd = _capture_argv(monkeypatch)
    assert "--allow-all-tools" not in cmd
    assert "--allowed-tools" in cmd


def test_default_scope_is_bash(monkeypatch) -> None:
    monkeypatch.delenv("COPILOT_ALLOWED_TOOLS", raising=False)
    cmd = _capture_argv(monkeypatch)
    idx = cmd.index("--allowed-tools")
    assert cmd[idx + 1] == "Bash"


def test_env_var_overrides_scope(monkeypatch) -> None:
    monkeypatch.setenv("COPILOT_ALLOWED_TOOLS", "Read")
    cmd = _capture_argv(monkeypatch)
    idx = cmd.index("--allowed-tools")
    assert cmd[idx + 1] == "Read"

"""Tests for Copilot CLI tool-scope reduction.

The sleep engine's Copilot backend used to launch the CLI with bare
``--allow-all-tools``, granting the model every tool the CLI exposes. Scoping
now happens on the *visibility* axis via ``--available-tools`` (default
``bash``, overridable with ``COPILOT_AVAILABLE_TOOLS``).

The two axes are independent and both are needed:

* ``--allow-all-tools`` waives the interactive approval prompt. The CLI's own
  help states it is "required for non-interactive mode", so dropping it makes
  every headless call hang or fail -- it is not the flag to scope on.
* ``--available-tools`` is what actually narrows the surface: "Only these tools
  will be available to the model".

Verified against GitHub Copilot CLI 1.0.80: ``--allowed-tools`` does not exist
(``error: unknown option``), and the selector is the lowercase tool name
``bash`` -- ``--available-tools=Bash`` silently blocks the tool instead of
allowing it, since selectors are case-sensitive.

These tests capture the constructed argv without spawning the CLI.
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


def test_nonexistent_allowed_tools_flag_is_never_sent(monkeypatch) -> None:
    # Guards the original regression: `--allowed-tools` is not a Copilot CLI
    # option, so sending it aborts the process before any work happens.
    cmd = _capture_argv(monkeypatch)
    assert "--allowed-tools" not in cmd


def test_non_interactive_permission_flag_is_kept(monkeypatch) -> None:
    cmd = _capture_argv(monkeypatch)
    assert "--allow-all-tools" in cmd


def test_visibility_is_scoped(monkeypatch) -> None:
    cmd = _capture_argv(monkeypatch)
    assert "--available-tools" in cmd


def test_default_scope_is_lowercase_bash(monkeypatch) -> None:
    monkeypatch.delenv("COPILOT_AVAILABLE_TOOLS", raising=False)
    cmd = _capture_argv(monkeypatch)
    idx = cmd.index("--available-tools")
    # Lowercase matters: selectors are case-sensitive and "Bash" matches nothing.
    assert cmd[idx + 1] == "bash"


def test_env_var_overrides_scope(monkeypatch) -> None:
    monkeypatch.setenv("COPILOT_AVAILABLE_TOOLS", "bash,write")
    cmd = _capture_argv(monkeypatch)
    idx = cmd.index("--available-tools")
    assert cmd[idx + 1] == "bash,write"

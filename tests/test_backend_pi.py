"""Tests for the pi CLI backend (`--backend pi`)."""
from __future__ import annotations

from unittest import mock

from skillopt_sleep.backend import PiCliBackend, get_backend


class _FakeProc:
    def __init__(self, stdout: str, stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr


def test_get_backend_pi_aliases():
    for alias in ("pi", "pi_cli", "pi_coding_agent", "pi-coding-agent", "PI"):
        be = get_backend(alias, model="zai/glm-5.2")
        assert isinstance(be, PiCliBackend), alias
        assert be.name == "pi"


def test_default_model_from_env(monkeypatch):
    monkeypatch.setenv("SKILLOPT_SLEEP_PI_MODEL", "zai/glm-5.2")
    be = PiCliBackend()
    assert be.model == "zai/glm-5.2"
    assert be.pi_path == "pi"


def test_call_builds_isolated_command_and_returns_stdout():
    be = PiCliBackend(model="zai/glm-5.2", pi_path="/usr/local/bin/pi")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        return _FakeProc("answer text")

    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
        out = be._call("do the thing")

    assert out == "answer text"
    cmd = captured["cmd"]
    assert cmd[0:2] == ["/usr/local/bin/pi", "-p"]
    # isolation flags must be present (no ambient skills/context/tools)
    assert "--no-tools" in cmd
    assert "--no-skills" in cmd
    assert "--no-context-files" in cmd
    assert "--no-extensions" in cmd
    assert "--no-session" in cmd
    assert "--model" in cmd and "zai/glm-5.2" in cmd
    assert cmd[-1] == "do the thing"
    # ran from a clean temp cwd, not inherited
    assert captured["cwd"] is not None and captured["cwd"] != ""


def test_call_detects_auth_error_and_logs():
    be = PiCliBackend()
    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        return_value=_FakeProc("", stderr="Authentication required: not logged in"),
    ):
        out = be._call("hi")
    assert out == ""  # empty stdout
    assert "Authentication required" in be.last_call_error

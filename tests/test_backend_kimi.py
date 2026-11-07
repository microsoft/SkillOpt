"""Tests for the Kimi Code CLI backend (`--backend kimicode`)."""
from __future__ import annotations

import os
import subprocess
from unittest import mock

from skillopt_sleep.backend import (
    _NO_WINDOW,
    KimiCliBackend,
    build_backend,
    get_backend,
)


class _FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_get_backend_kimi_aliases():
    for alias in ("kimi", "kimicode", "kimi_code", "kimi-cli", "kimi_cli", "KIMICODE"):
        be = get_backend(alias, model="k2")
        assert isinstance(be, KimiCliBackend), alias
        assert be.name == "kimicode"
        assert be.model == "k2"


def test_get_backend_kimi_path_override():
    be = get_backend("kimicode", kimi_path="/opt/kimi/bin/kimi")
    assert isinstance(be, KimiCliBackend)
    assert be.kimi_path == "/opt/kimi/bin/kimi"


def test_default_model_from_env(monkeypatch):
    monkeypatch.setenv("SKILLOPT_SLEEP_KIMI_MODEL", "kimi-for-coding")
    be = KimiCliBackend()
    assert be.model == "kimi-for-coding"


def test_path_falls_back_to_desktop_install(monkeypatch):
    # PATH lookup fails -> the desktop install location is used when it exists
    with mock.patch("shutil.which", return_value=None):
        with mock.patch("os.path.exists", return_value=True):
            be = KimiCliBackend(kimi_path="kimi")
    assert be.kimi_path.endswith(os.path.join(".kimi-code", "bin", "kimi.exe"))


def test_call_builds_isolated_command_and_decodes_utf8():
    be = KimiCliBackend(model="k2", kimi_path="/usr/local/bin/kimi")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return _FakeProc("• OK".encode("utf-8"))

    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
        out = be._call("Reply with exactly: OK")

    assert out == "• OK"
    cmd = captured["cmd"]
    assert cmd[0] == "/usr/local/bin/kimi"
    # kimi's --prompt takes the prompt INLINE (unlike claude -p via stdin):
    # --output-format must not be swallowed as the prompt value.
    assert "--prompt" in cmd
    assert cmd[cmd.index("--prompt") + 1] == "Reply with exactly: OK"
    assert cmd[cmd.index("--output-format") + 1] == "text"
    assert cmd[cmd.index("-m") + 1] == "k2"
    # isolated cwd, no console window, no stdin prompt
    assert captured["cwd"] and "skillopt_sleep_kimi_" in captured["cwd"]
    assert captured["creationflags"] == _NO_WINDOW
    assert "input" not in captured


def test_call_decodes_utf8_despite_gbk_locale():
    be = KimiCliBackend(kimi_path="/usr/local/bin/kimi")
    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        return_value=_FakeProc("中文回答".encode("utf-8")),
    ):
        out = be._call("question")
    assert out == "中文回答"


def test_call_flags_auth_error(monkeypatch):
    be = KimiCliBackend(kimi_path="/usr/local/bin/kimi")
    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        return_value=_FakeProc(b"", stderr="Error: Invalid API key".encode("utf-8")),
    ):
        out = be._call("question")
    assert out == ""
    assert "Invalid API key" in be.last_call_error


def test_build_backend_routes_kimi_path():
    be = build_backend(backend="kimicode", kimi_path="/opt/kimi/bin/kimi")
    assert isinstance(be, KimiCliBackend)
    assert be.kimi_path == "/opt/kimi/bin/kimi"


def test_default_timeout_is_generous_and_env_overridable(monkeypatch):
    monkeypatch.delenv("SKILLOPT_SLEEP_KIMI_TIMEOUT", raising=False)
    be = KimiCliBackend(kimi_path="kimi")
    assert be.timeout == 600  # cold starts under desktop-app load exceed 180s
    monkeypatch.setenv("SKILLOPT_SLEEP_KIMI_TIMEOUT", "900")
    be = KimiCliBackend(kimi_path="kimi")
    assert be.timeout == 900


def test_timeout_is_reported_distinctly(monkeypatch):
    be = KimiCliBackend(kimi_path="/usr/local/bin/kimi")
    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="kimi", timeout=be.timeout),
    ):
        out = be._call("question")
    assert out == ""
    assert "timed out" in be.last_call_error

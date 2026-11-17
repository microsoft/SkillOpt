"""CLAUDE_SETTING_SOURCES must be able to ask the Claude CLI for no settings at all."""
from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import pytest

from skillopt.model import claude_backend


def _fresh_backend(monkeypatch, value: str | None):
    if value is None:
        monkeypatch.delenv("CLAUDE_SETTING_SOURCES", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_SETTING_SOURCES", value)
    spec = importlib.util.spec_from_file_location("_claude_backend_fresh", claude_backend.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_argv(monkeypatch, module) -> list[str]:
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        payload = {"type": "result", "result": "ok", "usage": {}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._run_claude_print(
        system="system",
        prompt="prompt",
        model="",
        tools=None,
        tool_choice=None,
        return_message=False,
        timeout=10,
    )
    return captured["cmd"]


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, "user,project"),
        # Omitting the flag makes the CLI load every source (user, project and
        # local), so an empty value must reach it as an explicit empty list.
        ("", ""),
        ("  ", ""),
        ("user", "user"),
    ],
)
def test_setting_sources_flag_follows_the_environment(monkeypatch, env_value, expected) -> None:
    module = _fresh_backend(monkeypatch, env_value)

    cmd = _cli_argv(monkeypatch, module)

    assert cmd.count("--setting-sources") == 1
    assert cmd[cmd.index("--setting-sources") + 1] == expected

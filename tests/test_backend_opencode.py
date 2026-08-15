"""Tests for the OpenCode CLI backend."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from skillopt_sleep import cycle
from skillopt_sleep.__main__ import _add_common, _cfg_from_args
from skillopt_sleep.backend import (
    _NO_WINDOW,
    _OPENCODE_SYNTHETIC_TOOL_QUERY,
    _OPENCODE_SYNTHETIC_TOOL_RESULT,
    DualBackend,
    MockBackend,
    OpenCodeCliBackend,
    OpenCodeError,
    _parse_opencode_jsonl_events,
    build_backend,
    get_backend,
    resolve_opencode_path,
)
from skillopt_sleep.config import DEFAULTS, SleepConfig, load_config
from skillopt_sleep.types import TaskRecord


class _FakeProc:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _event(event_type: str, *, session: str = "session-1", part=None) -> str:
    event = {"type": event_type, "sessionID": session}
    if part is not None:
        event["part"] = part
    return json.dumps(event)


def _success_stream(*texts: str) -> str:
    lines = [
        _event("step_start", part={"type": "step-start"}),
        *(_event("text", part={"type": "text", "text": text}) for text in texts),
        _event("step_finish", part={"type": "step-finish"}),
    ]
    return "\n".join(lines)


def _tool_event(
    tool_id: str,
    *,
    call_id: str = "call-1",
    status: str = "completed",
    output: str | None = None,
    query=_OPENCODE_SYNTHETIC_TOOL_QUERY,
) -> str:
    state = {"status": status, "input": {"query": query}}
    if output is not None:
        state["output"] = output
    return _event(
        "tool_use",
        part={"type": "tool", "tool": tool_id, "callID": call_id, "state": state},
    )


def _tool_success_stream(tool_id: str, text: str = "answer") -> str:
    return "\n".join(
        [
            _event("step_start", part={"type": "step-start"}),
            _tool_event(tool_id, output=_OPENCODE_SYNTHETIC_TOOL_RESULT),
            _event("step_finish", part={"type": "step-finish"}),
            _event("step_start", part={"type": "step-start"}),
            _event("text", part={"type": "text", "text": text}),
            _event("step_finish", part={"type": "step-finish"}),
        ]
    )


def _resolved_mcp(
    *names: str,
    disabled: bool = False,
    snapshot: bool | None = None,
) -> str:
    mcp = {
        name: {
            "type": "local",
            "command": ["mcp-server"],
            **({"enabled": False} if disabled else {}),
        }
        for name in names
    }
    resolved = {"mcp": mcp}
    if snapshot is not None:
        resolved["snapshot"] = snapshot
    return json.dumps(resolved)


def _successful_plain_results(*mcp_names: str, answer: str = "answer") -> list[_FakeProc]:
    return [
        _FakeProc(_resolved_mcp(*mcp_names, snapshot=False)),
        _FakeProc(_resolved_mcp(*mcp_names, disabled=True, snapshot=False)),
        _FakeProc(_success_stream(answer)),
    ]


def _assert_replay_permissions(config, env, agent_name: str, expected_permission: dict[str, str]) -> None:
    assert json.loads(env["OPENCODE_PERMISSION"]) == expected_permission
    assert config["permission"] == expected_permission
    assert config["agent"][agent_name]["permission"] == expected_permission


def _assert_controlled_tool_environment(env, work: str) -> None:
    expected_npm = {
        "npm_config_audit": "false",
        "npm_config_cache": os.path.join(work, ".npm-cache"),
        "npm_config_fetch_retries": "0",
        "npm_config_fetch_retry_maxtimeout": "100",
        "npm_config_fetch_retry_mintimeout": "100",
        "npm_config_fetch_timeout": "1000",
        "npm_config_fund": "false",
        "npm_config_offline": "true",
        "npm_config_update_notifier": "false",
    }
    for key, value in expected_npm.items():
        matches = {
            candidate: candidate_value for candidate, candidate_value in env.items() if candidate.upper() == key.upper()
        }
        assert matches == {key: value}

    assert env["OPENCODE_DISABLE_LSP_DOWNLOAD"] == "1"
    assert env["OPENCODE_DISABLE_MODELS_FETCH"] == "1"


def _assert_replay_project_artifacts(
    work: str,
    *,
    project_id: str,
    tool_id: str,
    forbidden_text: str,
) -> None:
    root = Path(work)
    assert (root / ".git").is_dir()
    assert (root / ".git" / "opencode").read_text(encoding="ascii") == project_id + "\n"

    tool_files = list((root / ".opencode" / "tools").glob("*.js"))
    assert [path.stem for path in tool_files] == [tool_id]
    assert (root / ".opencode" / "node_modules").is_dir()

    package = json.loads((root / ".opencode" / "package.json").read_text())
    lock = json.loads((root / ".opencode" / "package-lock.json").read_text())
    assert package["dependencies"] == {"@opencode-ai/plugin": "0.0.0"}
    assert lock["packages"][""]["dependencies"] == package["dependencies"]

    source = tool_files[0].read_text(encoding="utf-8")
    assert forbidden_text not in source
    assert json.dumps(_OPENCODE_SYNTHETIC_TOOL_QUERY) in source
    assert json.dumps(_OPENCODE_SYNTHETIC_TOOL_RESULT) in source


def test_resolve_opencode_path_precedence(monkeypatch):
    monkeypatch.setenv("SKILLOPT_SLEEP_OPENCODE_PATH", "env-opencode")
    with mock.patch("shutil.which", side_effect=lambda value: os.path.abspath(f"resolved-{value}")):
        assert resolve_opencode_path("explicit-opencode") == os.path.abspath("resolved-explicit-opencode")
        assert resolve_opencode_path() == os.path.abspath("resolved-env-opencode")


def test_resolve_opencode_path_falls_back_to_command(monkeypatch):
    monkeypatch.delenv("SKILLOPT_SLEEP_OPENCODE_PATH", raising=False)
    with mock.patch("shutil.which", return_value=None):
        assert resolve_opencode_path() == "opencode"


@pytest.mark.skipif(os.name != "nt", reason="Windows PATHEXT shim behavior")
def test_resolve_opencode_path_preserves_windows_cmd_shim(monkeypatch):
    monkeypatch.delenv("SKILLOPT_SLEEP_OPENCODE_PATH", raising=False)
    executable = r"C:\npm\opencode.CMD"

    with mock.patch("shutil.which", return_value=executable):
        assert resolve_opencode_path() == executable


@pytest.mark.parametrize("which_result", [None, os.path.join("bin", "opencode")])
@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_resolve_opencode_path_anchors_relative_paths(monkeypatch, tmp_path, which_result, source):
    monkeypatch.chdir(tmp_path)
    relative = os.path.join("bin", "opencode")
    explicit = relative if source == "explicit" else ""
    if source == "environment":
        monkeypatch.setenv("SKILLOPT_SLEEP_OPENCODE_PATH", relative)

    with mock.patch("shutil.which", return_value=which_result):
        assert resolve_opencode_path(explicit) == os.path.abspath(relative)


def test_constructor_uses_explicit_or_environment_model(monkeypatch):
    monkeypatch.setenv("SKILLOPT_SLEEP_OPENCODE_MODEL", "env/model")
    with mock.patch("shutil.which", return_value=None):
        assert OpenCodeCliBackend(model="explicit/model").model == "explicit/model"
        assert OpenCodeCliBackend().model == "env/model"


def test_backend_runtime_uses_current_mutable_settings():
    be = OpenCodeCliBackend(model="old/model", opencode_path="old-opencode", timeout=1)
    be.model = "new/model"
    be.opencode_path = "new-opencode"
    be.timeout = 7

    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=_successful_plain_results(),
    ) as run:
        assert be._call("hello") == "answer"

    assert all(call.args[0][0] == "new-opencode" for call in run.call_args_list)
    assert all(call.kwargs["timeout"] == 7 for call in run.call_args_list)
    command = run.call_args_list[-1].args[0]
    assert command[command.index("--model") + 1] == "new/model"


def test_parse_opencode_jsonl_collects_text_and_ignores_extensions():
    raw = "\n".join(
        [
            _event("step_start", part={"type": "step-start"}),
            _event("future_event"),
            _event("reasoning", part={"type": "reasoning", "text": "hidden"}),
            _event("text", part={"type": "text", "text": "first"}),
            "",
            _event("text", part={"type": "text", "text": "second"}),
            _event("step_finish", part={"type": "step-finish"}),
        ]
    )
    assert _parse_opencode_jsonl_events(raw) == ("first\nsecond", [], "")


@pytest.mark.parametrize(
    ("raw", "expected_code"),
    [
        ("not json", "malformed_jsonl"),
        ('{"value":' + "9" * 5000 + "}", "malformed_jsonl"),
        (json.dumps(["not", "an", "object"]), "invalid_event"),
        (json.dumps({"type": "text"}), "invalid_event"),
        (
            "\n".join(
                [
                    _event("step_start", session="a", part={"type": "step-start"}),
                    _event("step_finish", session="b", part={"type": "step-finish"}),
                ]
            ),
            "mixed_session",
        ),
        (_event("error"), "error_event"),
        (_event("reasoning", part={"type": "text"}), "invalid_event"),
        (
            "\n".join(
                [
                    _event("step_start", part={"type": "step-start"}),
                    _event("tool_use", part={"type": "tool", "tool": "shell"}),
                ]
            ),
            "unexpected_tool_event",
        ),
        (_event("step_start", part={"type": "step-start"}), "incomplete_stream"),
        (_event("step_finish", part={"type": "step-finish"}), "incomplete_stream"),
        (_success_stream("  "), "empty_response"),
    ],
)
def test_parse_opencode_jsonl_rejects_invalid_streams(raw, expected_code):
    assert _parse_opencode_jsonl_events(raw) == ("", [], expected_code)


def test_parse_opencode_tool_jsonl_accepts_only_expected_completed_tools():
    tool_id = "skillopt_replay_abc123"
    raw = _tool_success_stream(tool_id, "final answer")

    assert _parse_opencode_jsonl_events(raw, {tool_id}) == (
        "final answer",
        [tool_id],
        "",
    )


@pytest.mark.parametrize(
    ("raw", "expected_code"),
    [
        (_tool_success_stream("other"), "unexpected_tool_id"),
        (
            "\n".join(
                [
                    _event("step_start", part={"type": "step-start"}),
                    _tool_event("expected", status="error"),
                ]
            ),
            "tool_error",
        ),
        (
            "\n".join(
                [
                    _event("step_start", part={"type": "step-start"}),
                    _tool_event("expected", output="unexpected result"),
                    _event("step_finish", part={"type": "step-finish"}),
                ]
            ),
            "invalid_tool_event",
        ),
        (
            "\n".join(
                [
                    _event("step_start", part={"type": "step-start"}),
                    _tool_event(
                        "expected",
                        query="task-derived query",
                        output=_OPENCODE_SYNTHETIC_TOOL_RESULT,
                    ),
                ]
            ),
            "invalid_tool_event",
        ),
        (
            "\n".join(
                [
                    _event("step_start", part={"type": "step-start"}),
                    _tool_event("expected", output=_OPENCODE_SYNTHETIC_TOOL_RESULT),
                    _event("step_finish", part={"type": "step-finish"}),
                ]
            ),
            "missing_final_text",
        ),
    ],
)
def test_parse_opencode_tool_jsonl_rejects_unverified_tools(raw, expected_code):
    assert _parse_opencode_jsonl_events(raw, {"expected"}) == ("", [], expected_code)


def test_call_uses_stdin_temp_workspace_and_user_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-provider-key")
    monkeypatch.setenv("HOME", "/home/example")
    monkeypatch.setenv("PWD", "/original/project")
    monkeypatch.setenv("OLDPWD", "/previous/project")
    inline_config = json.dumps(
        {
            "default_agent": "unsafe-user-agent",
            "agent": {"unsafe-user-agent": {"permission": {"bash": "allow", "edit": "allow"}}},
        }
    )
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", inline_config)
    config_path = str(tmp_path / "custom-opencode.json")
    config_dir = str(tmp_path / ".opencode")
    monkeypatch.setenv("OPENCODE_CONFIG", config_path)
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", config_dir)
    for key in (
        "NO_COLOR",
        "OPENCODE_DISABLE_AUTOUPDATE",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS",
        "OPENCODE_DISABLE_PROJECT_CONFIG",
        "OPENCODE_DISABLE_SHARE",
        "OPENCODE_DISABLE_TERMINAL_TITLE",
        "OPENCODE_PURE",
    ):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("OPENCODE_PERMISSION", '{"*":"allow"}')
    monkeypatch.delenv("OPENCODE_DISABLE_DEFAULT_PLUGINS", raising=False)
    captured = []
    results = iter(_successful_plain_results("alpha", "beta"))

    def fake_run(cmd, **kwargs):
        snapshot = dict(kwargs)
        snapshot["env"] = kwargs["env"].copy()
        captured.append((cmd, snapshot))
        assert os.path.isdir(kwargs["cwd"])
        assert os.listdir(kwargs["cwd"]) == []
        return next(results)

    executable = str(tmp_path / "opencode")
    be = OpenCodeCliBackend(
        model="provider/model",
        opencode_path=executable,
        timeout=37,
    )
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
        assert be._call("do the thing") == "answer"

    discovery_cmd, discovery_call = captured[0]
    verification_cmd, verification_call = captured[1]
    cmd, run_call = captured[2]
    expected_child_env = {
        "NO_COLOR": "1",
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
        "OPENCODE_DISABLE_SHARE": "1",
        "OPENCODE_DISABLE_TERMINAL_TITLE": "1",
        "OPENCODE_PERMISSION": '{"*":"deny"}',
        "OPENCODE_PURE": "1",
    }
    for _, call in captured:
        assert call["capture_output"] is True
        assert call["creationflags"] == _NO_WINDOW
        assert call["text"] is True
        assert call["encoding"] == "utf-8"
        assert call["errors"] == "replace"
        assert call["timeout"] == be.timeout
        assert call["cwd"] == run_call["cwd"]
        assert call["env"]["PWD"] == call["cwd"]
        assert "OLDPWD" not in call["env"]
        for key, value in expected_child_env.items():
            assert call["env"][key] == value
    assert discovery_cmd == [executable, "debug", "config", "--pure"]
    assert verification_cmd == discovery_cmd
    assert cmd[:5] == [
        executable,
        "run",
        "--pure",
        "--format",
        "json",
    ]
    agent_name = cmd[cmd.index("--agent") + 1]
    assert agent_name.startswith("skillopt-sleep-")
    assert cmd[cmd.index("--title") + 1] == "skillopt-sleep"
    assert cmd[cmd.index("--dir") + 1] == run_call["cwd"]
    assert cmd[cmd.index("--model") + 1] == "provider/model"
    assert run_call["input"] == "do the thing"
    assert "input" not in discovery_call
    assert "input" not in verification_call
    assert "do the thing" not in cmd
    assert run_call["env"]["OPENAI_API_KEY"] == "ambient-provider-key"
    assert run_call["env"]["HOME"] == "/home/example"
    assert run_call["env"]["PWD"] != "/original/project"
    assert run_call["env"]["OPENCODE_CONFIG"] == config_path
    assert run_call["env"]["OPENCODE_CONFIG_DIR"] == config_dir
    discovered = json.loads(discovery_call["env"]["OPENCODE_CONFIG_CONTENT"])
    assert "mcp" not in discovered
    injected = json.loads(run_call["env"]["OPENCODE_CONFIG_CONTENT"])
    assert injected == {
        "snapshot": False,
        "agent": {
            agent_name: {
                "mode": "primary",
                "permission": {"*": "deny"},
            }
        },
        "mcp": {
            "alpha": {"enabled": False},
            "beta": {"enabled": False},
        },
    }
    assert verification_call["env"]["OPENCODE_CONFIG_CONTENT"] == run_call["env"]["OPENCODE_CONFIG_CONTENT"]
    assert "OPENCODE_DISABLE_DEFAULT_PLUGINS" not in run_call["env"]
    assert not os.path.exists(run_call["cwd"])
    assert be.last_call_error == ""


def test_relative_opencode_config_paths_become_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config_path = os.path.join("profiles", "opencode.json")
    config_dir = os.path.join("profiles", "opencode")
    monkeypatch.setenv("OPENCODE_CONFIG", config_path)
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", config_dir)
    captured = []
    results = iter(_successful_plain_results())

    def fake_run(cmd, **kwargs):
        captured.append(kwargs["env"].copy())
        return next(results)

    be = OpenCodeCliBackend(opencode_path=str(tmp_path / "opencode"))
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
        assert be._call("hello") == "answer"

    assert all(env["OPENCODE_CONFIG"] == os.path.abspath(config_path) for env in captured)
    assert all(env["OPENCODE_CONFIG_DIR"] == os.path.abspath(config_dir) for env in captured)


def test_empty_model_leaves_model_selection_to_opencode(monkeypatch):
    monkeypatch.delenv("SKILLOPT_SLEEP_OPENCODE_MODEL", raising=False)
    be = OpenCodeCliBackend(opencode_path="opencode")
    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=_successful_plain_results(),
    ) as run:
        assert be._call("hello") == "answer"
    assert "--model" not in run.call_args_list[-1].args[0]


def test_each_call_uses_a_new_agent_name():
    be = OpenCodeCliBackend(opencode_path="opencode")
    with (
        mock.patch("secrets.token_hex", side_effect=["a" * 32, "b" * 32]),
        mock.patch(
            "skillopt_sleep.backend.subprocess.run",
            side_effect=_successful_plain_results() + _successful_plain_results(),
        ) as run,
    ):
        assert be._call("first") == "answer"
        assert be._call("second") == "answer"

    commands = [call.args[0] for call in run.call_args_list if call.args[0][1] == "run"]
    assert commands[0][commands[0].index("--agent") + 1] == ("skillopt-sleep-" + "a" * 32)
    assert commands[1][commands[1].index("--agent") + 1] == ("skillopt-sleep-" + "b" * 32)


@pytest.mark.parametrize(
    ("run_result", "error_fragment"),
    [
        (subprocess.TimeoutExpired("opencode", 1), "timed out"),
        (OSError("secret path"), "could not be executed"),
        (_FakeProc("misleading", returncode=2), "exited 2"),
        (_FakeProc("not json"), "malformed JSONL"),
    ],
)
def test_call_records_process_and_protocol_failures(run_result, error_fragment):
    be = OpenCodeCliBackend(opencode_path="opencode", timeout=1)
    effects = _successful_plain_results()[:2] + [run_result]
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=effects):
        assert be._call("hello") == ""
    assert error_fragment in be.last_call_error
    assert "secret path" not in be.last_call_error


def test_call_handles_workspace_creation_failure_without_starting_child():
    be = OpenCodeCliBackend(opencode_path="opencode")
    with (
        mock.patch(
            "skillopt_sleep.backend.tempfile.TemporaryDirectory",
            side_effect=OSError("private workspace detail"),
        ),
        mock.patch("skillopt_sleep.backend.subprocess.run") as run,
    ):
        assert be._call("hello") == ""

    run.assert_not_called()
    assert "workspace could not be prepared" in be.last_call_error
    assert "private workspace detail" not in be.last_call_error


@pytest.mark.parametrize(
    ("discovered_names", "final_mcp"),
    [
        pytest.param(
            ("global-server",),
            {
                "global-server": {
                    "type": "local",
                    "command": ["mcp-server"],
                    "enabled": True,
                }
            },
            id="known-server-re-enabled",
        ),
        pytest.param(
            ("known-server",),
            {
                "known-server": {"enabled": False},
                "new-server": {
                    "type": "local",
                    "command": ["mcp-server"],
                    "enabled": True,
                },
            },
            id="new-server-enabled",
        ),
    ],
)
def test_enabled_mcp_in_final_config_stops_before_model_call(discovered_names, final_mcp):
    discovered = _FakeProc(_resolved_mcp(*discovered_names))
    verification = _FakeProc(json.dumps({"mcp": final_mcp}))
    be = OpenCodeCliBackend(opencode_path="opencode")

    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=[discovered, verification],
    ) as run:
        assert be._call("hello") == ""

    assert run.call_count == 2
    assert all(call.args[0][1:3] == ["debug", "config"] for call in run.call_args_list)
    assert "disable every configured MCP server" in be.last_call_error


def test_plain_replay_fails_if_snapshots_remain_enabled():
    be = OpenCodeCliBackend(opencode_path="opencode")
    results = [
        _FakeProc(_resolved_mcp(snapshot=False)),
        _FakeProc(_resolved_mcp(snapshot=True)),
    ]

    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=results) as run:
        assert be._call("hello") == ""

    assert run.call_count == 2
    assert "disable session snapshots" in be.last_call_error


@pytest.mark.parametrize(
    ("bad_result", "error_fragment"),
    [
        (subprocess.TimeoutExpired("opencode", 1), "timed out"),
        (OSError("private config path"), "could not be completed"),
        (_FakeProc("", returncode=3), "exited 3"),
        (_FakeProc("not json"), "invalid configuration"),
        (_FakeProc('{"mcp":' + "9" * 5000 + "}"), "invalid configuration"),
        (_FakeProc(json.dumps(["not", "an", "object"])), "invalid configuration"),
        (_FakeProc(json.dumps({"mcp": []})), "invalid MCP configuration"),
        (
            _FakeProc(json.dumps({"mcp": {"server": "not an object"}})),
            "invalid MCP configuration",
        ),
    ],
)
def test_mcp_discovery_failure_stops_before_model_call(bad_result, error_fragment):
    be = OpenCodeCliBackend(opencode_path="opencode", timeout=1)

    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=[bad_result],
    ) as run:
        assert be._call("hello") == ""

    assert run.call_count == 1
    assert run.call_args.args[0][1:3] == ["debug", "config"]
    assert error_fragment in be.last_call_error
    assert "private config path" not in be.last_call_error


def test_mcp_verification_failure_stops_before_model_call():
    be = OpenCodeCliBackend(opencode_path="opencode")

    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=[_FakeProc(_resolved_mcp("server")), _FakeProc("not json")],
    ) as run:
        assert be._call("hello") == ""

    assert run.call_count == 2
    assert all(call.args[0][1:3] == ["debug", "config"] for call in run.call_args_list)
    assert "MCP verification" in be.last_call_error


def test_resolved_config_secrets_are_not_exposed(caplog):
    secret = "resolved-provider-secret-74f92"
    discovered = _FakeProc(
        json.dumps(
            {
                "provider": {"custom": {"options": {"apiKey": secret}}},
                "mcp": {
                    "private-server": {
                        "type": "remote",
                        "url": "https://mcp.invalid",
                        "headers": {"Authorization": secret},
                    }
                },
            }
        )
    )
    verification = _FakeProc(secret, stderr=secret, returncode=5)
    be = OpenCodeCliBackend(opencode_path="opencode")

    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=[discovered, verification],
    ) as run:
        assert be._call("hello") == ""

    assert run.call_count == 2
    assert secret not in be.last_call_error
    assert secret not in caplog.text
    assert secret not in run.call_args.kwargs["env"]["OPENCODE_CONFIG_CONTENT"]


def test_mcp_checks_run_once_per_cache_miss():
    be = OpenCodeCliBackend(opencode_path="opencode")
    results = _successful_plain_results("server") + _successful_plain_results("server")

    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=results) as run:
        assert be._cached_call("attempt:first", "first prompt") == "answer"
        assert be._cached_call("attempt:first", "first prompt") == "answer"
        assert be._cached_call("attempt:second", "second prompt") == "answer"

    commands = [call.args[0][1:3] for call in run.call_args_list]
    assert commands == [
        ["debug", "config"],
        ["debug", "config"],
        ["run", "--pure"],
        ["debug", "config"],
        ["debug", "config"],
        ["run", "--pure"],
    ]


def test_failed_call_is_not_cached():
    be = OpenCodeCliBackend(opencode_path="opencode")
    with mock.patch.object(be, "_call", side_effect=["", "recovered"]) as call:
        assert be._cached_call("attempt:key", "prompt") == ""
        assert be._cached_call("attempt:key", "prompt") == "recovered"
    assert call.call_count == 2


def test_cached_success_clears_stale_call_error():
    be = OpenCodeCliBackend(opencode_path="opencode")
    with mock.patch.object(be, "_call", return_value="answer") as call:
        assert be._cached_call("attempt:key", "prompt") == "answer"
        be.last_call_error = "unrelated later failure"
        assert be._cached_call("attempt:key", "prompt") == "answer"
    assert call.call_count == 1
    assert be.last_call_error == ""


def test_attempt_with_tools_requires_explicit_opt_in_without_starting_child():
    be = OpenCodeCliBackend(opencode_path="opencode")
    with mock.patch("skillopt_sleep.backend.subprocess.run") as run:
        assert be.attempt_with_tools(mock.Mock(), "skill", "memory", ["search"]) == ("", [])
    run.assert_not_called()
    assert "opencode_tool_replay" in be.last_call_error


@pytest.mark.parametrize("configured", [False, 0, 1, "false", "true", None])
def test_tool_replay_opt_in_requires_boolean_true(configured):
    be = OpenCodeCliBackend(
        opencode_path="opencode",
        tool_replay=configured,
    )

    assert be.tool_replay is False


def test_attempt_with_tools_builds_isolated_random_replay_request(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret-stays-in-env")
    monkeypatch.setenv("OPENCODE_DIRECT_TRACE", "1")
    for key in ("GIT_COMMON_DIR", "GIT_DIR", "GIT_OBJECT_DIRECTORY", "GIT_WORK_TREE"):
        monkeypatch.setenv(key, str(tmp_path / "outside-repository"))
    monkeypatch.setenv("git_dir", str(tmp_path / "lowercase-outside-repository"))
    hostile_npm = {
        "NPM_CONFIG_AUDIT": "true",
        "NpM_CoNfIg_CaChE": str(tmp_path / "outside-npm-cache"),
        "NPM_CONFIG_FETCH_RETRIES": "99",
        "npm_CONFIG_FETCH_RETRY_MAXTIMEOUT": "99999",
        "NPM_CONFIG_FETCH_RETRY_MINTIMEOUT": "99999",
        "NPM_CONFIG_FETCH_TIMEOUT": "99999",
        "nPm_CoNfIg_FuNd": "true",
        "NPM_CONFIG_OFFLINE": "false",
        "NPM_CONFIG_UPDATE_NOTIFIER": "true",
    }
    for key, value in hostile_npm.items():
        monkeypatch.setenv(key, value)

    captured = []
    project_id = "skillopt-sleep-" + "a" * 64
    agent_name = "skillopt-sleep-" + "b" * 32
    tool_id = "skillopt_replay_" + "c" * 32
    expected_permission = {"*": "deny", tool_id: "allow"}
    intent = "OPENCODE_INTENT_SENTINEL_1937"
    secret_context = "task-secret-value-4819"
    skill = "OPENCODE_SKILL_SENTINEL_2864"
    memory = "OPENCODE_MEMORY_SENTINEL_7351"

    def fake_run(cmd, **kwargs):
        snapshot = dict(kwargs)
        snapshot["env"] = kwargs["env"].copy()
        captured.append((cmd, snapshot))
        _assert_controlled_tool_environment(kwargs["env"], kwargs["cwd"])
        if cmd[1:3] == ["debug", "config"]:
            config = json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])
            _assert_replay_permissions(config, kwargs["env"], agent_name, expected_permission)
            disabled = bool(config.get("mcp"))
            return _FakeProc(
                _resolved_mcp(
                    "configured-server",
                    disabled=disabled,
                    snapshot=False,
                )
            )
        if cmd[1:3] == ["debug", "agent"]:
            config = json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])
            _assert_replay_permissions(config, kwargs["env"], agent_name, expected_permission)
            assert cmd[3] == agent_name
            agent = config["agent"][agent_name]
            assert agent["model"] == "provider/model"
            return _FakeProc(json.dumps({"tools": {"bash": False, tool_id: True}}))

        work = kwargs["cwd"]
        config = json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])
        _assert_replay_permissions(config, kwargs["env"], agent_name, expected_permission)
        assert config["snapshot"] is False
        assert cmd[cmd.index("--agent") + 1] == agent_name
        _assert_replay_project_artifacts(
            work,
            project_id=project_id,
            tool_id=tool_id,
            forbidden_text=secret_context,
        )
        return _FakeProc(_tool_success_stream(tool_id))

    be = OpenCodeCliBackend(
        model="provider/model",
        opencode_path="opencode",
        timeout=41,
        tool_replay=True,
    )
    task = TaskRecord(
        id="opencode-tool-prompt",
        project=str(tmp_path),
        intent=intent,
        context_excerpt=secret_context,
    )
    with (
        mock.patch(
            "skillopt_sleep.backend.secrets.token_hex",
            side_effect=["a" * 64, "b" * 32, "c" * 32],
        ),
        mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run),
    ):
        response, called = be.attempt_with_tools(task, skill, memory, ["search"])

    assert response == "answer"
    assert called == ["search"]
    assert be.last_call_error == ""
    assert [call[0][1:3] for call in captured] == [
        ["debug", "config"],
        ["debug", "config"],
        ["debug", "agent"],
        ["run", "--pure"],
    ]
    run_call = captured[-1][1]
    assert run_call["env"]["OPENCODE_DISABLE_PROJECT_CONFIG"] == "0"
    assert run_call["env"]["OPENCODE_PURE"] == "1"
    assert run_call["env"]["OPENAI_API_KEY"] == "provider-secret-stays-in-env"
    assert "OPENCODE_DIRECT_TRACE" not in run_call["env"]
    for key in ("GIT_COMMON_DIR", "GIT_DIR", "GIT_OBJECT_DIRECTORY", "GIT_WORK_TREE"):
        assert not any(candidate.upper() == key for candidate in run_call["env"])
    for sentinel in (intent, secret_context, skill, memory):
        assert sentinel in run_call["input"]
    assert '{"query":"synthetic"}' in run_call["input"]
    assert "do not call a tool merely because it is listed" in run_call["input"]
    assert "Learned preferences" in run_call["input"]
    assert "override earlier conflicting skill text" in run_call["input"]
    assert "Call every listed random internal ID" not in run_call["input"]
    assert not os.path.exists(run_call["cwd"])


@pytest.mark.parametrize(
    "tools",
    [
        pytest.param(["../search"], id="invalid-prefix"),
        pytest.param([], id="empty"),
        pytest.param(["search", 123], id="non-string"),
        pytest.param(["x" * 129], id="too-long"),
        pytest.param([f"tool-{index}" for index in range(33)], id="too-many"),
    ],
)
def test_attempt_with_tools_rejects_invalid_names_before_starting_child(tools, tmp_path):
    be = OpenCodeCliBackend(opencode_path="opencode", tool_replay=True)
    task = TaskRecord(id="invalid-tools", project=str(tmp_path), intent="answer")
    with mock.patch("skillopt_sleep.backend.subprocess.run") as run:
        assert be.attempt_with_tools(task, "", "", tools) == ("", [])
    run.assert_not_called()
    assert "invalid tool list" in be.last_call_error


@pytest.mark.parametrize(
    ("child_result", "expected_fragment"),
    [
        pytest.param(
            _FakeProc(json.dumps({"tools": {"expected": False}})),
            "restrict tools",
            id="expected-disabled",
        ),
        pytest.param(
            _FakeProc(json.dumps({"tools": {"expected": True, "bash": True}})),
            "restrict tools",
            id="extra-enabled",
        ),
        pytest.param(
            _FakeProc(json.dumps({"tools": {"expected": "allow"}})),
            "invalid configuration",
            id="non-boolean-status",
        ),
        pytest.param(
            _FakeProc(json.dumps({"tools": []})),
            "invalid configuration",
            id="invalid-tools-shape",
        ),
        pytest.param(
            subprocess.TimeoutExpired(cmd=["opencode"], timeout=1),
            "timed out",
            id="timeout",
        ),
        pytest.param(OSError("private child detail"), "could not be completed", id="spawn-error"),
        pytest.param(_FakeProc("", returncode=9), "failed", id="nonzero-exit"),
        pytest.param(_FakeProc("not-json"), "invalid configuration", id="invalid-json"),
    ],
)
def test_tool_permission_verification_fails_closed(child_result, expected_fragment, tmp_path):
    backend = OpenCodeCliBackend(opencode_path=str(tmp_path / "opencode"), timeout=1)

    with (
        mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=[child_result]) as run,
        pytest.raises(OpenCodeError) as error,
    ):
        backend._verify_tool_allowlist({}, str(tmp_path), "skillopt-sleep-agent", {"expected"})

    run.assert_called_once()
    assert expected_fragment in str(error.value)
    assert "private child detail" not in str(error.value)


def test_tool_replay_fails_before_model_if_snapshots_remain_enabled():
    captured_work = ""

    def fake_run(cmd, **kwargs):
        nonlocal captured_work
        captured_work = kwargs["cwd"]
        assert cmd[1:3] == ["debug", "config"]
        snapshot = len(run.call_args_list) == 2
        return _FakeProc(_resolved_mcp(snapshot=snapshot))

    be = OpenCodeCliBackend(opencode_path="opencode", tool_replay=True)
    task = mock.Mock(intent="search", context_excerpt="")
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run) as run:
        assert be.attempt_with_tools(task, "", "", ["search"]) == ("", [])

    assert run.call_count == 2
    assert "disable session snapshots" in be.last_call_error
    assert captured_work and not os.path.exists(captured_work)


@pytest.mark.parametrize(
    ("run_result", "error_fragment"),
    [
        (subprocess.TimeoutExpired(cmd=["opencode"], timeout=1), "timed out"),
        (OSError("private run detail"), "could not be executed"),
        (_FakeProc("", returncode=9), "tool replay failed"),
    ],
)
def test_tool_replay_handles_model_child_failures(run_result, error_fragment):
    captured_work = ""

    def fake_run(cmd, **kwargs):
        nonlocal captured_work
        captured_work = kwargs["cwd"]
        if cmd[1:3] == ["debug", "config"]:
            return _FakeProc(_resolved_mcp(snapshot=False))
        if cmd[1:3] == ["debug", "agent"]:
            config = json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])
            permission = config["agent"][cmd[3]]["permission"]
            allowed = {name: True for name, value in permission.items() if value == "allow"}
            return _FakeProc(json.dumps({"tools": allowed}))
        if isinstance(run_result, BaseException):
            raise run_result
        return run_result

    be = OpenCodeCliBackend(
        opencode_path="opencode",
        timeout=1,
        tool_replay=True,
    )
    task = mock.Mock(intent="search", context_excerpt="private prompt detail")
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run) as run:
        assert be.attempt_with_tools(task, "", "", ["search"]) == ("", [])

    assert run.call_count == 4
    assert be.tokens_used() > 0
    assert error_fragment in be.last_call_error
    assert "private" not in be.last_call_error
    assert captured_work and not os.path.exists(captured_work)


def test_tool_replay_maps_and_deduplicates_only_completed_requested_tools(tmp_path):
    search_id = "skillopt_replay_" + "c" * 32
    lookup_id = "skillopt_replay_" + "d" * 32

    def fake_run(cmd, **kwargs):
        if cmd[1:3] == ["debug", "config"]:
            return _FakeProc(_resolved_mcp(snapshot=False))
        config = json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])
        if cmd[1:3] == ["debug", "agent"]:
            permission = config["agent"][cmd[3]]["permission"]
            assert permission == {"*": "deny", search_id: "allow", lookup_id: "allow"}
            return _FakeProc(json.dumps({"tools": {search_id: True, lookup_id: True}}))

        return _FakeProc(
            "\n".join(
                [
                    _event("step_start", part={"type": "step-start"}),
                    _tool_event(
                        lookup_id,
                        call_id="call-1",
                        output=_OPENCODE_SYNTHETIC_TOOL_RESULT,
                    ),
                    _tool_event(
                        lookup_id,
                        call_id="call-2",
                        output=_OPENCODE_SYNTHETIC_TOOL_RESULT,
                    ),
                    _event("step_finish", part={"type": "step-finish"}),
                    _event("step_start", part={"type": "step-start"}),
                    _event("text", part={"type": "text", "text": "answer"}),
                    _event("step_finish", part={"type": "step-finish"}),
                ]
            )
        )

    be = OpenCodeCliBackend(opencode_path="opencode", tool_replay=True)
    task = TaskRecord(id="partial-tool-replay", project=str(tmp_path), intent="use lookup")
    with (
        mock.patch(
            "skillopt_sleep.backend.secrets.token_hex",
            side_effect=["a" * 64, "b" * 32, "c" * 32, "d" * 32],
        ),
        mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run),
    ):
        response, called = be.attempt_with_tools(task, "", "", ["search", "lookup"])

    assert response == "answer"
    assert called == ["lookup"]


def test_tool_replay_does_not_treat_a_self_report_as_a_tool_call():
    def fake_run(cmd, **kwargs):
        if cmd[1:3] == ["debug", "config"]:
            return _FakeProc(_resolved_mcp(snapshot=False))
        if cmd[1:3] == ["debug", "agent"]:
            config = json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])
            permission = config["agent"][cmd[3]]["permission"]
            return _FakeProc(
                json.dumps({"tools": {name: value == "allow" for name, value in permission.items() if name != "*"}})
            )
        return _FakeProc(_success_stream("TOOL_CALL: search"))

    be = OpenCodeCliBackend(opencode_path="opencode", tool_replay=True)
    task = mock.Mock(intent="answer", context_excerpt="")
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
        assert be.attempt_with_tools(task, "", "", ["search"]) == (
            "TOOL_CALL: search",
            [],
        )
    assert be.last_call_error == ""


def test_get_and_build_backend_route_opencode_path():
    with mock.patch("shutil.which", return_value=None):
        for alias in ("opencode", "opencode_cli", "opencode-cli", "OPENCODE"):
            assert isinstance(get_backend(alias), OpenCodeCliBackend)

        single = build_backend(
            backend="opencode",
            opencode_path="custom-opencode",
            opencode_tool_replay=True,
        )
        assert isinstance(single, OpenCodeCliBackend)
        assert single.opencode_path == "custom-opencode"
        assert single.tool_replay is True

        stringly_enabled = build_backend(
            backend="opencode",
            opencode_path="custom-opencode",
            opencode_tool_replay="true",
        )
        assert stringly_enabled.tool_replay is False

        dual = build_backend(
            backend="mock",
            optimizer_backend="opencode",
            target_backend="opencode",
            opencode_path="custom-opencode",
            opencode_tool_replay=True,
        )
        assert isinstance(dual, DualBackend)
        assert dual.optimizer.opencode_path == "custom-opencode"
        assert dual.target.opencode_path == "custom-opencode"
        assert dual.optimizer.tool_replay is True
        assert dual.target.tool_replay is True


def test_cli_makes_relative_opencode_path_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    parser = argparse.ArgumentParser()
    _add_common(parser)
    relative = os.path.join("bin", "opencode")
    args = parser.parse_args(
        [
            "--backend",
            "opencode",
            "--opencode-path",
            relative,
            "--opencode-tool-replay",
        ]
    )
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)
    cfg = _cfg_from_args(args)
    assert cfg.get("backend") == "opencode"
    assert cfg.get("opencode_path") == os.path.abspath(relative)
    assert cfg.get("opencode_tool_replay") is True


def test_cli_without_tool_replay_flag_preserves_enabled_user_config(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    _add_common(parser)
    args = parser.parse_args(["--backend", "opencode"])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"opencode_tool_replay": True}), encoding="utf-8")
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: str(config_path))

    cfg = _cfg_from_args(args)

    assert cfg.get("opencode_tool_replay") is True


def test_cycle_diagnostic_build_forwards_opencode_settings():
    cfg = load_config(
        backend="opencode",
        opencode_path="custom-opencode",
        opencode_tool_replay=True,
    )
    with mock.patch("skillopt_sleep.cycle.build_backend", return_value=MockBackend()) as builder:
        cycle._make_model_key(cfg)
    builder.assert_called_once()
    assert builder.call_args.kwargs["opencode_path"] == "custom-opencode"
    assert builder.call_args.kwargs["opencode_tool_replay"] is True


@pytest.mark.parametrize("configured", [True, "true"])
def test_runtime_cycle_build_forwards_opencode_settings(tmp_path, configured):
    project = tmp_path / "project"
    project.mkdir()
    cfg = SleepConfig(
        data={
            **DEFAULTS,
            "backend": "opencode",
            "opencode_path": "custom-opencode",
            "opencode_tool_replay": configured,
            "projects": "invoked",
            "invoked_project": str(project),
            "state_dir": str(tmp_path / "state"),
            "claude_home": str(tmp_path / "claude-home"),
            "evidence_log": False,
        }
    )

    with mock.patch("skillopt_sleep.cycle.build_backend", return_value=MockBackend()) as builder:
        cycle.run_sleep_cycle(cfg, seed_tasks=[], dry_run=True)

    assert builder.call_count == 1
    assert builder.call_args.kwargs["opencode_path"] == "custom-opencode"
    assert builder.call_args.kwargs["opencode_tool_replay"] == configured


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(True, True), ("true", False)],
)
def test_cycle_artifacts_record_only_a_boolean_tool_replay_opt_in(
    tmp_path,
    configured,
    expected,
):
    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    cfg = SleepConfig(
        data={
            **DEFAULTS,
            "backend": "opencode",
            "opencode_tool_replay": configured,
            "projects": "invoked",
            "invoked_project": str(project),
            "state_dir": str(state_dir),
            "claude_home": str(tmp_path / "claude-home"),
            "evidence_log": True,
            "evolve_skill": False,
            "evolve_memory": False,
        }
    )
    task = TaskRecord(
        id="opencode-artifacts",
        project=str(project),
        intent="record the resolved replay setting",
        reference_kind="exact",
        reference="expected response",
        split="val",
    )

    with mock.patch("skillopt_sleep.cycle.build_backend", return_value=MockBackend()):
        outcome = cycle.run_sleep_cycle(cfg, seed_tasks=[task])

    staging_dir = Path(outcome.staging_dir)
    records = [json.loads(line) for line in (staging_dir / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    start = next(record for record in records if record["stage"] == "cycle" and record["event"] == "start")
    assert start["config"]["opencode_tool_replay"] is expected

    diagnostics_path = staging_dir / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["opencode_tool_replay"] is expected

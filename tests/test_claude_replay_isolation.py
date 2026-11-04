"""Offline command/report regressions; real CLI isolation is checked separately."""
import json
import os
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from skillopt_sleep.backend import ClaudeCliBackend, DualBackend, MockBackend
from skillopt_sleep.config import load_config
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.staging import StagingError
from skillopt_sleep.types import TaskRecord


def result(stdout="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def test_oauth_safe_mode_is_cached_across_both_replay_paths(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = ClaudeCliBackend(claude_path="/synthetic/claude", timeout=5)
    commands = []

    def cli(command, **kwargs):
        commands.append(command)
        return result("  --safe-mode  Disable customizations" if "--help" in command else "OK")

    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=cli):
        assert backend._call("first") == "OK"
        assert backend._call("second") == "OK"
        answer, _ = backend.attempt_with_tools(TaskRecord(id="t", project="synthetic", intent="test"), "", "", ["search"])
        assert answer == "OK"
    assert sum("--help" in command for command in commands) == 1
    replays = [command for command in commands if "-p" in command]
    assert len(replays) == 3
    assert all("--safe-mode" in command and "--bare" not in command for command in replays)


@pytest.mark.parametrize("help_result", [
    result("  --other  No safe mode here"),
    result("This version does not support --safe-mode"),
    result("  --safe-mode  Disable customizations", returncode=1),
    subprocess.TimeoutExpired("claude --help", 5),
    FileNotFoundError("missing CLI"),
])
def test_unsupported_or_failed_probe_never_runs_an_unisolated_replay(monkeypatch, help_result):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = ClaudeCliBackend(claude_path="/synthetic/claude", timeout=5)
    with mock.patch("skillopt_sleep.backend.subprocess.run") as run:
        if isinstance(help_result, Exception):
            run.side_effect = help_result
        else:
            run.return_value = help_result
        with pytest.raises(StagingError):
            backend._call("first")
        with pytest.raises(StagingError):
            backend.attempt_with_tools(TaskRecord(id="t", project="synthetic", intent="test"), "", "", ["search"])
    assert run.call_count == 1
    assert run.call_args.args[0] == [backend.claude_path, "--help"]


def test_api_key_uses_bare_without_a_capability_probe(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-unit-test-key")
    backend = ClaudeCliBackend(claude_path="/synthetic/claude", timeout=5)
    with mock.patch("skillopt_sleep.backend.subprocess.run", return_value=result("OK")) as run:
        assert backend._call("first") == "OK"
        backend.attempt_with_tools(TaskRecord(id="t", project="synthetic", intent="test"), "", "", ["search"])
    assert run.call_count == 2
    assert all("--bare" in call.args[0] and "--safe-mode" not in call.args[0]
               for call in run.call_args_list)


def test_switching_auth_selects_each_isolated_mode_without_reprobing(monkeypatch):
    backend = ClaudeCliBackend(claude_path="/synthetic/claude", timeout=5)
    commands = []

    def cli(command, **kwargs):
        commands.append(command)
        return result("  --safe-mode  Disable customizations" if "--help" in command else "OK")

    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=cli):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-unit-test-key")
        backend._call("api key")
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        backend._call("oauth")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-unit-test-key")
        backend._call("api key again")
    assert sum("--help" in command for command in commands) == 1
    assert ["--bare" in command for command in commands if "-p" in command] == [True, False, True]


@pytest.mark.parametrize("split", [False, True])
def test_night_records_the_actual_isolation_mode_in_diagnostics_and_report(tmp_path, monkeypatch, split):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    cfg = load_config(invoked_project=str(project), projects="invoked", backend="claude",
                      claude_home=str(tmp_path / "synthetic-home"), auto_adopt=False,
                      multi_skill_report=False)
    claude = ClaudeCliBackend(claude_path="/synthetic/claude", timeout=5)
    backend = DualBackend(target=claude, optimizer=MockBackend()) if split else claude
    tasks = [TaskRecord(id=str(i), project=str(project), intent="Reply OK", reference="OK",
                        reference_kind="exact", split="train" if i < 2 else "val") for i in range(4)]

    def cli(command, **kwargs):
        return result("  --safe-mode  Disable customizations" if "--help" in command else "OK")

    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=cli):
        outcome = run_sleep_cycle(cfg, seed_tasks=tasks, backend=backend)
    with open(os.path.join(outcome.staging_dir, "diagnostics.json")) as handle:
        diagnostics = json.load(handle)
    assert diagnostics["claude_isolation_modes"] == {"target" if split else "backend": ["safe-mode"]}
    for name in ["report.json", "report.md"]:
        with open(os.path.join(outcome.staging_dir, name)) as handle:
            assert "--safe-mode" in handle.read()

    # The same tasks are already consumed: no new calls means no new mode evidence.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-unit-test-key")
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=cli):
        second = run_sleep_cycle(cfg, seed_tasks=tasks, backend=backend)
    with open(os.path.join(second.staging_dir, "diagnostics.json")) as handle:
        assert json.load(handle)["claude_isolation_modes"] == {}

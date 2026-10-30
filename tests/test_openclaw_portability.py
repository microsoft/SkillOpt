"""Portability regressions for the OpenClaw reference adapter."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "plugins" / "openclaw"

sys.path.insert(0, str(PLUGIN))
import openclaw_paths  # noqa: E402


def _set_openclaw_env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    for key in ("OPENCLAW_HOME", "OPENCLAW_STATE_DIR", "OPENCLAW_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _run_python(script: str, cwd: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _usable_bash() -> str | None:
    candidates = [
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        probe = subprocess.run(
            [candidate, "-lc", "printf ok"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if probe.returncode == 0 and probe.stdout == "ok":
            return candidate
    return None


def _run_cron(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    bash = _usable_bash()
    if not bash:
        pytest.skip("bash is not available")
    return subprocess.run(
        [bash, str(PLUGIN / "run_sleep_cron.sh"), "missing-category"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_state_directory_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    explicit_state = tmp_path / "explicit-state"
    cases = [
        ({"OPENCLAW_HOME": str(home)}, home / ".openclaw"),
        ({"HOME": str(home)}, home / ".openclaw"),
        ({"OPENCLAW_HOME": str(home), "OPENCLAW_STATE_DIR": str(explicit_state)}, explicit_state),
        ({"OPENCLAW_HOME": str(home), "OPENCLAW_PROFILE": "dev"}, home / ".openclaw-dev"),
        ({"OPENCLAW_HOME": str(home), "OPENCLAW_PROFILE": "default"}, home / ".openclaw"),
        ({"OPENCLAW_HOME": str(home), "OPENCLAW_PROFILE": "../../escape"}, home / ".openclaw"),
    ]
    for env, expected in cases:
        _set_openclaw_env(monkeypatch, **env)
        assert openclaw_paths.openclaw_state_dir() == expected


def test_home_tilde_and_blank_path_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    _set_openclaw_env(monkeypatch, HOME=str(home), OPENCLAW_HOME="~/relocated")
    assert openclaw_paths.openclaw_state_dir() == home / "relocated" / ".openclaw"

    explicit_state = tmp_path / "state"
    _set_openclaw_env(monkeypatch, OPENCLAW_STATE_DIR=str(explicit_state))
    resolved = openclaw_paths.apply_openclaw_defaults(
        {"claude_home": "", "invoked_project": ""}
    )
    assert resolved == {
        "claude_home": str(explicit_state / "agents"),
        "invoked_project": str(explicit_state / "workspace"),
    }

    explicit = openclaw_paths.apply_openclaw_defaults(
        {"claude_home": "/custom/agents", "invoked_project": "/custom/workspace"}
    )
    assert explicit["claude_home"] == "/custom/agents"
    assert explicit["invoked_project"] == "/custom/workspace"


def test_symlink_home_path_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    linked_home = tmp_path / "linked-home"
    try:
        linked_home.symlink_to(real_home, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    _set_openclaw_env(monkeypatch, OPENCLAW_HOME=str(linked_home))
    assert openclaw_paths.openclaw_state_dir() == Path(
        os.path.abspath(linked_home)
    ) / ".openclaw"


def test_plugin_has_no_machine_specific_home_path() -> None:
    offenders = [
        str(path.relative_to(REPO))
        for path in PLUGIN.rglob("*")
        if path.is_file() and "ethanclaw" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


def test_sample_config_is_runtime_resolved() -> None:
    config = json.loads((PLUGIN / "config.json").read_text(encoding="utf-8"))
    assert config["claude_home"] == ""
    assert config["invoked_project"] == ""


def test_runner_uses_bundled_config_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_openclaw_env(monkeypatch, OPENCLAW_HOME=str(tmp_path))
    script = f"""
import runpy
import sys
sys.path.insert(0, {str(PLUGIN)!r})
runner = runpy.run_path({str(PLUGIN / "run_sleep.py")!r}, run_name="fallback_test")
print(runner["PLUGIN_DIR"])
print(runner["REPO"])
print(runner["DEFAULT_CONFIG"])
"""
    result = _run_python(script, tmp_path, OPENCLAW_HOME=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert [Path(line) for line in result.stdout.splitlines()] == [
        PLUGIN,
        REPO,
        PLUGIN / "config.json",
    ]


def test_runner_prefers_installed_openclaw_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_openclaw_env(monkeypatch, OPENCLAW_HOME=str(tmp_path))
    installed = (
        tmp_path
        / ".openclaw"
        / "workspace"
        / "skills"
        / "skillopt-sleep"
        / "config.json"
    )
    installed.parent.mkdir(parents=True)
    installed.write_text("{}", encoding="utf-8")
    script = f"""
import runpy
import sys
sys.path.insert(0, {str(PLUGIN)!r})
runner = runpy.run_path({str(PLUGIN / "run_sleep.py")!r}, run_name="installed_test")
print(runner["DEFAULT_CONFIG"])
"""
    result = _run_python(script, tmp_path, OPENCLAW_HOME=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == installed


def test_slash_helper_derives_plugin_directory(tmp_path: Path) -> None:
    script = f"""
import runpy
import sys
sys.path.insert(0, {str(PLUGIN)!r})
print(runpy.run_path({str(PLUGIN / "slash_sleep.py")!r}, run_name="slash_test")["SKILL_DIR"])
"""
    result = _run_python(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == PLUGIN


def test_cron_wrapper_uses_openclaw_state_and_profile(tmp_path: Path) -> None:
    cases = [
        {
            "home": tmp_path / "explicit-home",
            "openclaw_home": tmp_path / "openclaw-home",
            "profile": None,
            "state": tmp_path / "explicit-state",
            "override": True,
        },
        {
            "home": tmp_path / "tilde-home",
            "openclaw_home": "~/relocated",
            "profile": "dev",
            "state": tmp_path / "tilde-home" / "relocated" / ".openclaw-dev",
            "override": False,
        },
        {
            "home": tmp_path / "invalid-profile-home",
            "openclaw_home": tmp_path / "invalid-profile-home",
            "profile": "../../escape",
            "state": tmp_path / "invalid-profile-home" / ".openclaw",
            "override": False,
        },
    ]
    for case in cases:
        case["state"].mkdir(parents=True)
        (case["state"] / ".env").write_text(
            "DEEPSEEK_API_KEY=test-only\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.update(
            {
                "HOME": case["home"].as_posix(),
                "OPENCLAW_HOME": str(case["openclaw_home"]).replace("\\", "/"),
            }
        )
        if case["profile"]:
            env["OPENCLAW_PROFILE"] = case["profile"]
        else:
            env.pop("OPENCLAW_PROFILE", None)
        if case["override"]:
            env["OPENCLAW_STATE_DIR"] = case["state"].as_posix()
        else:
            env.pop("OPENCLAW_STATE_DIR", None)
        result = _run_cron(env)
        assert result.returncode == 0, result.stderr
        assert "SKIP: missing-category" in result.stdout

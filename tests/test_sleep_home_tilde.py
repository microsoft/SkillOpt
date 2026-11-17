"""Configured Claude/Codex homes may start with ``~`` (#394)."""
import argparse
import json
import os

from skillopt_sleep.__main__ import _add_common, _cfg_from_args
from skillopt_sleep.config import CLAUDE_HOME, HOME_STATE_DIR, load_config


def _isolate_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)
    return str(home)


def test_config_file_homes_expand_tilde(monkeypatch, tmp_path):
    home = _isolate_home(monkeypatch, tmp_path)
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"claude_home": "~/.claude-work", "codex_home": "~/.codex-work"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "skillopt_sleep.config._user_config_path", lambda: str(config_file)
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_config()

    claude = os.path.join(home, ".claude-work")
    assert cfg.transcripts_dir == os.path.join(claude, "projects")
    assert cfg.history_path == os.path.join(claude, "history.jsonl")
    assert cfg.managed_skill_path() == os.path.join(
        claude, "skills", "skillopt-sleep-learned", "SKILL.md"
    )
    assert cfg.state_dir == os.path.join(home, ".skillopt-sleep")
    assert cfg.codex_archived_sessions_dir == os.path.join(
        home, ".codex-work", "archived_sessions"
    )


def test_tilde_default_claude_home_keeps_default_state_dir(monkeypatch):
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)
    cfg = load_config(claude_home="~/.claude")
    assert os.path.expanduser("~/.claude") == CLAUDE_HOME
    assert cfg.state_dir == HOME_STATE_DIR


def test_cli_homes_expand_tilde(monkeypatch, tmp_path):
    home = _isolate_home(monkeypatch, tmp_path)
    parser = argparse.ArgumentParser()
    _add_common(parser)
    args = parser.parse_args(
        ["--claude-home", "~/.claude-work", "--codex-home", "~/.codex-work"]
    )

    cfg = _cfg_from_args(args)

    assert cfg.get("claude_home") == os.path.join(home, ".claude-work")
    assert cfg.get("codex_home") == os.path.join(home, ".codex-work")

"""Tests for harvesting Codex CLI rollout sessions (sessions/YYYY/MM/DD)."""
from __future__ import annotations

import json

from skillopt_sleep.harvest_codex import digest_codex_archived_session, harvest_codex


def _rec(ts, rtype, payload):
    return {"timestamp": ts, "type": rtype, "payload": payload}


def _write_rollout(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(path)


def _cli_rollout_records():
    return [
        _rec("2026-07-18T10-11-28.000Z", "session_meta", {
            "session_id": "abc", "cwd": "/home/u/proj",
        }),
        _rec("2026-07-18T10-11-29.000Z", "response_item", {
            "type": "message", "role": "developer",
            "content": [{"type": "input_text", "text": "developer instructions"}],
        }),
        _rec("2026-07-18T10-11-30.000Z", "response_item", {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text":
                "<environment_context>\n  <cwd>/home/u/proj</cwd>\n</environment_context>"}],
        }),
        _rec("2026-07-18T10-11-31.000Z", "response_item", {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "fix the failing tests"}],
        }),
        # CLI rollouts repeat each user turn as event_msg/user_message.
        _rec("2026-07-18T10-11-32.000Z", "event_msg", {
            "type": "user_message", "message": "fix the failing tests",
        }),
        # Assistant turn: event_msg first, response_item second.
        _rec("2026-07-18T10-11-33.000Z", "event_msg", {
            "type": "agent_message", "message": "Running the suite now.",
        }),
        _rec("2026-07-18T10-11-34.000Z", "response_item", {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "Running the suite now."}],
        }),
        _rec("2026-07-18T10-12-00.000Z", "response_item", {
            "type": "exec_command_end",
        }),
        _rec("2026-07-18T10-12-30.000Z", "response_item", {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "wrong flag, try again"}],
        }),
        _rec("2026-07-18T10-12-31.000Z", "event_msg", {
            "type": "user_message", "message": "wrong flag, try again",
        }),
    ]


def test_harvest_finds_cli_rollout_sessions(tmp_path):
    codex_home = tmp_path / "codex"
    rollout = codex_home / "sessions" / "2026" / "07" / "18" / (
        "rollout-2026-07-18T10-11-28-abc.jsonl"
    )
    _write_rollout(rollout, _cli_rollout_records())

    digests = harvest_codex(str(codex_home / "archived_sessions"), scope="all")
    assert len(digests) == 1
    d = digests[0]
    assert d.project == "/home/u/proj"
    # duplicated turns counted once; environment_context dropped
    assert d.n_user_turns == 2
    assert d.user_prompts == ["fix the failing tests", "wrong flag, try again"]
    assert d.tools_used == ["exec_command"]
    assert d.started_at.startswith("2026-07-18T10-11-28")
    assert d.ended_at.startswith("2026-07-18T10-12-31")


def test_harvest_combines_desktop_and_cli_layouts(tmp_path):
    codex_home = tmp_path / "codex"
    _write_rollout(
        codex_home / "sessions" / "2026" / "07" / "18" / "rollout-cli.jsonl",
        _cli_rollout_records(),
    )
    archived = codex_home / "archived_sessions"
    _write_rollout(archived / "desktop-session.jsonl", [
        _rec("2026-06-01T09:00:00Z", "session", {"cwd": "/home/u/other"}),
        _rec("2026-06-01T09:00:01Z", "payload", {
            "type": "user_message", "message": "desktop prompt",
        }),
    ])

    digests = harvest_codex(str(archived), scope="all")
    assert {d.project for d in digests} == {"/home/u/proj", "/home/u/other"}


def test_digest_returns_none_without_messages(tmp_path):
    path = _write_rollout(
        tmp_path / "rollout-empty.jsonl",
        [_rec("2026-07-18T10-11-28.000Z", "session_meta",
              {"session_id": "abc", "cwd": "/home/u/proj"})],
    )
    assert digest_codex_archived_session(path) is None

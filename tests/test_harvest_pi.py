"""Tests for the pi (pi-coding-agent) transcript harvester."""
from __future__ import annotations

import json

from skillopt_sleep.harvest_pi import _redact_secrets, digest_pi_session, harvest_pi


def _write_session(tmp_path, slug, name, entries):
    d = tmp_path / slug
    d.mkdir(parents=True)
    p = d / f"{name}.jsonl"
    with open(p, "w") as f:
        for rec in entries:
            f.write(json.dumps(rec) + "\n")
    return str(p)


PI_SESSION = [
    {"type": "session", "version": 1, "id": "s1", "timestamp": "2026-06-23T11:52:04.333Z", "cwd": "/home/u/proj"},
    {"type": "model_change", "id": "m", "timestamp": "2026-06-23T11:52:05.000Z", "modelId": "gpt-x"},
    {"type": "message", "id": "a1", "parentId": "s1", "timestamp": "2026-06-23T11:52:06.000Z",
     "message": {"role": "user", "content": [{"type": "text", "text": "fix the failing tests"}]}},
    {"type": "message", "id": "a2", "parentId": "a1", "timestamp": "2026-06-23T11:52:07.000Z",
     "message": {"role": "assistant", "content": [
         {"type": "thinking", "thinking": "private reasoning"},
         {"type": "text", "text": "Running the suite now."},
         {"type": "toolCall", "id": "call_1", "name": "bash", "arguments": {"command": "pytest"}},
     ]}},
    {"type": "message", "id": "a3", "parentId": "a2", "timestamp": "2026-06-23T11:52:08.000Z",
     "message": {"role": "toolResult", "toolCallId": "call_1", "toolName": "bash",
                 "content": [{"type": "text", "text": "1 failed"}], "isError": True}},
    {"type": "message", "id": "a4", "parentId": "a3", "timestamp": "2026-06-23T11:52:30.000Z",
     "message": {"role": "user", "content": "thanks, that works now"}},
    {"type": "message", "id": "a5", "parentId": "a4", "timestamp": "2026-06-23T11:52:31.000Z",
     "message": {"role": "assistant", "content": [{"type": "text", "text": "Glad it's fixed."}]}},
]


def test_digest_extracts_fields():
    d = digest_pi_session("/tmp/abc-123.jsonl")  # missing file -> None
    assert d is None


def test_digest_full_session(tmp_path):
    p = _write_session(tmp_path, "--home-u-proj", "abc-123", PI_SESSION)
    d = digest_pi_session(p)
    assert d is not None
    assert d.project == "/home/u/proj"
    assert d.n_user_turns == 2
    assert d.n_assistant_turns == 2
    assert "bash" in d.tools_used                 # from toolCall block
    assert any("works" in f for f in d.feedback_signals)   # pos feedback
    assert all(
        not f.startswith("neg:tool_error") for f in d.feedback_signals
    )  # isError deliberately NOT surfaced (recovered errors ≠ task failure)
    assert all("private reasoning" not in f for f in d.feedback_signals)
    # thinking blocks must not leak into finals
    assert "private reasoning" not in " ".join(d.assistant_finals)
    assert d.started_at.startswith("2026-06-23T11:52:04")
    assert d.ended_at.startswith("2026-06-23T11:52:31")


def test_harvest_scope_filter(tmp_path):
    _write_session(tmp_path, "--home-u-proj", "abc-123", PI_SESSION)
    other = list(PI_SESSION)
    other[0] = dict(PI_SESSION[0], cwd="/other/place")
    _write_session(tmp_path, "--other", "xyz", other)

    all_scope = harvest_pi(str(tmp_path), scope="all")
    assert len(all_scope) == 2
    invoked = harvest_pi(str(tmp_path), scope="invoked", invoked_project="/home/u/proj")
    assert len(invoked) == 1
    assert invoked[0].project == "/home/u/proj"


def test_secret_redaction():
    out = _redact_secrets("Authorization: Bearer sk-1234567890abcdefghij")
    assert "sk-1234567890abcdefghij" not in out
    assert "[REDACTED]" in out

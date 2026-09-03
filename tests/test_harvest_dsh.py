"""Coverage for the read-only DeepSeek Harness transcript harvester."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import pytest

from skillopt_sleep.__main__ import _add_common, _cfg_from_args
from skillopt_sleep.config import load_config
from skillopt_sleep.harvest_dsh import (
    DSH_REPLAY_SENTINEL,
    _encode_segment,
    _project_key,
    digest_dsh_session,
    harvest_dsh,
)
from skillopt_sleep.harvest_sources import harvest_for_config
from skillopt_sleep.types import SessionDigest

_BASE_TIME = 1_800_000_000_000


def _header(session_id: str, cwd: str | None, **extra):
    value = {
        "type": "session",
        "version": 0,
        "id": session_id,
        "createdAt": _BASE_TIME,
        "delegationDepth": 0,
    }
    if cwd is not None:
        value["cwd"] = cwd
    value.update(extra)
    return value


def _user(seq: int, text: str, *, source="user", append=True):
    event = {
        "type": "user/message",
        "seq": seq,
        "time": _BASE_TIME + 1000 * (seq + 1),
        "data": {
            "id": f"user-{seq}",
            "role": "user",
            "content": [{"type": "text", "text": text}],
            "source": {"kind": source},
        },
    }
    if append:
        event["surfaceOp"] = "append"
    return event


def _assistant(seq: int, text: str, *, tool_name="", replace=False):
    content = [
        {"type": "reasoning", "text": "private chain of thought"},
        {"type": "text", "text": text},
    ]
    if tool_name:
        content.append({"type": "tool-call", "id": f"call-{seq}", "name": tool_name, "arguments": '{"secret":true}'})
    event = {
        "type": "assistant/message",
        "seq": seq,
        "time": _BASE_TIME + 1000 * (seq + 1),
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": f"assistant-{seq}",
                "role": "assistant",
                "content": content,
                "source": {"kind": "model", "provider": "test", "model": "test-model"},
            },
        },
        "surfaceOp": {"op": "replace", "start": 0, "end": 0} if replace else "append",
    }
    return event


def _tool_call(seq: int, name: str):
    return {
        "type": "tool/call",
        "seq": seq,
        "time": _BASE_TIME + 1000 * (seq + 1),
        "data": {"turn": 1, "step": 1, "callId": f"call-{seq}", "name": name, "arguments": '{"api_key":"secret"}'},
    }


def _feedback(seq: int, text: str):
    return {
        "type": "feedback/record",
        "seq": seq,
        "time": _BASE_TIME + 1000 * (seq + 1),
        "data": {"text": text},
    }


def _metadata(seq: int, event_type: str):
    return {
        "type": event_type,
        "seq": seq,
        "time": _BASE_TIME + 1000 * (seq + 1),
        "data": {},
    }


def _write_raw(root: Path, session_id: str, cwd: str | None, records: list[dict], **header_extra) -> Path:
    project_dir = "_no-cwd" if cwd is None else _project_key(cwd)
    path = root / project_dir / _encode_segment(session_id) / "session.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_header(session_id, cwd, **header_extra), *records]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_digest_extracts_only_safe_dsh_fields(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    path = _write_raw(
        tmp_path,
        "session-1",
        project,
        [
            _user(0, "Fix the tests. Authorization: Bearer sk-1234567890abcdefghij"),
            _assistant(1, "I fixed it.", tool_name="shell/run <unsafe>"),
            _tool_call(2, "shell/run <unsafe>"),
            _feedback(3, "Perfect, that works now. token=super-secret"),
        ],
    )

    digest = digest_dsh_session(str(path), root=str(tmp_path))

    assert digest is not None
    assert digest.project == project
    assert digest.user_prompts and "sk-1234567890abcdefghij" not in digest.user_prompts[0]
    assert digest.assistant_finals == ["I fixed it."]
    assert digest.tools_used == ["shell_run_unsafe_"]
    assert digest.n_user_turns == 1
    assert digest.n_assistant_turns == 1
    assert any(signal.startswith("pos:") for signal in digest.feedback_signals)
    persisted = json.dumps(digest.to_dict())
    assert "private chain of thought" not in persisted
    assert '"api_key"' not in persisted
    assert "super-secret" not in persisted


def test_packed_rows_are_validated_and_do_not_leak_chunks(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    packed = {
        "type": "text-chunks",
        "seq0": 1,
        "time0": _BASE_TIME + 2000,
        "data": {
            "turn": 1,
            "step": 1,
            "index": 0,
            "dt": [0, 7, 9],
            "texts": ["private", " streamed", " output"],
        },
    }
    path = _write_raw(tmp_path, "packed", project, [_user(0, "request"), packed, _assistant(4, "final")])

    digest = digest_dsh_session(str(path), root=str(tmp_path))

    assert digest is not None
    assert digest.user_prompts == ["request"]
    assert digest.assistant_finals == ["final"]
    assert "private streamed output" not in json.dumps(digest.to_dict())


def test_malformed_packed_row_rejects_the_whole_session(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    path = _write_raw(
        tmp_path,
        "bad-packed",
        project,
        [
            _user(0, "request"),
            {
                "type": "text-chunks",
                "seq0": 1,
                "time0": _BASE_TIME + 2000,
                "data": {"turn": 1, "step": 1, "index": 0, "dt": [0], "texts": ["a", "b", "c"]},
            },
        ],
    )

    assert digest_dsh_session(str(path), root=str(tmp_path)) is None


def test_fork_is_retained_but_subagent_and_replay_are_excluded(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    _write_raw(
        tmp_path,
        "fork",
        project,
        [_user(0, "inherited request"), _assistant(1, "superseded", replace=True), _assistant(2, "active final")],
        parentSession="parent",
        seedLength=1,
    )
    _write_raw(
        tmp_path,
        "subagent",
        project,
        [_user(0, "machine task"), _assistant(1, "machine final")],
        origin="subagent",
        delegationDepth=1,
    )
    _write_raw(
        tmp_path,
        "replay",
        project,
        [_user(0, DSH_REPLAY_SENTINEL + "\nrun internal task"), _assistant(1, "internal")],
    )

    digests = harvest_dsh(str(tmp_path), scope="all")

    assert [digest.session_id for digest in digests] == ["fork"]
    assert digests[0].assistant_finals == ["active final"]


def test_bad_session_is_silent_and_does_not_block_other_sessions(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    _write_raw(tmp_path, "good", project, [_user(0, "good request"), _assistant(1, "good final")])
    bad = tmp_path / _project_key(project) / _encode_segment("bad") / "session.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_text('{"type":"session"}\nnot-json\n', encoding="utf-8")

    digests = harvest_dsh(str(tmp_path), scope="all")

    assert [digest.session_id for digest in digests] == ["good"]


def test_unknown_required_event_rejects_but_ignorable_event_is_skipped(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    _write_raw(
        tmp_path,
        "ignorable",
        project,
        [
            _user(0, "request"),
            {"type": "plugin/info", "seq": 1, "time": _BASE_TIME + 2000, "data": {}, "ignorable": True},
            _assistant(2, "final"),
        ],
    )
    _write_raw(
        tmp_path,
        "required",
        project,
        [
            _user(0, "request"),
            {"type": "plugin/required", "seq": 1, "time": _BASE_TIME + 2000, "data": {}},
            _assistant(2, "final"),
        ],
    )

    assert [digest.session_id for digest in harvest_dsh(str(tmp_path), scope="all")] == ["ignorable"]


def test_current_dsh_lifecycle_metadata_is_accepted_without_retention(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    metadata_types = [
        "permission/preset",
        "sandbox/mode",
        "approval/policy",
        "agent/inbox/spliced",
        "session/title",
        "session/title-llm-request",
        "web/deepseek-search-llm-request",
        "approval/asked",
        "approval/decided",
    ]
    records = [_metadata(index, event_type) for index, event_type in enumerate(metadata_types)]
    records.extend([_user(len(records), "actual user request"), _assistant(len(records) + 1, "actual final")])

    path = _write_raw(tmp_path, "metadata", project, records)
    digest = digest_dsh_session(str(path), root=str(tmp_path))

    assert digest is not None
    assert digest.user_prompts == ["actual user request"]
    assert digest.assistant_finals == ["actual final"]


def test_scope_since_limit_and_identity_checks(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    other = str((tmp_path / "other").resolve())
    _write_raw(tmp_path, "one", project, [_user(0, "one"), _assistant(1, "one")])
    _write_raw(tmp_path, "two", other, [_user(0, "two"), _assistant(1, "two")])
    wrong = tmp_path / _project_key(project) / "not-the-id" / "session.jsonl"
    wrong.parent.mkdir(parents=True)
    wrong.write_text(json.dumps(_header("wrong", project)) + "\n", encoding="utf-8")

    invoked = harvest_dsh(str(tmp_path), scope="invoked", invoked_project=project, limit=1)
    assert [digest.session_id for digest in invoked] == ["one"]
    assert harvest_dsh(str(tmp_path), scope="all", since_iso="2030-01-01T00:00:00Z") == []


def test_zstd_concatenated_frames_are_read(tmp_path: Path):
    zstd = pytest.importorskip("zstandard")
    project = str((tmp_path / "repo").resolve())
    path = tmp_path / _project_key(project) / _encode_segment("compressed") / "session.jsonl.zstd"
    path.parent.mkdir(parents=True)
    compressor = zstd.ZstdCompressor(write_checksum=True)
    header = json.dumps(_header("compressed", project)).encode() + b"\n"
    events = b"".join(
        json.dumps(row).encode() + b"\n"
        for row in [_user(0, "compressed request"), _assistant(1, "compressed final")]
    )
    path.write_bytes(compressor.compress(header) + compressor.compress(events))

    digest = digest_dsh_session(str(path), root=str(tmp_path))

    assert digest is not None
    assert digest.user_prompts == ["compressed request"]
    assert digest.assistant_finals == ["compressed final"]


def test_cli_config_and_source_dispatch_for_dsh(monkeypatch, tmp_path: Path):
    parser = argparse.ArgumentParser()
    _add_common(parser)
    args = parser.parse_args(["--source", "dsh", "--dsh-session-root", "~/dsh-sessions"])
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)

    cfg = _cfg_from_args(args)
    expected_root = str(Path("~/dsh-sessions").expanduser().resolve())
    assert cfg.get("transcript_source") == "dsh"
    assert cfg.dsh_session_root == expected_root

    project = str((tmp_path / "repo").resolve())
    configured = load_config(transcript_source="dsh", dsh_session_root=str(tmp_path), invoked_project=project)
    expected = [SessionDigest(session_id="dsh", project=project)]
    with (
        mock.patch("skillopt_sleep.harvest_sources.harvest_dsh", return_value=expected) as dsh,
        mock.patch("skillopt_sleep.harvest_sources.harvest") as claude,
        mock.patch("skillopt_sleep.harvest_sources.harvest_codex") as codex,
    ):
        assert harvest_for_config(configured, since_iso="2026-01-01T00:00:00Z", limit=2) == expected
    dsh.assert_called_once_with(
        configured.dsh_session_root,
        scope="invoked",
        invoked_project=project,
        since_iso="2026-01-01T00:00:00Z",
        limit=2,
    )
    claude.assert_not_called()
    codex.assert_not_called()


def test_dsh_uses_the_standard_home_sessions_directory(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)
    dsh_home = tmp_path / "dsh-home"
    monkeypatch.setenv("DSH_HOME", str(dsh_home))

    cfg = load_config(transcript_source="dsh")

    assert cfg.dsh_session_root == str((dsh_home / "sessions").resolve())


def test_auto_source_does_not_add_dsh_precedence(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    cfg = load_config(transcript_source="auto", invoked_project=project, dsh_session_root=str(tmp_path))
    expected = [SessionDigest(session_id="claude", project=project)]
    with (
        mock.patch("skillopt_sleep.harvest_sources.harvest_codex", return_value=[]),
        mock.patch("skillopt_sleep.harvest_sources.harvest", return_value=expected),
        mock.patch("skillopt_sleep.harvest_sources.harvest_dsh") as dsh,
    ):
        assert harvest_for_config(cfg) == expected
    dsh.assert_not_called()

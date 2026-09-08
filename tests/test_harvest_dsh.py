"""Coverage for the read-only DeepSeek Harness transcript harvester."""
from __future__ import annotations

import argparse
import json
import logging
import weakref
from datetime import datetime, timedelta, timezone
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
_MASTER_V2_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "dsh"


def _header(session_id: str, cwd: str | None, *, version=0, **extra):
    value = {
        "type": "session",
        "version": version,
        "id": session_id,
        "createdAt": _BASE_TIME,
        "delegationDepth": 0,
    }
    if version >= 2:
        # V2 makes the fork-lineage bit explicit.  Keep it out of legacy
        # headers, where the historical codec used seedLength instead.
        value["isSeeded"] = False
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


def _replace_surface(event: dict, start: int, end: int, *source_event_seqs: int) -> dict:
    event["surfaceOp"] = {"op": "replace", "start": start, "end": end}
    if source_event_seqs:
        event["sourceEventSeqs"] = list(source_event_seqs)
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


def _end_seed(seq: int, *, inherited=True):
    return {
        "type": "session/end-seed",
        "seq": seq,
        "time": _BASE_TIME + 1000 * (seq + 1),
        "data": {"inherited": inherited} if inherited else {},
    }


def _write_raw(
    root: Path,
    session_id: str,
    cwd: str | None,
    records: list[dict],
    *,
    version=0,
    **header_extra,
) -> Path:
    project_dir = "_no-cwd" if cwd is None else _project_key(cwd)
    filename = "session.jsonl" if version == 0 else f"session.v{version}.jsonl"
    path = root / project_dir / _encode_segment(session_id) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_header(session_id, cwd, version=version, **header_extra), *records]
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
        [_user(0, "inherited request"), _end_seed(1), _assistant(2, "active final")],
        version=2,
        parentSession="parent",
        isSeeded=True,
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


def test_surface_replacement_removes_shadowed_assistant_text(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    compacted = _replace_surface(
        _user(2, "compacted context", source="system", append=False),
        0,
        1,
        0,
        1,
    )
    _write_raw(
        tmp_path,
        "compacted",
        project,
        [
            _user(0, "request"),
            _assistant(1, "obsolete assistant text"),
            compacted,
            _assistant(3, "current assistant text"),
        ],
    )

    digest = harvest_dsh(str(tmp_path), scope="all")[0]

    assert digest.assistant_finals == ["current assistant text"]
    assert "obsolete assistant text" not in json.dumps(digest.to_dict())


def test_v2_range_encoded_replacement_provenance_is_replayed(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    tool_result = {**_metadata(2, "tool/result"), "surfaceOp": "append"}
    tool_result["data"] = {"output": "private tool output"}
    replacement = _replace_surface(
        _user(4, "compacted context", source="system", append=False), 1, 3, 1, 2, 3,
    )
    # This is the physical v2 representation written by the upstream codec:
    # a run of three adjacent source sequence numbers becomes [start, end].
    replacement["sourceEventSeqs"] = [[1, 3]]
    _write_raw(
        tmp_path,
        "range-provenance",
        project,
        [
            _user(0, "request"),
            _assistant(1, "obsolete assistant text"),
            tool_result,
            _user(3, "injected context", source="plugin"),
            replacement,
            _assistant(5, "current assistant text"),
        ],
        version=2,
    )

    digest = harvest_dsh(str(tmp_path), scope="all")[0]

    assert digest.user_prompts == ["request"]
    assert digest.assistant_finals == ["current assistant text"]
    assert "obsolete assistant text" not in json.dumps(digest.to_dict())


def test_bad_session_is_silent_and_does_not_block_other_sessions(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    _write_raw(tmp_path, "good", project, [_user(0, "good request"), _assistant(1, "good final")])
    bad = tmp_path / _project_key(project) / _encode_segment("bad") / "session.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_text('{"type":"session"}\nnot-json\n', encoding="utf-8")

    digests = harvest_dsh(str(tmp_path), scope="all")

    assert [digest.session_id for digest in digests] == ["good"]


def test_v2_only_session_is_harvested(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    path = _write_raw(
        tmp_path,
        "v2-only",
        project,
        [_user(0, "current request"), _assistant(1, "current final")],
        version=2,
    )

    digests = harvest_dsh(str(tmp_path), scope="all")

    assert [digest.session_id for digest in digests] == ["v2-only"]
    assert digests[0].raw_path == str(path)


def test_v2_header_uses_the_published_required_and_allowed_keys(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    missing = _write_raw(
        tmp_path,
        "missing-seeded",
        project,
        [_user(0, "request"), _assistant(1, "final")],
        version=2,
    )
    rows = [json.loads(line) for line in missing.read_text(encoding="utf-8").splitlines()]
    del rows[0]["isSeeded"]
    missing.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_raw(
        tmp_path,
        "retired-seed-length",
        project,
        [_user(0, "request"), _assistant(1, "final")],
        version=2,
        seedLength=0,
    )
    _write_raw(
        tmp_path,
        "relative-cwd",
        "relative/project",
        [_user(0, "request"), _assistant(1, "final")],
        version=2,
    )

    assert harvest_dsh(str(tmp_path), scope="all") == []


def test_v2_seeded_header_must_match_end_seed_marker(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    _write_raw(
        tmp_path,
        "seeded-without-marker",
        project,
        [_user(0, "request"), _assistant(1, "final")],
        version=2,
        isSeeded=True,
    )
    _write_raw(
        tmp_path,
        "unseeded-with-marker",
        project,
        [_user(0, "request"), _end_seed(1), _assistant(2, "final")],
        version=2,
    )
    invalid_marker = _end_seed(1)
    invalid_marker["data"]["inherited"] = False
    _write_raw(
        tmp_path,
        "invalid-marker",
        project,
        [_user(0, "request"), invalid_marker, _assistant(2, "final")],
        version=2,
    )

    assert harvest_dsh(str(tmp_path), scope="all") == []


def test_master_v2_fixture_is_harvested():
    pytest.importorskip("zstandard")

    digests = harvest_dsh(str(_MASTER_V2_FIXTURE_ROOT), scope="all")

    assert [digest.session_id for digest in digests] == ["master-v2-fixture"]
    assert digests[0].project == "/fixture/project"
    assert digests[0].user_prompts == ["fixture user request"]
    assert digests[0].assistant_finals == ["fixture assistant response"]
    assert digests[0].raw_path.endswith("session.v2.jsonl.zstd")


def test_v2_packed_rows_are_rejected(tmp_path: Path):
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
            "texts": ["a", "b", "c"],
        },
    }
    _write_raw(
        tmp_path,
        "v2-packed",
        project,
        [_user(0, "request"), packed, _assistant(4, "final")],
        version=2,
    )

    assert harvest_dsh(str(tmp_path), scope="all") == []


def test_highest_generation_wins_over_retained_predecessors(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    _write_raw(
        tmp_path,
        "migrated",
        project,
        [_user(0, "old request"), _assistant(1, "old final")],
        version=0,
    )
    _write_raw(
        tmp_path,
        "migrated",
        project,
        [_user(0, "v1 request"), _assistant(1, "v1 final")],
        version=1,
    )
    current = _write_raw(
        tmp_path,
        "migrated",
        project,
        [_user(0, "current request"), _assistant(1, "current final")],
        version=2,
    )

    digests = harvest_dsh(str(tmp_path), scope="all")

    assert [digest.session_id for digest in digests] == ["migrated"]
    assert digests[0].user_prompts == ["current request"]
    assert digests[0].raw_path == str(current)


def test_unsupported_highest_generation_does_not_fall_back(tmp_path: Path):
    project = str((tmp_path / "repo").resolve())
    _write_raw(
        tmp_path,
        "future",
        project,
        [_user(0, "old request"), _assistant(1, "old final")],
        version=0,
    )
    _write_raw(
        tmp_path,
        "future",
        project,
        [_user(0, "future request"), _assistant(1, "future final")],
        version=10,
    )

    assert harvest_dsh(str(tmp_path), scope="all") == []


def test_unsupported_highest_generation_is_diagnosed(tmp_path: Path, caplog, capsys):
    project = str((tmp_path / "repo").resolve())
    _write_raw(
        tmp_path,
        "future",
        project,
        [_user(0, "future request"), _assistant(1, "future final")],
        version=10,
    )

    with caplog.at_level(logging.DEBUG, logger="skillopt_sleep.harvest_dsh"):
        assert harvest_dsh(str(tmp_path), scope="all", progress=True) == []

    diagnostic = " ".join(record.getMessage() for record in caplog.records)
    assert "session.v10.jsonl" in diagnostic
    assert "highest generation v10" in diagnostic
    assert "Upgrade SkillOpt" in diagnostic
    assert "session.v10.jsonl" in capsys.readouterr().err


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
    path = tmp_path / _project_key(project) / _encode_segment("compressed") / "session.v2.jsonl.zstd"
    path.parent.mkdir(parents=True)
    compressor = zstd.ZstdCompressor(write_checksum=True)
    header = json.dumps(_header("compressed", project, version=2)).encode() + b"\n"
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
        progress=False,
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


@pytest.mark.parametrize("offset_hours", [-7, 0, 8])
@pytest.mark.parametrize("cutoff_delta_ms,keep", [(-1, True), (0, True), (1, False)])
def test_since_compares_instants_with_millisecond_precision(tmp_path, offset_hours, cutoff_delta_ms, keep):
    end_ms = _BASE_TIME + 2123
    answer = _assistant(1, "final")
    answer["time"] = end_ms
    _write_raw(tmp_path, "timestamp", str(tmp_path), [_user(0, "request"), answer], version=2)
    cutoff = datetime.fromtimestamp((end_ms + cutoff_delta_ms) / 1000, timezone.utc)
    since = cutoff.astimezone(timezone(timedelta(hours=offset_hours))).isoformat()

    assert bool(harvest_dsh(str(tmp_path), since_iso=since)) is keep


def test_since_accepts_local_sleep_checkpoint(tmp_path):
    _write_raw(tmp_path, "local-time", str(tmp_path), [_user(0, "request"), _assistant(1, "final")])
    # Match state._now_iso's host-local, offset-free timestamps on any platform.
    before = datetime.fromtimestamp((_BASE_TIME + 1000) / 1000).isoformat()
    after = datetime.fromtimestamp((_BASE_TIME + 3000) / 1000).isoformat()

    assert len(harvest_dsh(str(tmp_path), since_iso=before)) == 1
    assert harvest_dsh(str(tmp_path), since_iso=after) == []


def test_master_fixture_since_accepts_equivalent_offsets():
    pytest.importorskip("zstandard")
    utc = harvest_dsh(str(_MASTER_V2_FIXTURE_ROOT), since_iso="2027-01-15T08:00:16Z")
    offset = harvest_dsh(str(_MASTER_V2_FIXTURE_ROOT), since_iso="2027-01-15T16:00:16+08:00")

    assert len(utc) == 1
    assert offset == utc


@pytest.mark.parametrize("kind", ["turn/start", "plugin/info"])
@pytest.mark.parametrize("bad_fields", [
    {"seq": True}, {"seq": 1.0}, {"seq": -1}, {"seq": 2},
    {"time": True}, {"time": -1}, {"data": []}, {"data": None},
    {"type": []}, {"type": ""}, {"type": None},
])
def test_known_and_ignorable_events_share_envelope_validation(tmp_path, kind, bad_fields):
    invalid = {**_metadata(1, kind), "ignorable": True, **bad_fields}
    _write_raw(tmp_path, "invalid", str(tmp_path), [_user(0, "request"), invalid, _assistant(2, "final")])
    _write_raw(tmp_path, "valid", str(tmp_path), [_user(0, "request"), _assistant(1, "final")])

    assert [digest.session_id for digest in harvest_dsh(str(tmp_path))] == ["valid"]


@pytest.mark.parametrize("version", [0, 1, 2])
def test_incremental_replacements_keep_result_and_injected_context_positions(tmp_path, version):
    result = {**_metadata(2, "tool/result"), "surfaceOp": "append"}
    result["data"] = {"output": "private tool output"}
    records = [
        _user(0, "request"),
        _assistant(1, "obsolete", tool_name="obsolete-tool"),
        result,
        _user(3, "injected context", source="plugin"),
        _replace_surface(_user(4, "summary", source="system"), 1, 3, 1, 2, 3),
        _assistant(5, "intermediate", tool_name="intermediate-tool"),
        _replace_surface(_user(6, "new summary", source="system"), 4, 5, 4, 5),
        _assistant(7, "current", tool_name="current-tool"),
    ]
    path = _write_raw(tmp_path, "replaced", str(tmp_path), records, version=version)

    digest = digest_dsh_session(str(path), root=str(tmp_path))

    assert digest is not None
    assert digest.user_prompts == ["request"]
    assert digest.assistant_finals == ["current"]
    assert digest.tools_used == ["current-tool"]
    assert digest.n_assistant_turns == 1


@pytest.mark.parametrize("replacement_fields", [
    {"sourceEventSeqs": [0]},
    {"sourceEventSeqs": [0, 1, 1]},
    {"sourceEventSeqs": [0, True]},
    {"sourceEventSeqs": [[1]]},
    {"sourceEventSeqs": [[1, 0]]},
    {"sourceEventSeqs": [[0, 2]]},
    {"sourceEventSeqs": [[0, 1], [1, 2]]},
    {"sourceEventSeqs": None},
    {"surfaceOp": {"op": "replace", "start": 1, "end": 0}},
    {"surfaceOp": {"op": "replace", "start": 0, "end": 99}},
])
def test_invalid_v2_replacement_discards_complete_session(tmp_path, replacement_fields):
    replacement = _replace_surface(_user(2, "summary", source="system"), 0, 1, 0, 1)
    replacement.update(replacement_fields)
    path = _write_raw(tmp_path, "bad-replace", str(tmp_path), [
        _user(0, "request"), _assistant(1, "old"), replacement, _assistant(3, "new"),
    ], version=2)

    assert digest_dsh_session(str(path), root=str(tmp_path)) is None


def test_digest_releases_raw_payloads_while_reading(tmp_path):
    class Payload(str):
        pass

    refs = []
    path = _write_raw(tmp_path, "streamed", str(tmp_path), [], version=2)

    def records():
        yield _header("streamed", str(tmp_path), version=2)
        yield _user(0, "request")
        for seq in range(1, 21):
            # Allow the reader's current row; earlier payloads must be freed
            # before EOF, even if their surface positions remain visible.
            assert all(ref() is None for ref in refs[:-1])
            payload = Payload("private data " * 10000)
            refs.append(weakref.ref(payload))
            if seq % 2:
                row = _assistant(seq, "visible reply", tool_name="shell")
                row["data"]["message"]["content"][0]["text"] = payload
                row["data"]["message"]["content"][-1]["arguments"] = payload
            else:
                row = {**_metadata(seq, "tool/result"), "surfaceOp": "append", "data": {"output": payload}}
            yield row
            del row, payload

    with mock.patch("skillopt_sleep.harvest_dsh._iter_records", side_effect=lambda _path: records()):
        digest = digest_dsh_session(str(path), root=str(tmp_path))

    assert digest is not None
    assert digest.assistant_finals == ["visible reply"] * 5
    assert digest.n_assistant_turns == 10
    assert digest.tools_used == ["shell"]
    assert all(ref() is None for ref in refs)


def test_unrelated_project_is_skipped_before_events_are_read(tmp_path):
    _write_raw(tmp_path, "other", str(tmp_path / "other"), [], version=2)
    closed = []

    def records(_path):
        try:
            yield _header("other", str(tmp_path / "other"), version=2)
            raise AssertionError("unrelated session body should not be read")
        finally:
            closed.append(True)

    with mock.patch("skillopt_sleep.harvest_dsh._iter_records", side_effect=records):
        assert harvest_dsh(str(tmp_path), scope="invoked", invoked_project=str(tmp_path / "wanted")) == []

    assert closed == [True]

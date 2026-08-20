"""Offline coverage for OpenCode's SQLite transcript source."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from skillopt_sleep.__main__ import _add_common, _cfg_from_args
from skillopt_sleep.config import load_config
from skillopt_sleep.harvest_opencode import (
    _open_database,
    default_opencode_db,
    harvest_opencode,
)
from skillopt_sleep.harvest_sources import harvest_for_config
from skillopt_sleep.types import SessionDigest


def _schema(*, include_metadata: bool = True) -> str:
    metadata_column = ",\n    metadata TEXT" if include_metadata else ""
    return f"""
CREATE TABLE session (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    parent_id TEXT,
    directory TEXT,
    title TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    agent TEXT{metadata_column}
);
CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
CREATE TABLE part (
    id TEXT PRIMARY KEY,
    message_id TEXT,
    session_id TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
"""


_BASE_MS = 1_767_225_600_000  # 2026-01-01T00:00:00Z


def _new_store(
    tmp_path: Path,
    name: str = "opencode.db",
    *,
    wal: bool = False,
    include_metadata: bool = True,
) -> tuple[Path, sqlite3.Connection]:
    path = tmp_path / name
    connection = sqlite3.connect(path)
    connection.executescript(_schema(include_metadata=include_metadata))
    if wal:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return path, connection


def _add_session(
    connection: sqlite3.Connection,
    session_id: str,
    project: str,
    *,
    created: int = _BASE_MS,
    updated: int = _BASE_MS + 60_000,
    parent_id: str | None = None,
    title: str = "Interactive session",
    agent: str = "build",
    metadata: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        "INSERT INTO session "
        "(id, project_id, parent_id, directory, title, time_created, "
        "time_updated, agent, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            "project-1",
            parent_id,
            project,
            title,
            created,
            updated,
            agent,
            json.dumps(metadata if metadata is not None else {}),
        ),
    )


def _add_message(
    connection: sqlite3.Connection,
    session_id: str,
    message_id: str,
    role: str,
    *,
    at: int,
    **extra: Any,
) -> None:
    data = {"role": role, **extra}
    connection.execute(
        "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
        (message_id, session_id, at, at, json.dumps(data)),
    )


def _add_part(
    connection: sqlite3.Connection,
    session_id: str,
    message_id: str,
    part_id: str,
    data: dict[str, Any] | str,
    *,
    at: int,
) -> None:
    raw = data if isinstance(data, str) else json.dumps(data)
    connection.execute(
        "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?, ?)",
        (part_id, message_id, session_id, at, at, raw),
    )


def _add_text_message(
    connection: sqlite3.Connection,
    session_id: str,
    message_id: str,
    role: str,
    text: str,
    *,
    at: int,
    **message_extra: Any,
) -> None:
    _add_message(
        connection,
        session_id,
        message_id,
        role,
        at=at,
        **message_extra,
    )
    _add_part(
        connection,
        session_id,
        message_id,
        f"{message_id}-text",
        {"type": "text", "text": text},
        at=at + 1,
    )


def _add_basic_transcript(
    connection: sqlite3.Connection,
    session_id: str,
    project: str,
    *,
    created: int = _BASE_MS,
    updated: int = _BASE_MS + 60_000,
    prompt: str | None = None,
    answer: str = "The requested change is complete.",
    parent_id: str | None = None,
    title: str = "Interactive session",
    agent: str = "build",
) -> None:
    _add_session(
        connection,
        session_id,
        project,
        created=created,
        updated=updated,
        parent_id=parent_id,
        title=title,
        agent=agent,
    )
    if prompt is None:
        prompt = f"Please finish the real task recorded in session {session_id}."
    user_id = f"{session_id}-user"
    assistant_id = f"{session_id}-assistant"
    _add_text_message(
        connection,
        session_id,
        user_id,
        "user",
        prompt,
        at=created + 1_000,
    )
    _add_text_message(
        connection,
        session_id,
        assistant_id,
        "assistant",
        answer,
        at=updated - 1_000,
    )


def _iso_from_millis(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _ids(digests: Iterable[SessionDigest]) -> list[str]:
    return [digest.session_id for digest in digests]


def test_open_database_closes_connection_if_read_only_setup_fails(monkeypatch) -> None:
    connection = mock.Mock(spec=sqlite3.Connection)
    connection.execute.side_effect = sqlite3.OperationalError("setup failed")
    monkeypatch.setattr(sqlite3, "connect", mock.Mock(return_value=connection))

    with pytest.raises(sqlite3.OperationalError, match="setup failed"):
        _open_database("opencode.db")

    connection.close.assert_called_once_with()


def test_open_database_rejects_writes(tmp_path: Path) -> None:
    path, writer = _new_store(tmp_path)
    writer.commit()
    writer.close()

    connection = _open_database(str(path))
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("INSERT INTO session (id) VALUES ('should-fail')")
    finally:
        connection.close()


# Content and privacy


def test_maps_visible_text_tools_feedback_and_session_fields(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    path, connection = _new_store(tmp_path)
    _add_session(
        connection,
        "s1",
        project,
        metadata={"gitBranch": "feature/opencode"},
    )
    _add_message(connection, "s1", "u1", "user", at=_BASE_MS + 1_000)
    _add_part(
        connection,
        "s1",
        "u1",
        "p1",
        {"type": "text", "text": "Please fix the parser."},
        at=_BASE_MS + 1_100,
    )
    _add_part(
        connection,
        "s1",
        "u1",
        "p2",
        {"type": "text", "text": "It is still broken."},
        at=_BASE_MS + 1_200,
    )
    _add_message(connection, "s1", "a1", "assistant", at=_BASE_MS + 50_000)
    _add_part(
        connection,
        "s1",
        "a1",
        "p3",
        {"type": "text", "text": "I fixed the parser and ran its tests."},
        at=_BASE_MS + 50_100,
    )
    _add_part(
        connection,
        "s1",
        "a1",
        "p4",
        {"type": "tool", "tool": "read.file", "state": {"input": {}, "output": "hidden"}},
        at=_BASE_MS + 50_200,
    )
    _add_part(
        connection,
        "s1",
        "a1",
        "p5",
        {"type": "tool", "tool": "bad tool/<arg>", "state": {"status": "completed"}},
        at=_BASE_MS + 50_300,
    )
    _add_part(
        connection,
        "s1",
        "a1",
        "p6",
        {"type": "tool", "tool": "read.file"},
        at=_BASE_MS + 50_400,
    )
    _add_part(
        connection,
        "s1",
        "a1",
        "p7",
        {
            "type": "tool-invocation",
            "toolInvocation": {"toolName": "legacy.tool", "arguments": "hidden"},
        },
        at=_BASE_MS + 50_500,
    )
    connection.commit()
    connection.close()

    [digest] = harvest_opencode(str(path), scope="all")

    assert digest.session_id == "s1"
    assert digest.project == project
    assert digest.git_branch == "feature/opencode"
    assert digest.started_at == _iso_from_millis(_BASE_MS)
    assert digest.ended_at == _iso_from_millis(_BASE_MS + 60_000)
    assert digest.user_prompts == ["Please fix the parser.\nIt is still broken."]
    assert digest.assistant_finals == ["I fixed the parser and ran its tests."]
    assert digest.tools_used == ["read.file", "bad_tool_arg_", "legacy.tool"]
    assert digest.files_touched == []
    assert digest.n_user_turns == 1
    assert digest.n_assistant_turns == 1
    assert any(signal.startswith("neg:still broken") for signal in digest.feedback_signals)
    assert "neg:opencode_message_error" not in digest.feedback_signals
    assert digest.raw_path == "opencode://s1"


def test_excludes_reasoning_tool_io_files_patches_and_synthetic_text(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    path, connection = _new_store(tmp_path)
    _add_session(
        connection,
        "private",
        project,
        metadata={"providerSecret": "hidden provider metadata"},
    )
    _add_message(connection, "private", "user", "user", at=_BASE_MS + 1_000)
    _add_part(
        connection,
        "private",
        "user",
        "visible-user",
        {"type": "text", "text": "Keep only this user request."},
        at=_BASE_MS + 1_100,
    )
    _add_part(
        connection,
        "private",
        "user",
        "synthetic-user",
        {"type": "text", "text": "hidden synthetic user text", "synthetic": True},
        at=_BASE_MS + 1_200,
    )
    _add_message(
        connection,
        "private",
        "assistant",
        "assistant",
        at=_BASE_MS + 50_000,
        provider="hidden provider id",
        model="hidden model id",
        account="hidden account id",
    )
    private_parts = [
        ("reasoning", {"type": "reasoning", "text": "hidden chain of thought"}),
        (
            "tool",
            {
                "type": "tool",
                "tool": "shell",
                "state": {
                    "input": {"command": "echo hidden tool input"},
                    "output": "hidden tool output",
                },
            },
        ),
        ("file", {"type": "file", "text": "hidden file text", "content": "hidden file content"}),
        ("patch", {"type": "patch", "text": "hidden patch"}),
        ("snapshot", {"type": "snapshot", "snapshot": "hidden snapshot"}),
        ("synthetic", {"type": "text", "text": "hidden synthetic answer", "synthetic": True}),
        ("ignored", {"type": "text", "text": "hidden ignored answer", "ignored": True}),
        ("visible", {"type": "text", "text": "Only this answer is visible."}),
    ]
    for offset, (part_id, data) in enumerate(private_parts, start=1):
        _add_part(
            connection,
            "private",
            "assistant",
            part_id,
            data,
            at=_BASE_MS + 50_000 + offset,
        )
    connection.commit()
    connection.close()

    [digest] = harvest_opencode(str(path), scope="all")
    serialized_digest = json.dumps(digest.to_dict())

    assert digest.user_prompts == ["Keep only this user request."]
    assert digest.assistant_finals == ["Only this answer is visible."]
    assert digest.tools_used == ["shell"]
    assert digest.files_touched == []
    for hidden in (
        "chain of thought",
        "tool input",
        "tool output",
        "file text",
        "file content",
        "hidden patch",
        "hidden snapshot",
        "synthetic",
        "ignored",
        "hidden provider metadata",
        "hidden provider id",
        "hidden model id",
        "hidden account id",
    ):
        assert hidden not in serialized_digest


def test_redacts_visible_user_and_assistant_secrets(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    user_secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    assistant_secret = "super-secret-value-123456"
    path, connection = _new_store(tmp_path)
    _add_basic_transcript(
        connection,
        "secrets",
        project,
        prompt=f"Use Authorization: Bearer {user_secret} for this task.",
        answer=f"Configured api_key={assistant_secret}",
    )
    connection.commit()
    connection.close()

    [digest] = harvest_opencode(str(path), scope="all")
    harvested_text = "\n".join(digest.user_prompts + digest.assistant_finals)

    assert user_secret not in harvested_text
    assert assistant_secret not in harvested_text
    assert "[REDACTED" in harvested_text


def test_preserves_long_visible_text_while_redacting_secrets(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    tail = "::safe-tail::"
    path, connection = _new_store(tmp_path)
    _add_basic_transcript(
        connection,
        "long-redacted-text",
        project,
        prompt="x" * 5000 + secret + tail,
    )
    connection.commit()
    connection.close()

    [digest] = harvest_opencode(str(path), scope="all")

    assert secret not in digest.user_prompts[0]
    assert "sk-" not in digest.user_prompts[0]
    assert len(digest.user_prompts[0]) > 5000
    assert digest.user_prompts[0].endswith(tail)


def test_keeps_all_user_prompts_and_last_five_assistant_finals(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    path, connection = _new_store(tmp_path)
    _add_session(connection, "many-turns", project)

    prompt_count = 41  # Cross the old Copilot-style 40-prompt boundary.
    final_count = 6
    for index in range(prompt_count):
        message_id = f"user-{index:02d}"
        at = _BASE_MS + 1_000 + index
        _add_text_message(
            connection,
            "many-turns",
            message_id,
            "user",
            f"User prompt {index}",
            at=at,
        )

    for index in range(final_count):
        message_id = f"assistant-{index:02d}"
        at = _BASE_MS + 50_000 + index
        _add_text_message(
            connection,
            "many-turns",
            message_id,
            "assistant",
            f"Assistant final {index}",
            at=at,
        )

    connection.commit()
    connection.close()

    [digest] = harvest_opencode(str(path), scope="all")

    assert digest.user_prompts == [f"User prompt {index}" for index in range(prompt_count)]
    assert digest.assistant_finals == [f"Assistant final {index}" for index in range(final_count - 5, final_count)]


def test_records_assistant_errors_and_removes_nul_characters(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    path, connection = _new_store(tmp_path)
    _add_session(connection, "assistant-error", project)
    _add_text_message(
        connection,
        "assistant-error",
        "user",
        "user",
        "Fix\x00 the parser.",
        at=_BASE_MS + 1_000,
    )
    _add_text_message(
        connection,
        "assistant-error",
        "assistant",
        "assistant",
        "The attempt\x00 failed.",
        at=_BASE_MS + 50_000,
        error={"name": "ProviderError"},
    )
    _add_text_message(
        connection,
        "assistant-error",
        "assistant-success",
        "assistant",
        "The retry passed.",
        at=_BASE_MS + 51_000,
    )
    connection.commit()
    connection.close()

    [digest] = harvest_opencode(str(path), scope="all")

    assert digest.user_prompts == ["Fix the parser."]
    assert digest.assistant_finals == ["The attempt failed.", "The retry passed."]
    assert digest.feedback_signals.count("neg:opencode_message_error") == 1


def test_orders_messages_and_text_parts_independently_of_insert_order(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    path, connection = _new_store(tmp_path)
    _add_session(connection, "ordered", project)

    _add_text_message(
        connection,
        "ordered",
        "a-user-late",
        "user",
        "Second prompt",
        at=_BASE_MS + 2_000,
    )
    _add_message(connection, "ordered", "z-user-early", "user", at=_BASE_MS + 1_000)
    _add_part(
        connection,
        "ordered",
        "z-user-early",
        "part-b",
        {"type": "text", "text": "Part B"},
        at=_BASE_MS + 1_200,
    )
    _add_part(
        connection,
        "ordered",
        "z-user-early",
        "part-a",
        {"type": "text", "text": "Part A"},
        at=_BASE_MS + 1_100,
    )
    connection.commit()
    connection.close()

    [digest] = harvest_opencode(str(path), scope="all")

    assert digest.user_prompts == ["Part A\nPart B", "Second prompt"]


# Filtering and SQLite behavior


def test_applies_scope_since_order_and_limit_after_filtering(tmp_path: Path) -> None:
    repo = (tmp_path / "repo").resolve()
    child = (repo / "child").resolve()
    other = (tmp_path / "other").resolve()
    path, connection = _new_store(tmp_path)

    cases = [
        ("old", repo, 100_000),
        ("tie-a", repo, 300_000),
        ("tie-z", repo, 300_000),
        ("new-child", child, 400_000),
        ("other", other, 500_000),
    ]
    for session_id, project, updated_offset in cases:
        _add_basic_transcript(
            connection,
            session_id,
            str(project),
            created=_BASE_MS + updated_offset - 60_000,
            updated=_BASE_MS + updated_offset,
        )

    # The newest row has no visible prompt. It must be skipped before applying
    # a result limit, rather than consuming one of the requested slots.
    _add_session(
        connection,
        "empty-newest",
        str(repo),
        created=_BASE_MS + 540_000,
        updated=_BASE_MS + 600_000,
    )
    connection.commit()
    connection.close()

    assert _ids(harvest_opencode(str(path), scope="all")) == [
        "other",
        "new-child",
        "tie-z",
        "tie-a",
        "old",
    ]
    assert _ids(
        harvest_opencode(
            str(path),
            scope="invoked",
            invoked_project=str(repo),
        )
    ) == ["new-child", "tie-z", "tie-a", "old"]
    assert _ids(harvest_opencode(str(path), scope=[str(other)])) == ["other"]
    assert _ids(
        harvest_opencode(
            str(path),
            scope="all",
            since_iso=_iso_from_millis(_BASE_MS + 300_000),
        )
    ) == ["other", "new-child"]
    assert _ids(harvest_opencode(str(path), scope="all", limit=2)) == [
        "other",
        "new-child",
    ]


def test_filters_skillopt_replay_agents_and_all_child_sessions(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    path, connection = _new_store(tmp_path)
    replay_agent = "skillopt-sleep-0123456789abcdef0123456789abcdef"
    cases = [
        ("exact-self", None, "skillopt-sleep", replay_agent),
        ("same-title", None, "skillopt-sleep", "build"),
        ("rewritten-title", None, "Generated session title", replay_agent),
        (
            "uppercase-agent",
            None,
            "skillopt-sleep",
            "skillopt-sleep-0123456789ABCDEF0123456789ABCDEF",
        ),
        ("child", "same-title", "Child session", "build"),
        ("empty-parent", "", "Child session", "build"),
    ]
    for index, (session_id, parent_id, title, agent) in enumerate(cases):
        _add_basic_transcript(
            connection,
            session_id,
            project,
            created=_BASE_MS + index * 100_000,
            updated=_BASE_MS + index * 100_000 + 60_000,
            parent_id=parent_id,
            title=title,
            agent=agent,
        )
    connection.commit()
    connection.close()

    assert set(_ids(harvest_opencode(str(path), scope="all"))) == {
        "same-title",
        "uppercase-agent",
    }


def test_missing_corrupt_and_incompatible_databases_fail_soft(tmp_path: Path) -> None:
    assert harvest_opencode(str(tmp_path / "missing.db")) == []

    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"this is not a sqlite database")
    assert harvest_opencode(str(corrupt)) == []

    incompatible = tmp_path / "incompatible.db"
    connection = sqlite3.connect(incompatible)
    connection.execute("CREATE TABLE unrelated (id TEXT)")
    connection.commit()
    connection.close()
    assert harvest_opencode(str(incompatible)) == []

    partial = tmp_path / "partial.db"
    connection = sqlite3.connect(partial)
    connection.execute("CREATE TABLE session (id TEXT PRIMARY KEY)")
    connection.commit()
    connection.close()
    assert harvest_opencode(str(partial)) == []


def test_database_path_with_spaces_and_unicode(tmp_path: Path) -> None:
    store_dir = tmp_path / "OpenCode history ü"
    store_dir.mkdir()
    project = str((tmp_path / "repo").resolve())
    path, connection = _new_store(store_dir, "session # data.db")
    _add_basic_transcript(connection, "unicode-path", project)
    connection.commit()
    connection.close()

    assert _ids(harvest_opencode(str(path), scope="all")) == ["unicode-path"]


def test_optional_session_metadata_column_can_be_absent(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    path, connection = _new_store(
        tmp_path,
        "without-metadata.db",
        include_metadata=False,
    )
    connection.execute(
        "INSERT INTO session "
        "(id, project_id, parent_id, directory, title, time_created, time_updated, agent) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "without-metadata",
            "project-1",
            None,
            project,
            "Interactive session",
            _BASE_MS,
            _BASE_MS + 60_000,
            "build",
        ),
    )
    _add_message(connection, "without-metadata", "user", "user", at=_BASE_MS + 1_000)
    _add_part(
        connection,
        "without-metadata",
        "user",
        "user-text",
        {"type": "text", "text": "Harvest this session without metadata."},
        at=_BASE_MS + 1_100,
    )
    connection.commit()
    connection.close()

    [digest] = harvest_opencode(str(path), scope="all")

    assert digest.session_id == "without-metadata"
    assert digest.git_branch == ""


def test_malformed_message_or_part_json_skips_the_affected_sessions(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    path, connection = _new_store(tmp_path)
    _add_session(connection, "bad-message-session", project)
    connection.execute(
        "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
        (
            "bad-message",
            "bad-message-session",
            _BASE_MS + 500,
            _BASE_MS + 500,
            "{not-json",
        ),
    )
    _add_session(connection, "bad-part-session", project)
    _add_message(
        connection,
        "bad-part-session",
        "bad-part-user",
        "user",
        at=_BASE_MS + 1_000,
    )
    _add_part(
        connection,
        "bad-part-session",
        "bad-part-user",
        "bad-part",
        "{not-json",
        at=_BASE_MS + 1_050,
    )
    _add_part(
        connection,
        "bad-part-session",
        "bad-part-user",
        "good-user",
        {"type": "text", "text": "Keep the valid request."},
        at=_BASE_MS + 1_100,
    )
    _add_basic_transcript(
        connection,
        "unaffected",
        project,
        created=_BASE_MS + 100_000,
        updated=_BASE_MS + 160_000,
    )
    connection.commit()
    connection.close()

    assert _ids(harvest_opencode(str(path), scope="all")) == ["unaffected"]


def test_filters_generic_headless_replay_and_agent_sessions(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    path, connection = _new_store(tmp_path)
    _add_basic_transcript(
        connection,
        "headless",
        project,
        prompt="You are a strict grader. Score this response.",
    )
    _add_basic_transcript(
        connection,
        "agent",
        project,
        prompt="You are a Claude-Mem observer. Record this context.",
    )
    _add_basic_transcript(
        connection,
        "short-headless",
        project,
        prompt="Quick automated check",
        created=_BASE_MS + 100_000,
        updated=_BASE_MS + 102_000,
    )
    _add_basic_transcript(
        connection,
        "interactive",
        project,
        prompt="Please keep this normal interactive session.",
        created=_BASE_MS + 200_000,
        updated=_BASE_MS + 260_000,
    )
    connection.commit()
    connection.close()

    assert _ids(harvest_opencode(str(path), scope="all")) == ["interactive"]


def test_reads_committed_session_that_exists_only_in_live_wal(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    path, writer = _new_store(tmp_path, wal=True)
    try:
        _add_basic_transcript(writer, "wal-session", project)
        writer.commit()
        wal_path = Path(str(path) + "-wal")
        assert wal_path.is_file()
        assert wal_path.stat().st_size > 0

        assert _ids(harvest_opencode(str(path), scope="all")) == ["wal-session"]
    finally:
        writer.close()


def test_harvest_does_not_modify_database_or_live_wal(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    path, writer = _new_store(tmp_path, wal=True)
    try:
        _add_basic_transcript(writer, "read-only", project)
        writer.commit()
        wal_path = Path(str(path) + "-wal")
        before_db = path.read_bytes()
        before_wal = wal_path.read_bytes()

        assert _ids(harvest_opencode(str(path), scope="all")) == ["read-only"]

        assert path.read_bytes() == before_db
        assert wal_path.read_bytes() == before_wal
    finally:
        writer.close()


# Path, config, and source routing


def test_default_database_honors_xdg_and_opencode_db(monkeypatch, tmp_path: Path) -> None:
    data_home = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.delenv("OPENCODE_DB", raising=False)

    assert default_opencode_db() == os.path.abspath(data_home / "opencode" / "opencode.db")

    monkeypatch.setenv("OPENCODE_DB", "sessions/custom.db")
    assert default_opencode_db() == os.path.abspath(data_home / "opencode" / "sessions" / "custom.db")

    monkeypatch.setenv("OPENCODE_DB", "~/custom.db")
    assert default_opencode_db() == os.path.abspath(data_home / "opencode" / "~" / "custom.db")

    absolute = tmp_path / "elsewhere" / "custom.db"
    monkeypatch.setenv("OPENCODE_DB", str(absolute))
    assert default_opencode_db() == os.path.abspath(absolute)

    monkeypatch.setenv("OPENCODE_DB", ":memory:")
    assert default_opencode_db() == ""


def test_default_database_honors_windows_appdata(monkeypatch, tmp_path: Path) -> None:
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_DB", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.delenv("APPDATA", raising=False)

    assert default_opencode_db() == os.path.abspath(local_app_data / "opencode" / "opencode.db")


def test_default_database_honors_windows_roaming_appdata(monkeypatch, tmp_path: Path) -> None:
    """A session with only APPDATA set still resolves below Roaming."""
    roaming = tmp_path / "Roaming"
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_DB", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("APPDATA", str(roaming))

    assert default_opencode_db() == os.path.abspath(roaming / "opencode" / "opencode.db")


def test_default_database_prefers_the_appdata_root_that_has_the_database(
    monkeypatch, tmp_path: Path
) -> None:
    """Both roots are set (the usual Windows session) but only Roaming has the db."""
    local_app_data = tmp_path / "LocalAppData"
    roaming = tmp_path / "Roaming"
    roaming_db = roaming / "opencode" / "opencode.db"
    roaming_db.parent.mkdir(parents=True)
    roaming_db.write_bytes(b"")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_DB", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("APPDATA", str(roaming))

    assert default_opencode_db() == os.path.abspath(roaming_db)


def test_default_database_prefers_local_appdata_when_neither_exists(
    monkeypatch, tmp_path: Path
) -> None:
    """With no database on disk the Local root stays the reported default."""
    local_app_data = tmp_path / "LocalAppData"
    roaming = tmp_path / "Roaming"
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_DB", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("APPDATA", str(roaming))

    assert default_opencode_db() == os.path.abspath(
        local_app_data / "opencode" / "opencode.db"
    )


def test_relative_opencode_db_resolves_below_the_selected_windows_root(
    monkeypatch, tmp_path: Path
) -> None:
    """A relative OPENCODE_DB follows the root that actually holds the database."""
    local_app_data = tmp_path / "LocalAppData"
    roaming = tmp_path / "Roaming"
    roaming_db = roaming / "opencode" / "opencode.db"
    roaming_db.parent.mkdir(parents=True)
    roaming_db.write_bytes(b"")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("APPDATA", str(roaming))
    monkeypatch.setenv("OPENCODE_DB", "nightly.db")

    assert default_opencode_db() == os.path.abspath(
        roaming / "opencode" / "nightly.db"
    )


def test_default_database_falls_back_to_home_local_share(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_DB", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    assert default_opencode_db() == os.path.abspath(home / ".local" / "share" / "opencode" / "opencode.db")


def test_empty_database_path_harvests_from_xdg_default(monkeypatch, tmp_path: Path) -> None:
    data_home = tmp_path / "xdg-data"
    store_dir = data_home / "opencode"
    store_dir.mkdir(parents=True)
    project = str((tmp_path / "repo").resolve())
    path, connection = _new_store(store_dir)
    _add_basic_transcript(connection, "xdg-default", project)
    connection.commit()
    connection.close()
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.delenv("OPENCODE_DB", raising=False)

    assert path == store_dir / "opencode.db"
    assert _ids(harvest_opencode(scope="all")) == ["xdg-default"]


def test_cli_and_config_map_opencode_source_and_explicit_database(monkeypatch) -> None:
    parser = argparse.ArgumentParser()
    _add_common(parser)
    args = parser.parse_args(
        [
            "--source",
            "opencode",
            "--opencode-db",
            "~/opencode-test/transcripts.db",
        ]
    )
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)

    cfg = _cfg_from_args(args)
    expected = os.path.abspath(os.path.expanduser("~/opencode-test/transcripts.db"))

    assert cfg.get("transcript_source") == "opencode"
    assert cfg.get("opencode_db") == expected
    assert cfg.opencode_db_path == expected


def test_explicit_memory_database_has_no_persistent_history(monkeypatch) -> None:
    parser = argparse.ArgumentParser()
    _add_common(parser)
    args = parser.parse_args(["--source", "opencode", "--opencode-db", ":memory:"])
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)

    cfg = _cfg_from_args(args)

    assert cfg.opencode_db_path == ":memory:"
    assert harvest_opencode(cfg.opencode_db_path) == []


def test_explicit_config_database_wins_over_environment(monkeypatch, tmp_path: Path) -> None:
    env_db = tmp_path / "environment.db"
    configured_db = tmp_path / "configured.db"
    monkeypatch.setenv("OPENCODE_DB", str(env_db))
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)

    cfg = load_config(opencode_db=str(configured_db))

    assert cfg.opencode_db_path == os.path.abspath(configured_db)


def test_empty_config_leaves_default_database_resolution_to_harvester(monkeypatch) -> None:
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)

    assert load_config().opencode_db_path == ""


def test_explicit_opencode_source_routes_only_to_opencode_harvester(tmp_path: Path) -> None:
    db_path = tmp_path / "configured.db"
    project = str((tmp_path / "repo").resolve())
    cfg = load_config(
        transcript_source="opencode",
        projects="invoked",
        invoked_project=project,
        opencode_db=str(db_path),
    )
    expected = [SessionDigest(session_id="opencode-session", project=project)]
    with (
        mock.patch("skillopt_sleep.harvest_sources.harvest_opencode", return_value=expected) as opencode,
        mock.patch("skillopt_sleep.harvest_sources.harvest_codex") as codex,
        mock.patch("skillopt_sleep.harvest_sources.harvest") as claude,
        mock.patch("skillopt_sleep.harvest_sources.harvest_copilot") as copilot,
        mock.patch("skillopt_sleep.harvest_sources.harvest_copilot_cli") as copilot_cli,
        mock.patch("skillopt_sleep.harvest_sources.harvest_cursor") as cursor,
        mock.patch("skillopt_sleep.harvest_sources.harvest_pi") as pi,
    ):
        actual = harvest_for_config(cfg, since_iso="2026-01-01T00:00:00Z", limit=3)

    assert actual == expected
    opencode.assert_called_once_with(
        os.path.abspath(db_path),
        scope="invoked",
        invoked_project=project,
        since_iso="2026-01-01T00:00:00Z",
        limit=3,
    )
    codex.assert_not_called()
    claude.assert_not_called()
    copilot.assert_not_called()
    copilot_cli.assert_not_called()
    cursor.assert_not_called()
    pi.assert_not_called()


def test_auto_source_keeps_codex_then_claude_precedence_without_opencode(tmp_path: Path) -> None:
    project = str((tmp_path / "repo").resolve())
    cfg = load_config(
        transcript_source="auto",
        projects="invoked",
        invoked_project=project,
        opencode_db=str(tmp_path / "opencode.db"),
    )
    expected = [SessionDigest(session_id="claude-session", project=project)]
    with (
        mock.patch("skillopt_sleep.harvest_sources.harvest_codex", return_value=[]) as codex,
        mock.patch("skillopt_sleep.harvest_sources.harvest", return_value=expected) as claude,
        mock.patch("skillopt_sleep.harvest_sources.harvest_opencode") as opencode,
    ):
        actual = harvest_for_config(cfg, since_iso="2026-01-01T00:00:00Z", limit=4)

    assert actual == expected
    codex.assert_called_once()
    claude.assert_called_once()
    opencode.assert_not_called()

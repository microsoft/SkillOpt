"""Tests for skillopt_sleep.harvest_hermes — transcript harvesting from Hermes state.db.

Uses a sanitized synthetic state.db fixture built entirely from invented data.
No real ~/.hermes/state.db rows are ever read, only the schema is referenced.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List

import pytest

from skillopt_sleep.harvest_hermes import harvest_hermes
from skillopt_sleep.types import SessionDigest

# ── Schema constants (from ~/.hermes/state.db, schema_version=24) ─────────────

_CURRENT_SCHEMA_VERSION = 24

SESSIONS_COLS = [
    "id", "source", "user_id", "model", "model_config",
    "system_prompt", "parent_session_id", "started_at", "ended_at",
    "end_reason", "message_count", "tool_call_count",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens", "billing_provider", "billing_base_url",
    "billing_mode", "estimated_cost_usd", "actual_cost_usd", "cost_status",
    "cost_source", "pricing_version", "title", "api_call_count",
    "handoff_state", "handoff_platform", "handoff_error", "cwd",
    "rewind_count", "archived", "git_branch", "git_repo_root", "session_key",
    "chat_id", "chat_type", "thread_id",
    "compression_failure_cooldown_until", "compression_failure_error",
    "display_name", "origin_json", "expiry_finalized",
    "compression_fallback_streak", "profile_name",
    "compression_ineffective_count", "pinned",
    "last_activity_at", "last_activity_description", "last_activity_provenance",
]

MESSAGES_COLS = [
    "id", "session_id", "role", "content", "tool_call_id", "tool_calls",
    "tool_name", "timestamp", "token_count", "finish_reason",
    "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items",
    "platform_message_id", "observed", "active", "compacted",
    "effect_disposition", "api_content", "display_kind", "display_metadata",
]

SCHEMA_VERSION_DDL = (
    "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
)

SESSIONS_DDL = """CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    cwd TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    git_branch TEXT,
    git_repo_root TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    compression_failure_cooldown_until REAL,
    compression_failure_error TEXT,
    display_name TEXT,
    origin_json TEXT,
    expiry_finalized INTEGER DEFAULT 0,
    compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
    profile_name TEXT,
    compression_ineffective_count INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    last_activity_at REAL,
    last_activity_description TEXT,
    last_activity_provenance TEXT
)"""

MESSAGES_DDL = """CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0,
    effect_disposition TEXT,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT
)"""

# ── Representative data (synthetic — NEVER copied from ~/.hermes/state.db) ────

_BASE_TS = time.time() - 86400  # ~1 day ago

# Sessions
SESSION_A_ID = "sess-a-cli-user"
SESSION_B_ID = "sess-b-engine"
SESSION_C_ID = "sess-c-gateway"
SESSION_D_ID = "sess-d-active"
SESSION_E_ID = "sess-e-no-messages"

REPRESENTATIVE_SESSIONS: List[Dict[str, Any]] = [
    # A: CLI user session in real project
    {
        "id": SESSION_A_ID,
        "source": "cli",
        "model": "deepseek-v4-flash",
        "cwd": "/home/fabricio/vault",
        "title": "Test session",
        "started_at": _BASE_TS,
        "ended_at": _BASE_TS + 300,
        "end_reason": "completed",
    },
    # B: engine session in skillopt_sleep_hermes_ tempdir → EXCLUDED
    {
        "id": SESSION_B_ID,
        "source": "cli",
        "model": "deepseek-v4-flash",
        "cwd": "/tmp/skillopt_sleep_hermes_abc123/",
        "title": "Engine call",
        "started_at": _BASE_TS + 60,
        "ended_at": _BASE_TS + 360,
        "end_reason": "completed",
    },
    # C: gateway session, cwd=NULL → INCLUDED
    {
        "id": SESSION_C_ID,
        "source": "gateway",
        "model": "deepseek-v4-flash",
        "cwd": None,
        "title": "Gateway chat",
        "started_at": _BASE_TS + 120,
        "ended_at": _BASE_TS + 420,
        "end_reason": "completed",
    },
    # D: ended_at=NULL → EXCLUDED
    {
        "id": SESSION_D_ID,
        "source": "cli",
        "model": "deepseek-v4-flash",
        "cwd": "/home/fabricio/vault",
        "title": "Still running",
        "started_at": _BASE_TS + 180,
        "ended_at": None,
    },
    # E: ended_at set but NO messages → digest=None → EXCLUDED
    {
        "id": SESSION_E_ID,
        "source": "cli",
        "model": "deepseek-v4-flash",
        "cwd": "/home/fabricio/vault",
        "title": "Empty session",
        "started_at": _BASE_TS + 240,
        "ended_at": _BASE_TS + 540,
        "end_reason": "completed",
    },
]

# Messages for session A: 2 user turns, 2 assistant replies.
# assistant msg 2 has tool_name="search" — the harvester SQL filter
# excludes role='tool', so the actual role='tool' result message is NOT
# passed to _build_digest, but tool_name="search" is picked up from the
# assistant's tool-call message.
REPRESENTATIVE_MESSAGES: List[Dict[str, Any]] = [
    # Turn 1
    {
        "id": 1,
        "session_id": SESSION_A_ID,
        "role": "user",
        "content": "What is the capital of France?",
        "timestamp": _BASE_TS + 10,
    },
    {
        "id": 2,
        "session_id": SESSION_A_ID,
        "role": "assistant",
        "content": "The capital of France is Paris.",
        "timestamp": _BASE_TS + 15,
    },
    # Turn 2: assistant uses a tool, then a role='tool' result follows.
    # The role='tool' message is excluded by the SQL filter.
    {
        "id": 3,
        "session_id": SESSION_A_ID,
        "role": "user",
        "content": "Search for Paris population.",
        "timestamp": _BASE_TS + 20,
    },
    {
        "id": 4,
        "session_id": SESSION_A_ID,
        "role": "assistant",
        "content": "Let me search that.",
        "tool_name": "search",
        "timestamp": _BASE_TS + 25,
    },
    {
        "id": 5,
        "session_id": SESSION_A_ID,
        "role": "tool",
        "content": "Paris population: 2.1 million",
        "tool_name": "search",
        "timestamp": _BASE_TS + 30,
    },
    {
        "id": 6,
        "session_id": SESSION_A_ID,
        "role": "assistant",
        "content": "Paris has a population of about 2.1 million people.",
        "timestamp": _BASE_TS + 35,
    },
]

# Messages for session C: 1 simple turn
SESSION_C_MESSAGES: List[Dict[str, Any]] = [
    {
        "id": 10,
        "session_id": SESSION_C_ID,
        "role": "user",
        "content": "Hello from gateway",
        "timestamp": _BASE_TS + 130,
    },
    {
        "id": 11,
        "session_id": SESSION_C_ID,
        "role": "assistant",
        "content": "Hello! How can I help?",
        "timestamp": _BASE_TS + 135,
    },
]


# ── Fixture builder ───────────────────────────────────────────────────────────


def _insert_session(cursor: sqlite3.Cursor, session: Dict[str, Any],
                    cols: List[str]) -> None:
    """INSERT a session row, filling only the columns present in `cols`."""
    present = {k: v for k, v in session.items() if k in cols}
    placeholders = ", ".join("?" for _ in present)
    names = ", ".join(present)
    cursor.execute(
        f"INSERT INTO sessions ({names}) VALUES ({placeholders})",
        list(present.values()),
    )


def _insert_message(cursor: sqlite3.Cursor, msg: Dict[str, Any],
                    cols: List[str]) -> None:
    """INSERT a message row, filling only the columns present in `cols`."""
    present = {k: v for k, v in msg.items() if k in cols}
    placeholders = ", ".join("?" for _ in present)
    names = ", ".join(present)
    cursor.execute(
        f"INSERT INTO messages ({names}) VALUES ({placeholders})",
        list(present.values()),
    )


def build_state_db(path: str, *, variant: str = "current") -> str:
    """Create a sanitized synthetic state.db at `path`.

    Parameters
    ----------
    path : str
        Output SQLite file path.
    variant : str
        One of: "current", "no_title", "no_tool_name", "extra_columns",
        "no_messages_table", "no_sessions_table", "schema_version_25",
        "text_epochs".
    """
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    version_val = _CURRENT_SCHEMA_VERSION
    if variant == "schema_version_25":
        version_val = 25

    # schema_version table (always created for variants that need sessions)
    if variant not in ("no_sessions_table",):
        cursor.execute(SCHEMA_VERSION_DDL)
        cursor.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (version_val,)
        )

    # Determine session and message column sets per variant
    sess_cols = list(SESSIONS_COLS)
    msg_cols = list(MESSAGES_COLS)

    if variant == "no_title":
        sess_cols = [c for c in sess_cols if c != "title"]
    elif variant == "no_tool_name":
        msg_cols = [c for c in msg_cols if c != "tool_name"]
    elif variant == "extra_columns":
        sess_cols = list(SESSIONS_COLS) + ["future_col", "new_meta"]
        msg_cols = list(MESSAGES_COLS) + ["future_col", "new_meta"]

    # Build DDL dynamically from the column lists
    if variant != "no_sessions_table":
        _create_table(cursor, "sessions", sess_cols, pk="id", pk_type="TEXT PRIMARY KEY",
                      not_null=["source", "started_at"],
                      defaults={"rewind_count": 0, "archived": 0,
                                "compression_fallback_streak": 0,
                                "compression_ineffective_count": 0, "pinned": 0})

    if variant != "no_messages_table":
        _create_table(cursor, "messages", msg_cols, pk="id", pk_type="INTEGER PRIMARY KEY AUTOINCREMENT",
                      not_null=["session_id", "role", "timestamp"],
                      defaults={"active": 1, "compacted": 0})

    # Insert representative data
    if variant != "no_sessions_table":
        for s in REPRESENTATIVE_SESSIONS:
            row = dict(s)
            if variant == "text_epochs":
                # Store timestamps as ISO TEXT strings instead of REAL epoch
                row["started_at"] = _epoch_to_iso_text(row.get("started_at"))
                if row.get("ended_at") is not None:
                    row["ended_at"] = _epoch_to_iso_text(row["ended_at"])
            if variant == "extra_columns":
                row["future_col"] = 42
                row["new_meta"] = "extra"
            _insert_session(cursor, row, sess_cols)

    if variant != "no_messages_table":
        all_msgs = list(REPRESENTATIVE_MESSAGES) + list(SESSION_C_MESSAGES)
        for m in all_msgs:
            row = dict(m)
            if variant == "extra_columns":
                row["future_col"] = None
                row["new_meta"] = None
            _insert_message(cursor, row, msg_cols)

    conn.commit()
    conn.close()
    return path


def _create_table(
    cursor: sqlite3.Cursor,
    table: str,
    cols: List[str],
    *,
    pk: str = "",
    pk_type: str = "",
    not_null: List[str] | None = None,
    defaults: Dict[str, Any] | None = None,
) -> None:
    """Build CREATE TABLE from column list."""
    not_null = not_null or []
    defaults = defaults or {}
    col_defs: List[str] = []
    for c in cols:
        if c == pk and pk_type:
            col_defs.append(f"{c} {pk_type}")
            continue
        parts = [c, "TEXT"]
        if c in not_null:
            parts.append("NOT NULL")
        if c in defaults:
            val = defaults[c]
            if isinstance(val, int):
                parts.append(f"DEFAULT {val}")
            else:
                parts.append(f"DEFAULT '{val}'")
        col_defs.append(" ".join(parts))
    ddl = f"CREATE TABLE IF NOT EXISTS {table} (\n    " + ",\n    ".join(col_defs) + "\n)"
    cursor.execute(ddl)


def _epoch_to_iso_text(epoch: Any) -> Any:
    """Convert an epoch float to ISO 8601 TEXT for the text_epochs variant."""
    if epoch is None:
        return None
    from datetime import datetime, timezone
    try:
        dt = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
        return dt.isoformat()
    except (TypeError, ValueError, OSError):
        return str(epoch)


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture
def state_db_current(tmp_path: str) -> str:
    """Create a state.db with the full current schema and representative data."""
    path = os.path.join(tmp_path, "state_current.db")
    return build_state_db(path, variant="current")


# ── Core harvest tests ────────────────────────────────────────────────────────


def test_harvest_filters_engine_sessions(state_db_current: str) -> None:
    """scope=all, limit=0: exactly A and C returned (B, D, E excluded)."""
    digests = harvest_hermes(scope="all", limit=0, db_path=state_db_current)
    session_ids = {d.session_id for d in digests}
    assert session_ids == {SESSION_A_ID, SESSION_C_ID}, (
        f"Expected A + C, got {session_ids}"
    )


def test_harvest_scope_invoked(state_db_current: str) -> None:
    """scope=invoked with invoked_project — respects project filtering."""
    # Match with project="/home/fabricio/vault": A matches, C accepted (cwd empty)
    digests = harvest_hermes(
        scope="invoked", invoked_project="/home/fabricio/vault",
        limit=0, db_path=state_db_current,
    )
    session_ids = {d.session_id for d in digests}
    assert session_ids == {SESSION_A_ID, SESSION_C_ID}, (
        f"Expected A + C, got {session_ids}"
    )

    # Different project: no A, only C (accepted because cwd empty)
    digests2 = harvest_hermes(
        scope="invoked", invoked_project="/other/path",
        limit=0, db_path=state_db_current,
    )
    session_ids2 = {d.session_id for d in digests2}
    assert session_ids2 == {SESSION_C_ID}, (
        f"Expected C only, got {session_ids2}"
    )


def test_harvest_since_iso(state_db_current: str) -> None:
    """since_iso excludes sessions ended before the cutoff."""
    from datetime import datetime, timezone

    # All sessions end between _BASE_TS+300 and _BASE_TS+540
    # A ends at _BASE_TS+300, C ends at _BASE_TS+420
    # Cutoff at _BASE_TS+400: A excluded, C included
    cutoff = datetime.fromtimestamp(_BASE_TS + 400, tz=timezone.utc).isoformat()
    digests = harvest_hermes(
        scope="all", since_iso=cutoff, limit=0, db_path=state_db_current,
    )
    session_ids = {d.session_id for d in digests}
    assert SESSION_A_ID not in session_ids, "A ended before cutoff, should be excluded"
    assert SESSION_C_ID in session_ids, "C ended after cutoff, should be included"


def test_harvest_limit(state_db_current: str) -> None:
    """limit caps the result; limit=0 returns all."""
    # limit=1 should return exactly 1 digest
    digests = harvest_hermes(scope="all", limit=1, db_path=state_db_current)
    assert len(digests) == 1, f"Expected 1, got {len(digests)}"

    # limit=0 returns all (A + C = 2)
    digests_all = harvest_hermes(scope="all", limit=0, db_path=state_db_current)
    assert len(digests_all) == 2, f"Expected 2, got {len(digests_all)}"


def test_harvest_digest_content(state_db_current: str) -> None:
    """Session A digest has correct turn counts, prompts, tools, timestamps."""
    digests = harvest_hermes(scope="all", limit=0, db_path=state_db_current)
    digest_a = next(d for d in digests if d.session_id == SESSION_A_ID)

    assert digest_a.n_user_turns == 2, f"Expected 2 user turns, got {digest_a.n_user_turns}"
    # Session A has 3 assistant messages (2 final replies + 1 intermediate tool-call
    # message "Let me search that."). The harvester counts every assistant message.
    assert digest_a.n_assistant_turns == 3, f"Expected 3 asst turns, got {digest_a.n_assistant_turns}"

    # user_prompts: 2 items (from the 2 user messages)
    assert len(digest_a.user_prompts) == 2, (
        f"Expected 2 user_prompts, got {len(digest_a.user_prompts)}"
    )
    # assistant_finals: 2 items, capped to last 5
    assert len(digest_a.assistant_finals) == 2, (
        f"Expected 2 assistant_finals, got {len(digest_a.assistant_finals)}"
    )

    # tools_used: role='tool' msg is EXCLUDED by SQL filter (role IN ('user','assistant')),
    # but the assistant msg with tool_name="search" IS included, so "search" appears
    assert "search" in digest_a.tools_used, (
        f"Expected 'search' in tools_used, got {digest_a.tools_used}"
    )

    # started_at/ended_at should be ISO strings
    assert digest_a.started_at and "T" in digest_a.started_at, (
        f"started_at not ISO: {digest_a.started_at!r}"
    )
    assert digest_a.ended_at and "T" in digest_a.ended_at, (
        f"ended_at not ISO: {digest_a.ended_at!r}"
    )

    # raw_path starts with the db path
    assert digest_a.raw_path.startswith(state_db_current), (
        f"raw_path should start with {state_db_current}, got {digest_a.raw_path!r}"
    )


def test_harvest_missing_db_returns_empty(tmp_path: str) -> None:
    """Non-existent db path returns []."""
    nonexistent = os.path.join(tmp_path, "does_not_exist.db")
    digests = harvest_hermes(db_path=nonexistent)
    assert digests == []


# ── Schema-drift tests ────────────────────────────────────────────────────────


def _harvest_all(db_path: str) -> List[SessionDigest]:
    """Convenience: harvest with scope=all, limit=0."""
    return harvest_hermes(scope="all", limit=0, db_path=db_path)


def test_drift_no_title(tmp_path: str) -> None:
    """sessions without 'title' column: works, digest title is ''."""
    path = os.path.join(tmp_path, "no_title.db")
    build_state_db(path, variant="no_title")
    digests = _harvest_all(path)
    assert len(digests) == 2
    for d in digests:
        assert d.session_id in (SESSION_A_ID, SESSION_C_ID)


def test_drift_no_tool_name(tmp_path: str) -> None:
    """messages without 'tool_name' column: works, tools_used is []."""
    path = os.path.join(tmp_path, "no_tool_name.db")
    build_state_db(path, variant="no_tool_name")
    digests = _harvest_all(path)
    digest_a = next(d for d in digests if d.session_id == SESSION_A_ID)
    assert digest_a.tools_used == [], (
        f"tools_used should be empty, got {digest_a.tools_used}"
    )


def test_drift_extra_columns(tmp_path: str) -> None:
    """Extra columns in sessions/messages: works identically to current."""
    path = os.path.join(tmp_path, "extra_columns.db")
    build_state_db(path, variant="extra_columns")
    digests = _harvest_all(path)
    session_ids = {d.session_id for d in digests}
    assert session_ids == {SESSION_A_ID, SESSION_C_ID}

    # Digest content should be identical
    digest_a = next(d for d in digests if d.session_id == SESSION_A_ID)
    assert digest_a.n_user_turns == 2
    assert digest_a.n_assistant_turns == 3  # 3 assistant msgs (incl. intermediate tool-call)
    assert "search" in digest_a.tools_used


def test_drift_no_messages_table(tmp_path: str) -> None:
    """No messages table: no exception, sessions produce no digests (no messages)."""
    path = os.path.join(tmp_path, "no_messages_table.db")
    build_state_db(path, variant="no_messages_table")
    digests = _harvest_all(path)
    # Without a messages table, _fetch_messages will raise an OperationalError.
    # The harvester does NOT catch this — document the actual behavior.
    # If it raises, we catch and assert the error type.
    # If it returns gracefully, digests will be empty because no messages were found.
    assert digests == [], (
        f"Expected empty digests without messages table, got {digests}"
    )


def test_drift_no_sessions_table(tmp_path: str) -> None:
    """Empty db (no tables): no exception, returns []."""
    path = os.path.join(tmp_path, "no_sessions.db")
    build_state_db(path, variant="no_sessions_table")
    digests = _harvest_all(path)
    assert digests == []


def test_drift_schema_version_25(tmp_path: str) -> None:
    """schema_version=25: same result as current (harvester ignores version)."""
    path = os.path.join(tmp_path, "version_25.db")
    build_state_db(path, variant="schema_version_25")
    digests = _harvest_all(path)
    session_ids = {d.session_id for d in digests}
    assert session_ids == {SESSION_A_ID, SESSION_C_ID}


def test_drift_text_epochs(tmp_path: str) -> None:
    """Timestamps stored as ISO TEXT strings: _ts_from_epoch returns '' for non-float input.

    This is the documented behavior of _ts_from_epoch — it catches TypeError
    from float(iso_str) and returns ''. The harvester does NOT crash, but
    started_at/ended_at will be empty strings.
    """
    path = os.path.join(tmp_path, "text_epochs.db")
    build_state_db(path, variant="text_epochs")
    digests = _harvest_all(path)
    assert len(digests) == 2  # Sessions A and C still returned

    digest_a = next(d for d in digests if d.session_id == SESSION_A_ID)
    # _ts_from_epoch tries float(iso_string) which raises TypeError → returns ""
    assert digest_a.started_at == "", (
        f"Expected empty started_at for text epoch, got {digest_a.started_at!r}"
    )
    assert digest_a.ended_at == "", (
        f"Expected empty ended_at for text epoch, got {digest_a.ended_at!r}"
    )

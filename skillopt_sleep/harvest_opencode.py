"""Read OpenCode session transcripts from its local SQLite database.

OpenCode normally stores sessions in ``opencode.db`` below its XDG data
directory.
This harvester opens that database read-only and keeps only visible user and
assistant text plus short tool names. Reasoning, tool arguments and results,
file payloads, patches, and provider metadata are never copied into digests.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional
from urllib.request import pathname2url

from skillopt_sleep.harvest import (
    _detect_feedback,
    _is_agent_session,
    _is_headless_replay,
    _is_meta_prompt,
    _project_matches,
)
from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.types import SessionDigest

OPENCODE_REPLAY_AGENT_RE = re.compile(r"skillopt-sleep-[0-9a-f]{32}\Z")

_REQUIRED_COLUMNS = {
    "session": {
        "id",
        "parent_id",
        "directory",
        "agent",
        "time_created",
        "time_updated",
    },
    "message": {"id", "session_id", "time_created", "data"},
    "part": {"id", "message_id", "session_id", "time_created", "data"},
}


def _opencode_data_candidates() -> List[str]:
    """Where OpenCode may keep its data directory, most specific first."""
    data_home = os.environ.get("XDG_DATA_HOME", "")
    if data_home:
        return [os.path.join(os.path.abspath(os.path.expanduser(data_home)), "opencode")]

    candidates: List[str] = []
    if sys.platform == "win32":
        for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("APPDATA")):
            if base:
                candidates.append(
                    os.path.join(os.path.abspath(os.path.expanduser(base)), "opencode")
                )
    candidates.append(
        os.path.join(os.path.expanduser("~"), ".local", "share", "opencode")
    )
    return list(dict.fromkeys(candidates))


def default_opencode_db() -> str:
    """Return the OpenCode database selected by its environment variables."""
    candidates = _opencode_data_candidates()
    opencode_data = candidates[0]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "opencode.db")):
            opencode_data = candidate
            break

    configured = os.environ.get("OPENCODE_DB", "")
    if configured == ":memory:":
        return ""
    if configured:
        if os.path.isabs(configured):
            return os.path.abspath(configured)
        return os.path.abspath(os.path.join(opencode_data, configured))
    return os.path.abspath(os.path.join(opencode_data, "opencode.db"))


def _ro_uri(path: str) -> str:
    return "file:" + pathname2url(os.path.abspath(path)) + "?mode=ro"


def _open_database(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(
        _ro_uri(path),
        uri=True,
        isolation_level=None,
        timeout=5,
    )
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection
    except sqlite3.Error:
        connection.close()
        raise


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _has_supported_schema(connection: sqlite3.Connection) -> bool:
    table_names = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}
    for table, required in _REQUIRED_COLUMNS.items():
        if table not in table_names:
            return False
        columns = _table_columns(connection, table)
        if not required.issubset(columns):
            return False
    return True


def _load_object(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _sanitize_text(parts: Iterable[str]) -> str:
    return str(redact_secrets("\n".join(parts))).replace("\x00", "").strip()


def _sanitize_tool_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())[:80]


def _tool_name(part: dict[str, Any]) -> str:
    if part.get("type") == "tool":
        return _sanitize_tool_name(part.get("tool"))
    # Older OpenCode exports used this shape. It is safe to accept because only
    # the short name is read; arguments and results remain ignored.
    if part.get("type") == "tool-invocation":
        invocation = part.get("toolInvocation")
        if isinstance(invocation, dict):
            return _sanitize_tool_name(invocation.get("toolName"))
    return ""


def _millis_to_iso(value: Any) -> str:
    if type(value) not in {int, float} or value <= 0:
        return ""
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (OSError, OverflowError, ValueError):
        return ""


def _iso_to_millis(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        # Sleep checkpoints without an offset are stored in local time, so let
        # timestamp() interpret them in the host timezone.
        return int(parsed.timestamp() * 1000)
    except (OSError, TypeError, ValueError):
        return None


def _dedup(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _is_skillopt_session(row: sqlite3.Row) -> bool:
    agent = row["agent"]
    return isinstance(agent, str) and OPENCODE_REPLAY_AGENT_RE.fullmatch(agent) is not None


def _is_harvestable_session(
    session: sqlite3.Row,
    scope: Any,
    invoked_project: str,
) -> bool:
    session_id = session["id"]
    directory = session["directory"]
    created = session["time_created"]
    updated = session["time_updated"]
    return (
        session["parent_id"] is None
        and isinstance(session_id, str)
        and bool(session_id)
        and isinstance(directory, str)
        and os.path.isabs(directory)
        and type(created) in {int, float}
        and created > 0
        and type(updated) in {int, float}
        and updated > 0
        and _project_matches(directory, scope, invoked_project)
        and not _is_skillopt_session(session)
    )


def _digest_session(
    session: sqlite3.Row,
    messages: list[sqlite3.Row],
    parts_by_message: dict[str, list[sqlite3.Row]],
) -> Optional[SessionDigest]:
    prompts: List[str] = []
    finals: List[str] = []
    tools: List[str] = []
    feedback: List[str] = []
    n_user = 0
    n_assistant = 0

    for message_row in messages:
        message = _load_object(message_row["data"])
        if message is None:
            return None
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue

        texts: List[str] = []
        message_tools: List[str] = []
        for part_row in parts_by_message.get(str(message_row["id"]), []):
            part = _load_object(part_row["data"])
            if part is None:
                return None
            if (
                part.get("type") == "text"
                and part.get("synthetic") is not True
                and part.get("ignored") is not True
                and isinstance(part.get("text"), str)
            ):
                texts.append(part["text"])
            if role == "assistant":
                name = _tool_name(part)
                if name:
                    message_tools.append(name)

        text = _sanitize_text(texts)
        if role == "user":
            if text and not _is_meta_prompt(text):
                n_user += 1
                feedback.extend(_detect_feedback(text))
                prompts.append(text)
        else:
            n_assistant += 1
            tools.extend(message_tools)
            if message.get("error"):
                feedback.append("neg:opencode_message_error")
            if text:
                finals.append(text)
                if len(finals) > 5:
                    finals.pop(0)

    if not prompts:
        return None

    metadata = _load_object(session["metadata"]) or {}
    branch = metadata.get("gitBranch")
    digest = SessionDigest(
        session_id=str(session["id"]),
        project=str(session["directory"]),
        git_branch=branch if isinstance(branch, str) else "",
        started_at=_millis_to_iso(session["time_created"]),
        ended_at=_millis_to_iso(session["time_updated"]),
        user_prompts=prompts,
        assistant_finals=finals,
        tools_used=_dedup(tools),
        files_touched=[],
        feedback_signals=_dedup(feedback),
        n_user_turns=n_user,
        n_assistant_turns=n_assistant,
        raw_path=f"opencode://{session['id']}",
    )
    if _is_headless_replay(digest) or _is_agent_session(digest):
        return None
    return digest


def harvest_opencode(
    db_path: str = "",
    *,
    scope: Any = "all",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
) -> List[SessionDigest]:
    """Read root OpenCode sessions and return matching transcript digests."""
    if db_path == ":memory:":
        return []
    path = os.path.abspath(os.path.expanduser(db_path)) if db_path else default_opencode_db()
    if not path or not os.path.isfile(path):
        return []
    if limit < 0:
        limit = 0

    cutoff = _iso_to_millis(since_iso)
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = _open_database(path)
        connection.execute("BEGIN")
        if not _has_supported_schema(connection):
            connection.execute("ROLLBACK")
            return []

        metadata_column = "metadata" if "metadata" in _table_columns(connection, "session") else "NULL AS metadata"
        session_where = "WHERE parent_id IS NULL"
        session_params: list[Any] = []
        if cutoff is not None:
            session_where += " AND time_updated > ?"
            session_params.append(cutoff)
        session_rows = connection.execute(
            f"SELECT id, parent_id, directory, agent, {metadata_column}, "
            "time_created, time_updated FROM session "
            f"{session_where} ORDER BY time_updated DESC, id DESC",
            session_params,
        )

        digests: List[SessionDigest] = []
        for session in session_rows:
            session_id = session["id"]
            if not _is_harvestable_session(session, scope, invoked_project):
                continue

            messages = connection.execute(
                "SELECT id, data FROM message WHERE session_id = ? ORDER BY time_created, id",
                (session_id,),
            ).fetchall()
            parts = connection.execute(
                "SELECT message_id, data FROM part WHERE session_id = ? ORDER BY message_id, id",
                (session_id,),
            ).fetchall()
            parts_by_message: dict[str, list[sqlite3.Row]] = {}
            for part in parts:
                parts_by_message.setdefault(str(part["message_id"]), []).append(part)
            digest = _digest_session(
                session,
                messages,
                parts_by_message,
            )
            if digest is None:
                continue
            digests.append(digest)
            if limit and len(digests) >= limit:
                break
        connection.execute("COMMIT")
        return digests
    except (OSError, sqlite3.Error):
        if connection is not None and connection.in_transaction:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        return []
    finally:
        if connection is not None:
            connection.close()

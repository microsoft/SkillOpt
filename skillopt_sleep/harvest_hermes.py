"""Hermes Agent session harvesting for SkillOpt-Sleep.

Reads session transcripts from the Hermes Agent state database
(``~/.hermes/state.db``) and returns ``SessionDigest`` objects.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional

from skillopt_sleep.types import SessionDigest

HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
STATE_DB = os.path.join(HERMES_HOME, "state.db")


def _filter_engine_sessions(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Skip sessions created by the engine's own backend calls.

    These sessions run in temp dirs (prefix ``skillopt_sleep_hermes_``) and
    represent optimizer/target/grader calls, not real user sessions. We filter
    by ``cwd`` matching the tempdir pattern used in ``HermesBackend._call()``.
    """
    out: List[Dict[str, Any]] = []
    for s in sessions:
        cwd = (s.get("cwd") or "").strip()
        if not cwd:
            # No cwd → probably a gateway session; keep it
            out.append(s)
        elif "skillopt_sleep_hermes_" in cwd:
            # Engine's own tempdir → skip
            continue
        elif cwd.startswith("/tmp/") and len(cwd.split("/", 3)) <= 4:
            # Very short-lived temp sessions; likely programmatic
            continue
        else:
            out.append(s)
    return out


def _fetch_messages(db_path: str, session_id: str) -> List[Dict[str, Any]]:
    """Return all messages for a session, ordered by id."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """SELECT role, content, tool_name, timestamp
           FROM messages
           WHERE session_id = ? AND role IN ('user', 'assistant')
           ORDER BY id""",
        (session_id,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def _build_digest(
    session: Dict[str, Any],
    messages: List[Dict[str, Any]],
    scope: str = "invoked",
    invoked_project: str = "",
) -> Optional[SessionDigest]:
    """Build a ``SessionDigest`` from one session + its messages.

    Returns ``None`` if the session has no user or assistant turns, or if it
    doesn't match the project scope.
    """
    session_id = session.get("id") or ""
    project = (session.get("cwd") or "").strip()
    title = (session.get("title") or "").strip()

    user_prompts: List[str] = []
    assistant_finals: List[str] = []
    tools: List[str] = []
    n_user = 0
    n_asst = 0

    # Collect last assistant message after each user turn (the "final" reply)
    last_assistant = ""
    for msg in messages:
        role = (msg.get("role") or "").strip()
        content = (msg.get("content") or "").strip()
        tool = (msg.get("tool_name") or "").strip()

        if role == "user" and content:
            n_user += 1
            user_prompts.append(content)
            # Flush any pending assistant final
            if last_assistant:
                assistant_finals.append(last_assistant)
                last_assistant = ""
        elif role == "assistant" and content:
            n_asst += 1
            last_assistant = content
        if tool:
            tools.append(tool)

    # Flush the last assistant message
    if last_assistant:
        assistant_finals.append(last_assistant)

    if n_user == 0 and n_asst == 0:
        return None

    # Project matching
    if not _project_matches(project, scope, invoked_project):
        return None

    # Dedup
    def _dedup(xs: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return SessionDigest(
        session_id=session_id,
        project=project,
        started_at=_ts_from_epoch(session.get("started_at")),
        ended_at=_ts_from_epoch(session.get("ended_at")),
        user_prompts=user_prompts,
        assistant_finals=assistant_finals[-5:],
        tools_used=_dedup(tools),
        files_touched=[],
        feedback_signals=[],
        n_user_turns=n_user,
        n_assistant_turns=n_asst,
        raw_path=f"{STATE_DB}:{session_id}",
    )


def _ts_from_epoch(epoch: Any) -> str:
    """Convert a Unix epoch (float/int) to ISO 8601 string."""
    if epoch is None:
        return ""
    try:
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
        return dt.isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def _project_matches(project: str, scope: str, invoked: str) -> bool:
    """Check whether ``project`` matches the scope."""
    if not invoked or scope == "all":
        return True
    if not project:
        return True  # no cwd → can't filter, accept
    a = os.path.abspath(project)
    b = os.path.abspath(invoked)
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def harvest_hermes(
    *,
    scope: str = "invoked",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
    db_path: str = "",
) -> List[SessionDigest]:
    """Walk ``~/.hermes/state.db`` and return matching digests.

    Parameters
    ----------
    scope : str
        ``"all"`` | ``"invoked"`` | list of paths
    invoked_project : str
        Used when ``scope == "invoked"``.
    since_iso : str | None
        ISO 8601; only sessions starting after this are kept.
    limit : int
        Cap number of digests (0 = no cap).
    db_path : str
        Override state.db path (default: ``~/.hermes/state.db``).
    """
    db = db_path or STATE_DB
    if not os.path.isfile(db):
        return []

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build query with optional since filter
    where = "WHERE cwd IS NOT NULL AND cwd != '' AND ended_at IS NOT NULL"
    params: List[Any] = []
    if since_iso:
        since_epoch = _epoch_from_iso(since_iso)
        if since_epoch is not None:
            where += " AND ended_at >= ?"
            params.append(since_epoch)

    cursor.execute(
        f"""SELECT id, cwd, title, started_at, ended_at, model
            FROM sessions
            {where}
            ORDER BY ended_at DESC
            LIMIT ?""",
        params + [(limit or 200)],
    )

    sessions = [dict(r) for r in cursor.fetchall()]
    conn.close()

    # Filter engine sessions
    sessions = _filter_engine_sessions(sessions)

    digests: List[SessionDigest] = []
    for s in sessions:
        sid = s.get("id") or ""
        msgs = _fetch_messages(db, sid)
        digest = _build_digest(
            s, msgs,
            scope=scope,
            invoked_project=invoked_project,
        )
        if digest is None:
            continue
        digests.append(digest)
        if limit and len(digests) >= limit:
            break

    return digests


def _epoch_from_iso(iso: str) -> Optional[float]:
    """Convert ISO 8601 string to Unix epoch. Returns None on failure."""
    try:
        from datetime import datetime, timezone

        # Handle Z suffix
        s = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None

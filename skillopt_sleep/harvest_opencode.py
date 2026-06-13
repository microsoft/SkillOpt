"""SkillOpt-Sleep - OpenCode transcript harvest.

OpenCode stores conversations in ``~/.local/share/opencode/opencode.db``.
This module reads that SQLite database in read-only mode and normalizes
``session`` + ``message`` + ``part`` rows into :class:`SessionDigest` objects.
It performs no writes and no network calls.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from skillopt_sleep.harvest import _detect_feedback, _is_meta_prompt
from skillopt_sleep.types import SessionDigest


def _connect_readonly(path: str) -> sqlite3.Connection:
    uri = "file:" + os.path.abspath(path) + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _loads(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        obj = json.loads(raw)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _text_from_part(data: Dict[str, Any]) -> str:
    if data.get("type") != "text":
        return ""
    text = data.get("text")
    return text if isinstance(text, str) else ""


def _tool_name_from_part(data: Dict[str, Any]) -> str:
    if data.get("type") != "tool":
        return ""
    tool = data.get("tool")
    return str(tool) if tool else ""


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _iter_strings(v)


def _looks_like_path(value: str) -> bool:
    if not value or len(value) > 500:
        return False
    if value.startswith(("/", "~/", "./", "../")):
        return True
    return any(sep in value for sep in ("/", "\\")) and "." in os.path.basename(value)


def _files_from_tool_part(data: Dict[str, Any]) -> List[str]:
    if data.get("type") != "tool":
        return []
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    inputs = state.get("input") if isinstance(state.get("input"), dict) else {}
    found: List[str] = []
    for value in _iter_strings(inputs):
        if _looks_like_path(value):
            found.append(value)
    return found


def _dedup(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _ms_to_iso(ms: int | None) -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _iso_to_ms(value: str | None) -> int:
    if not value:
        return 0
    try:
        normalized = value.replace("Z", "+00:00")
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)
    except Exception:
        return 0


def _project_matches(project: str, scope: Any, invoked: str) -> bool:
    if scope == "all":
        return True
    if isinstance(scope, (list, tuple)):
        return any(os.path.abspath(project) == os.path.abspath(p) for p in scope)
    if not invoked:
        return True
    a = os.path.abspath(project)
    b = os.path.abspath(invoked)
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def digest_opencode_session(conn: sqlite3.Connection, session_id: str) -> Optional[SessionDigest]:
    """Build a digest from one OpenCode session id."""
    cur = conn.execute(
        """
        select s.id, s.directory, s.title, s.time_created, s.time_updated,
               s.path, s.metadata, p.worktree
        from session s
        left join project p on p.id = s.project_id
        where s.id = ?
        """,
        (session_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None

    (_sid, directory, _title, created, updated, path_raw, metadata_raw, worktree) = row
    project = directory or worktree or ""
    git_branch = ""
    metadata = _loads(metadata_raw)
    if isinstance(metadata.get("gitBranch"), str):
        git_branch = metadata["gitBranch"]
    path_data = _loads(path_raw)
    if not project and isinstance(path_data.get("cwd"), str):
        project = path_data["cwd"]

    user_prompts: List[str] = []
    assistant_finals: List[str] = []
    tools: List[str] = []
    files: List[str] = []
    feedback: List[str] = []
    n_user = 0
    n_asst = 0

    rows = conn.execute(
        """
        select m.id, m.data, p.data
        from message m
        left join part p on p.message_id = m.id
        where m.session_id = ?
        order by m.time_created, m.id, p.time_created, p.id
        """,
        (session_id,),
    )
    message_parts: Dict[str, Dict[str, Any]] = {}
    for message_id, message_raw, part_raw in rows:
        msg = _loads(message_raw)
        mid = str(message_id or msg.get("id") or id(message_raw))
        entry = message_parts.setdefault(mid, {"role": msg.get("role"), "texts": [], "tools": [], "files": []})
        part = _loads(part_raw)
        text = _text_from_part(part)
        if text:
            entry["texts"].append(text)
        tool = _tool_name_from_part(part)
        if tool:
            entry["tools"].append(tool)
        entry["files"].extend(_files_from_tool_part(part))

    for entry in message_parts.values():
        role = entry.get("role")
        texts = [str(t).strip() for t in entry.get("texts", []) if str(t).strip()]
        if role == "user":
            usable = [t for t in texts if not _is_meta_prompt(t)]
            if usable:
                n_user += 1
                joined = "\n".join(usable).strip()
                user_prompts.append(joined)
                feedback.extend(_detect_feedback(joined))
        elif role == "assistant":
            n_asst += 1
            tools.extend(entry.get("tools", []))
            files.extend(entry.get("files", []))
            if texts:
                assistant_finals.append("\n".join(texts).strip())

    if n_user == 0 and n_asst == 0:
        return None

    return SessionDigest(
        session_id=session_id,
        project=project,
        git_branch=git_branch,
        started_at=_ms_to_iso(created),
        ended_at=_ms_to_iso(updated),
        user_prompts=user_prompts,
        assistant_finals=assistant_finals[-5:],
        tools_used=_dedup(tools),
        files_touched=_dedup(files)[:20],
        feedback_signals=feedback,
        n_user_turns=n_user,
        n_assistant_turns=n_asst,
        raw_path=f"opencode://{session_id}",
    )


def harvest_opencode(
    db_path: str,
    *,
    scope: Any = "all",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
) -> List[SessionDigest]:
    """Read OpenCode sessions from SQLite and return matching digests."""
    if not os.path.exists(db_path):
        return []

    since_ms = _iso_to_ms(since_iso)
    digests: List[SessionDigest] = []
    try:
        conn = _connect_readonly(db_path)
    except sqlite3.Error:
        return []
    try:
        query = "select id from session"
        params: List[Any] = []
        if since_ms:
            query += " where time_updated >= ?"
            params.append(since_ms)
        query += " order by time_updated desc"
        if limit:
            query += " limit ?"
            params.append(limit * 4)
        for (session_id,) in conn.execute(query, params):
            digest = digest_opencode_session(conn, session_id)
            if digest is None:
                continue
            if not _project_matches(digest.project or "", scope, invoked_project):
                continue
            digests.append(digest)
            if limit and len(digests) >= limit:
                break
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return digests

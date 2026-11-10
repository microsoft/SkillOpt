"""SkillOpt-Sleep — pi (pi-coding-agent) session harvesting.

Reads pi session transcript JSONL files (one per session, stored under
``~/.pi/agent/sessions/<project-slug>/<sessionId>.jsonl``) and normalizes them
into :class:`SessionDigest` records without copying tool arguments, private
reasoning blocks (``thinking``), or raw tool outputs.

pi schema (verified against the upstream session format):
  * A session file is a JSONL stream of entries with a ``type`` discriminator.
  * ``type == "session"``  — exactly one per file; carries ``cwd`` + ``timestamp``.
  * Version 1 sessions are linear. Versions 2 and 3 form an append-only tree
    using entry ``id`` / ``parentId`` fields; only the branch ending at the last
    entry is the active conversation and is harvested.
  * ``type == "message"``  — a conversational turn. ``message.role`` ∈
    {user, assistant, toolResult}; ``message.content`` is either a string or a
    list of content blocks. Block types include ``text`` (kept), ``thinking``
    (private reasoning, skipped), and ``toolCall`` (carries ``name``).
  * toolResult messages can carry ``isError`` and ``toolName``. Tool names are
    retained, while transient per-call errors are not treated as task feedback.
  * Other types (``model_change``, ``thinking_level_change``, ``custom``, ...) are
    metadata / tool-result payloads and are skipped for digestion.

This module performs NO writes and NO network calls.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable, List, Optional

from skillopt_sleep.harvest import (
    _detect_feedback,
    _ended_before,
    _is_headless_replay,
    _is_meta_prompt,
    _project_matches,
    _text_from_content,
)
from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.types import SessionDigest


def _redact_secrets(text: str) -> str:
    """Backward-compatible string wrapper around shared secret redaction."""
    return str(redact_secrets(text))


def _sanitize_tool_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(name))[:80]


def _read_pi_jsonl(path: str) -> Optional[List[dict[str, Any]]]:
    """Read one Pi transcript, rejecting the whole session on corruption.

    For tree-shaped v2/v3 sessions, silently skipping a malformed record could
    turn an older entry into the apparent active leaf.  Pi transcripts therefore
    use stricter parsing than the legacy shared harvester: blank lines are fine,
    but every non-blank line must be a JSON object and the complete file must be
    readable.
    """
    records: List[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    return None
                records.append(record)
    except (OSError, UnicodeError, ValueError):
        return None
    return records


def _pi_tool_names_from_content(content: Any) -> List[str]:
    """Extract tool names from pi content blocks.

    pi uses ``{"type": "toolCall", "name": ...}`` (cf. Claude's ``tool_use``).
    """
    names: List[str] = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "toolCall" and b.get("name"):
                names.append(_sanitize_tool_name(str(b["name"])))
    return names


def _active_branch_entries(records: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Return the linear history for v1 or the active leaf branch for v2/v3.

    Tree sessions are treated as an integrity boundary: a malformed graph is
    skipped in full instead of exposing an arbitrary reachable suffix to the
    optimizer.  Multiple roots remain valid because navigating back to the
    beginning of a Pi session can legitimately create one.
    """
    if (
        not records
        or any(not isinstance(rec, dict) for rec in records)
        or records[0].get("type") != "session"
    ):
        return []
    if sum(rec.get("type") == "session" for rec in records) != 1:
        return []

    header = records[0]
    entries = records[1:]
    header_id = header.get("id")
    if not isinstance(header_id, str) or not header_id:
        return []
    # Legacy v1 headers predate the version field; Pi itself interprets an
    # absent field as v1 and migrates those sessions on load.
    if "version" not in header:
        return entries
    version = header["version"]
    # ``bool`` is an ``int`` subclass in Python but is not a schema version.
    if type(version) is not int or version not in (1, 2, 3):
        return []
    if version == 1:
        return entries
    if not entries:
        return []

    by_id: dict[str, dict[str, Any]] = {}
    for rec in entries:
        entry_id = rec.get("id")
        if not isinstance(entry_id, str) or not entry_id or entry_id in by_id:
            return []
        if "parentId" not in rec:
            return []
        parent_id = rec["parentId"]
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                return []
            # Pi's writer is append-only: every child is appended after its
            # parent. Reject forward references rather than accepting a graph
            # that the official writer cannot produce.
            if parent_id not in by_id:
                return []
        by_id[entry_id] = rec

    branch: List[dict[str, Any]] = []
    seen: set[str] = set()
    current: Optional[dict[str, Any]] = entries[-1]

    while current is not None:
        entry_id = current["id"]
        if entry_id in seen:
            return []
        seen.add(entry_id)
        branch.append(current)

        parent_id = current["parentId"]
        if parent_id is None:
            break
        current = by_id[parent_id]

    branch.reverse()
    return branch


def _dedup(xs: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def digest_pi_session(path: str, project: str = "") -> Optional[SessionDigest]:
    """Build a :class:`SessionDigest` from one pi session transcript."""
    records = _read_pi_jsonl(path)
    if not records:
        return None
    active_entries = _active_branch_entries(records)
    session_id = os.path.splitext(os.path.basename(path))[0]
    started = ""
    ended = ""
    session_project = ""
    user_prompts: List[str] = []
    assistant_finals: List[str] = []
    tools: List[str] = []
    feedback: List[str] = []
    n_user = 0
    n_asst = 0

    header = records[0] if records[0].get("type") == "session" else None
    if isinstance(header, dict):
        ts = header.get("timestamp")
        if isinstance(ts, str) and ts:
            started = ts
            ended = ts
        cwd = header.get("cwd")
        if isinstance(cwd, str) and cwd:
            session_project = cwd

    for rec in active_entries:
        rtype = rec.get("type")
        ts = rec.get("timestamp")
        if isinstance(ts, str) and ts:
            if not started:
                started = ts
            ended = ts
        if rtype != "message":
            continue

        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            text = _text_from_content(content)
            text = _redact_secrets(text).strip()
            if text and not _is_meta_prompt(text):
                n_user += 1
                user_prompts.append(text)
                feedback.extend(_detect_feedback(text))
        elif role == "assistant":
            n_asst += 1
            tools.extend(_pi_tool_names_from_content(content))
            text = _text_from_content(content)
            if text.strip():
                assistant_finals.append(_redact_secrets(text).strip())
        elif role == "toolResult":
            # Corroborating tool-name source: pi records the resolved tool name
            # on the result, which catches calls even when the toolCall block's
            # `name` was absent. (toolName extraction only; see note below on isError.)
            tool_name = msg.get("toolName")
            if isinstance(tool_name, str) and tool_name:
                tools.append(_sanitize_tool_name(tool_name))
            # NOTE: pi also carries `isError` (bool) here — whether that one tool
            # invocation failed mechanically. We deliberately do NOT surface it
            # as a feedback signal: intermediate tool errors are normal in
            # agentic coding and are frequently followed by recovery and a
            # successful final result. Treating every recovered error as
            # `neg:` feedback would mislabel successful sessions as failures and
            # poison the miner's task-outcome labels. Task outcome should be
            # inferred from the user's judgment of the *final* result (the
            # lexical feedback phrases above), not from transient tool mechanics.

    if project and not _project_matches(session_project or "", "invoked", project):
        return None
    if n_user == 0 and n_asst == 0:
        return None

    digest = SessionDigest(
        session_id=session_id,
        project=session_project,
        started_at=started,
        ended_at=ended,
        user_prompts=user_prompts,
        assistant_finals=assistant_finals[-5:],
        tools_used=_dedup(tools),
        files_touched=[],  # not extractable from pi transcripts without heuristics
        feedback_signals=_dedup(feedback),
        n_user_turns=n_user,
        n_assistant_turns=n_asst,
        raw_path=path,
    )
    if _is_headless_replay(digest):
        return None
    return digest


def harvest_pi(
    sessions_dir: str,
    *,
    scope: Any = "all",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
) -> List[SessionDigest]:
    """Walk ``~/.pi/agent/sessions`` (one subdir per project slug) and return digests.

    Parameters mirror :func:`skillopt_sleep.harvest.harvest`.
    """
    digests: List[SessionDigest] = []
    if not os.path.isdir(sessions_dir):
        return digests

    paths: List[tuple[float, str]] = []
    for root, _dirs, files in os.walk(sessions_dir):
        for fn in files:
            if fn.endswith(".jsonl"):
                path = os.path.join(root, fn)
                try:
                    paths.append((os.path.getmtime(path), path))
                except OSError:
                    # A session can disappear while Pi rotates or cleans its
                    # store; skip that file without aborting the whole harvest.
                    continue
    paths.sort(key=lambda item: item[0], reverse=True)

    project_hint = invoked_project if scope == "invoked" else ""
    for _mtime, path in paths:
        digest = digest_pi_session(path, project=project_hint)
        if digest is None:
            continue
        if not _project_matches(digest.project or "", scope, invoked_project):
            continue
        if since_iso and digest.ended_at and _ended_before(digest.ended_at, since_iso):
            continue
        digests.append(digest)
        if limit and len(digests) >= limit:
            break
    return digests

"""Read Cursor Agent transcripts and normalize them into session digests.

Cursor writes workspace-scoped JSONL under
``~/.cursor/projects/<workspace>/agent-transcripts/<session>/<session>.jsonl``.
The observed local records contain user/assistant messages and tool-use metadata
but no timestamps, so this harvester uses each file's mtime as its end time.
Tool inputs and outputs are intentionally never copied.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional

from skillopt_sleep.harvest import (
    _detect_feedback,
    _is_meta_prompt,
    _iter_jsonl,
)
from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.types import SessionDigest

CURSOR_REPLAY_SENTINEL = "<skillopt_sleep_internal_replay_v1>"
_CURSOR_USER_QUERY_RE = re.compile(
    r"<user_query>\s*(.*?)\s*</user_query>\s*\Z",
    re.DOTALL,
)


def cursor_project_slug(project: str) -> str:
    """Return the filesystem-safe workspace name used under Cursor projects."""
    normalized = os.path.abspath(os.path.expanduser(project))
    return re.sub(r"[^A-Za-z0-9]+", "-", normalized).strip("-")


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block["text"])
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and block.get("text")
    )


def _tool_names(content: Any) -> List[str]:
    if not isinstance(content, list):
        return []
    names: List[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name")
        if isinstance(name, str) and name:
            names.append(re.sub(r"[^A-Za-z0-9_.:-]+", "_", name)[:80])
    return names


def _sanitize_text(text: str) -> str:
    sanitized = str(redact_secrets(text)).replace("\x00", "").strip()
    user_query = _CURSOR_USER_QUERY_RE.search(sanitized)
    if user_query:
        sanitized = user_query.group(1).strip()
    if not sanitized or _is_meta_prompt(sanitized):
        return ""
    return sanitized


def _dedup(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _mtime_iso(path: str) -> str:
    try:
        return (
            datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except OSError:
        return ""


def _mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _iso_epoch(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        # Existing state timestamps are local-time strings without an offset.
        return parsed.timestamp()
    except (TypeError, ValueError, OSError):
        return None


def digest_cursor_transcript(path: str, *, project: str = "") -> Optional[SessionDigest]:
    """Build a digest without retaining Cursor tool arguments or outputs."""
    session_id = os.path.splitext(os.path.basename(path))[0]
    user_prompts: List[str] = []
    assistant_finals: List[str] = []
    tools: List[str] = []
    feedback: List[str] = []
    n_user = 0
    n_assistant = 0

    for record in _iter_jsonl(path):
        if not isinstance(record, dict):
            continue
        if record.get("type") == "turn_ended":
            if record.get("status") == "error":
                feedback.append("neg:cursor_turn_error")
            continue

        message = record.get("message")
        if not isinstance(message, dict):
            continue
        role = record.get("role") or message.get("role")
        content = message.get("content")
        if role == "user":
            text = _sanitize_text(_text_from_content(content))
            if text:
                n_user += 1
                user_prompts.append(text)
                feedback.extend(_detect_feedback(text))
        elif role == "assistant":
            n_assistant += 1
            tools.extend(_tool_names(content))
            text = _sanitize_text(_text_from_content(content))
            if text:
                assistant_finals.append(text)

    if n_user == 0 and n_assistant == 0:
        return None

    return SessionDigest(
        session_id=session_id,
        project=project,
        ended_at=_mtime_iso(path),
        user_prompts=user_prompts,
        assistant_finals=assistant_finals[-5:],
        tools_used=_dedup(tools),
        files_touched=[],
        feedback_signals=feedback,
        n_user_turns=n_user,
        n_assistant_turns=n_assistant,
        raw_path=path,
    )


def _workspace_path(project_dir: str) -> str:
    metadata_path = os.path.join(project_dir, ".workspace-trusted")
    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, ValueError):
        return ""
    if not isinstance(metadata, dict):
        return ""
    workspace = metadata.get("workspacePath")
    if not isinstance(workspace, str) or not workspace.strip():
        return ""
    workspace = os.path.expanduser(workspace.strip())
    if not os.path.isabs(workspace):
        return ""
    return os.path.abspath(workspace)


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(path))))


def _is_workspace_ancestor(workspace: str, invoked: str) -> bool:
    try:
        workspace_norm = _normalized_path(workspace)
        invoked_norm = _normalized_path(invoked)
        return os.path.commonpath([workspace_norm, invoked_norm]) == workspace_norm
    except (OSError, ValueError):
        return False


def _available_project_dirs(projects_dir: str) -> List[tuple[str, str, str]]:
    try:
        names = sorted(os.listdir(projects_dir))
    except OSError:
        return []
    result: List[tuple[str, str, str]] = []
    for name in names:
        project_dir = os.path.join(projects_dir, name)
        if os.path.isdir(os.path.join(project_dir, "agent-transcripts")):
            result.append((project_dir, name, _workspace_path(project_dir)))
    return result


def _project_dirs(projects_dir: str, scope: Any, invoked_project: str) -> List[tuple[str, str]]:
    available = _available_project_dirs(projects_dir)
    if scope == "all":
        return [
            (project_dir, workspace or name)
            for project_dir, name, workspace in available
        ]

    projects: List[str]
    if isinstance(scope, (list, tuple)):
        projects = [str(project) for project in scope]
    else:
        projects = [invoked_project] if invoked_project else []

    selected: List[tuple[str, str]] = []
    seen = set()
    for project in projects:
        absolute_project = os.path.abspath(os.path.expanduser(project))
        matches = [
            (project_dir, workspace)
            for project_dir, _name, workspace in available
            if workspace and _is_workspace_ancestor(workspace, absolute_project)
        ]
        candidate = absolute_project
        while candidate:
            fallback = os.path.join(projects_dir, cursor_project_slug(candidate))
            if os.path.isdir(os.path.join(fallback, "agent-transcripts")):
                matches.append((fallback, candidate))
                break
            parent = os.path.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent

        if matches:
            longest = max(len(_normalized_path(workspace)) for _project_dir, workspace in matches)
            choices = [
                (project_dir, workspace)
                for project_dir, workspace in matches
                if len(_normalized_path(workspace)) == longest
            ]
        else:
            fallback = os.path.join(projects_dir, cursor_project_slug(absolute_project))
            choices = [(fallback, absolute_project)]
        for project_dir, workspace in choices:
            if project_dir not in seen:
                selected.append((project_dir, workspace))
                seen.add(project_dir)
    return selected


def _is_cursor_replay(digest: SessionDigest) -> bool:
    return any(prompt.lstrip().startswith(CURSOR_REPLAY_SENTINEL) for prompt in digest.user_prompts)


def harvest_cursor(
    projects_dir: str,
    *,
    scope: Any = "all",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
) -> List[SessionDigest]:
    """Return Cursor session digests for the selected workspace scope."""
    if not os.path.isdir(projects_dir):
        return []

    candidates: List[tuple[str, str, float]] = []
    for project_dir, project in _project_dirs(projects_dir, scope, invoked_project):
        transcripts_dir = os.path.join(project_dir, "agent-transcripts")
        try:
            session_names = sorted(os.listdir(transcripts_dir))
        except OSError:
            continue
        for session_name in session_names:
            session_dir = os.path.join(transcripts_dir, session_name)
            if not os.path.isdir(session_dir):
                continue
            try:
                filenames = sorted(os.listdir(session_dir))
            except OSError:
                continue
            for filename in filenames:
                path = os.path.join(session_dir, filename)
                if not filename.endswith(".jsonl") or not os.path.isfile(path):
                    continue
                modified = _mtime(path)
                if modified is not None:
                    candidates.append((path, project, modified))
    candidates.sort(key=lambda item: (-item[2], item[0]))

    since_epoch = _iso_epoch(since_iso)
    digests: List[SessionDigest] = []
    for path, project, modified in candidates:
        if since_epoch is not None and modified <= since_epoch:
            continue
        digest = digest_cursor_transcript(path, project=project)
        if digest is None or _is_cursor_replay(digest):
            continue
        digests.append(digest)
        if limit and len(digests) >= limit:
            break
    return digests

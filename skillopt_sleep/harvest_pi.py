"""SkillOpt-Sleep — pi (pi-coding-agent) session harvesting.

Reads pi session transcript JSONL files (one per session, stored under
``~/.pi/agent/sessions/<project-slug>/<sessionId>.jsonl``) and normalizes them
into :class:`SessionDigest` records without copying tool arguments, private
reasoning blocks (``thinking``), or raw tool outputs.

pi schema (verified against real transcripts):
  * A session file is a JSONL stream of entries with a ``type`` discriminator.
  * ``type == "session"``  — exactly one per file; carries ``cwd`` + ``timestamp``.
  * ``type == "message"``  — a conversational turn. ``message.role`` ∈
    {user, assistant, toolResult}; ``message.content`` is either a string or a
    list of content blocks. Block types include ``text`` (kept), ``thinking``
    (private reasoning, skipped), and ``toolCall`` (carries ``name``).
  * toolResult messages carry ``isError`` (bool) and ``toolName`` — a rare
    per-call success/failure signal, surfaced here as a feedback signal so the
    miner/gate can exploit checkable outcomes.
  * Other types (``model_change``, ``thinking_level_change``, ``custom``, ...) are
    metadata / tool-result payloads and are skipped for digestion.

This module performs NO writes and NO network calls.
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterable, List, Optional

from skillopt_sleep.harvest import (
    _detect_feedback,
    _is_headless_replay,
    _is_meta_prompt,
    _iter_jsonl,
    _project_matches,
    _text_from_content,
)
from skillopt_sleep.types import SessionDigest

# Mirror of skillopt_sleep.harvest_codex._SECRET_PATTERNS. Kept duplicated (not
# imported) so each harvester stays self-contained; if a third source appears,
# consider promoting these into a shared ``redact`` module.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_-]{10,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"(?i)(Authorization:\s*Bearer\s+)[^\s\"']+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(Authorization:\s*Basic\s+)[^\s\"']+"), r"\1[REDACTED]"),
    (
        re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\b(\s*[:=]\s*)[^\s\"']+"),
        r"\1\2[REDACTED]",
    ),
    (
        re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\b(\s+)[^\s\"']+"),
        r"\1\2[REDACTED]",
    ),
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
)


def _redact_secrets(text: str) -> str:
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _pi_tool_names_from_content(content: Any) -> List[str]:
    """Extract tool names from pi content blocks.

    pi uses ``{"type": "toolCall", "name": ...}`` (cf. Claude's ``tool_use``).
    """
    names: List[str] = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "toolCall" and b.get("name"):
                names.append(str(b["name"]))
    return names


def _sanitize_tool_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(name))[:80]


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

    for rec in _iter_jsonl(path):
        rtype = rec.get("type")
        ts = rec.get("timestamp")
        if isinstance(ts, str) and ts:
            if not started:
                started = ts
            ended = ts
        # cwd lives on the `session` entry, not on individual messages.
        if rtype == "session":
            cwd = rec.get("cwd")
            if isinstance(cwd, str) and cwd and not session_project:
                session_project = cwd
            continue
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

    paths: List[str] = []
    for root, _dirs, files in os.walk(sessions_dir):
        for fn in files:
            if fn.endswith(".jsonl"):
                paths.append(os.path.join(root, fn))
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    project_hint = invoked_project if scope == "invoked" else ""
    for path in paths:
        digest = digest_pi_session(path, project=project_hint)
        if digest is None:
            continue
        if not _project_matches(digest.project or "", scope, invoked_project):
            continue
        if since_iso and digest.ended_at and digest.ended_at < since_iso:
            continue
        digests.append(digest)
        if limit and len(digests) >= limit:
            break
    return digests

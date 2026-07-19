"""SkillOpt-Sleep — harvest Google Antigravity conversation stores.

Antigravity persists each conversation as a SQLite "trajectory" database in
``~/.gemini/antigravity/conversations/<uuid>.db``. The ``steps`` table holds
protobuf-encoded step payloads; without the proprietary schema we extract the
human-readable content with a conservative protobuf walker that collects
UTF-8 string fields:

  * step_type 14  -> user messages (the typed prompt, e.g. "/goal ...")
  * step_type  5  -> artifact/answer content the agent produced
  * step_type 33  -> tool calls (JSON with toolSummary/toolAction)

That is enough to build the same ``SessionDigest`` the Claude/Codex
harvesters produce: user prompts, assistant finals, tools used, feedback
signals. Databases may be locked by a live Antigravity process, so each file
is copied to a temp path before opening (read-only URI otherwise).

Heuristic by design: if Antigravity's schema changes, the walker degrades to
returning fewer strings — never to crashing the night (a session that yields
no user prompts is simply skipped, same as an empty transcript).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from typing import List, Optional

from skillopt_sleep.harvest import _detect_feedback, _is_meta_prompt
from skillopt_sleep.types import SessionDigest

DEFAULT_CONVERSATIONS_DIR = os.path.expanduser(
    "~/.gemini/antigravity/conversations")

_USER_STEP_TYPES = {14}
_ARTIFACT_STEP_TYPES = {5}
_TOOL_STEP_TYPES = {33}

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

# Antigravity injects a system wrapper around /goal tasks, and user steps also
# carry a permission-history of tool echoes like ``read_url(github.com)`` —
# neither is the user's own words.
_BOILERPLATE_MARKERS = (
    "marked this task with /goal",
    "The system will force you to continue",
)
_TOOL_ECHO_RE = re.compile(r"^[\w$.\\/-]+\([^()]*\)$")


# ── generic protobuf string extraction ────────────────────────────────────────

def _read_varint(buf: bytes, i: int):
    val = 0
    shift = 0
    n = len(buf)
    while i < n:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return val, i
        if shift > 63:
            break
    return None, i


def _proto_strings(buf: bytes, depth: int = 0, out: Optional[List[str]] = None) -> List[str]:
    """Collect plausible UTF-8 string fields from a protobuf blob (schema-less)."""
    if out is None:
        out = []
    if depth > 6 or len(out) > 400:
        return out
    i, n = 0, len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        if tag is None:
            break
        wire = tag & 7
        if wire == 0:
            _v, i = _read_varint(buf, i)
            if _v is None:
                break
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        elif wire == 2:
            ln, i = _read_varint(buf, i)
            if ln is None or ln < 0 or i + ln > n:
                break
            chunk = buf[i:i + ln]
            i += ln
            text = None
            try:
                text = chunk.decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text is not None and len(text) >= 16 and _looks_natural(text):
                out.append(text)
            else:
                # possibly a nested message — recurse; a failed walk just
                # contributes nothing
                _proto_strings(chunk, depth + 1, out)
        else:  # unknown/deprecated wire types: bail out of this blob
            break
    return out


def _looks_natural(text: str) -> bool:
    """Keep human/markdown text; drop ids, uuids, base64 runs, file URIs."""
    t = text.strip()
    if not t or _UUID_RE.match(t):
        return False
    if t.startswith(("file:///", "http://", "https://")) and " " not in t:
        return False
    if " " not in t and len(t) > 40:  # long spaceless token: id/base64
        return False
    letters = sum(c.isalpha() or c.isspace() for c in t)
    return letters / max(1, len(t)) > 0.55


# ── per-database digestion ────────────────────────────────────────────────────

def _clean_user_prompt(text: str) -> str:
    t = text.strip()
    for prefix in ("/goal ", "/task ", "/ask "):
        if t.lower().startswith(prefix):
            t = t[len(prefix):]
    return t.strip()


def _digest_db(path: str, project: str) -> Optional[SessionDigest]:
    tmp = os.path.join(tempfile.gettempdir(),
                       f"skillopt_agy_{os.path.basename(path)}")
    try:
        shutil.copy2(path, tmp)
    except OSError:
        return None
    try:
        con = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT idx, step_type, step_payload FROM steps ORDER BY idx"
        ).fetchall()
        con.close()
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    prompts: List[str] = []
    finals: List[str] = []
    tools: List[str] = []
    for _idx, stype, payload in rows:
        blob = payload if isinstance(payload, bytes) else str(payload or "").encode()
        if not blob:
            continue
        if stype in _USER_STEP_TYPES:
            strs = [
                s for s in _proto_strings(blob)
                if not s.startswith("{")
                and not any(m in s for m in _BOILERPLATE_MARKERS)
                and not _TOOL_ECHO_RE.match(s.strip())
            ]
            if strs:
                p = _clean_user_prompt(max(strs, key=len))
                if p and not _is_meta_prompt(p):
                    prompts.append(p)
        elif stype in _ARTIFACT_STEP_TYPES:
            strs = _proto_strings(blob)
            # prefer the artifact body over its ArtifactMetadata JSON envelope
            body = [s for s in strs if not s.lstrip().startswith("{")]
            if body or strs:
                finals.append(max(body or strs, key=len))
        elif stype in _TOOL_STEP_TYPES:
            for s in _proto_strings(blob):
                if s.startswith("{"):
                    try:
                        obj = json.loads(s)
                        name = obj.get("toolSummary") or obj.get("toolAction")
                        if name and name not in tools:
                            tools.append(str(name))
                    except Exception:
                        pass

    if not prompts:
        return None
    mtime = os.path.getmtime(path)
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime))
    feedback = _detect_feedback(" \n".join(prompts))
    return SessionDigest(
        session_id=os.path.splitext(os.path.basename(path))[0],
        project=project,
        started_at=iso, ended_at=iso,
        user_prompts=prompts,
        assistant_finals=finals[-3:],
        tools_used=tools[:12],
        feedback_signals=feedback,
    )


def harvest_antigravity(
    conversations_dir: str = "",
    *,
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
) -> List[SessionDigest]:
    """Digest the most recent Antigravity conversations (newest first)."""
    root = os.path.expanduser(conversations_dir or DEFAULT_CONVERSATIONS_DIR)
    if not os.path.isdir(root):
        return []
    dbs = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".db")]
    dbs.sort(key=os.path.getmtime, reverse=True)
    out: List[SessionDigest] = []
    for path in dbs:
        if limit and len(out) >= limit:
            break
        if since_iso:
            mtime_iso = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path)))
            if mtime_iso < since_iso:
                continue
        d = _digest_db(path, project=invoked_project or "antigravity")
        if d is not None:
            out.append(d)
    return out

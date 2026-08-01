"""Read Google Antigravity conversations and normalize them into digests.

Antigravity persists each conversation as a SQLite "trajectory" database in
``~/.gemini/antigravity/conversations/<uuid>.db``. The ``steps`` table holds
protobuf-encoded step payloads; without the proprietary schema we extract the
human-readable content with a conservative protobuf walker that collects
UTF-8 string fields:

  * step_type 14  -> user messages (the typed prompt, e.g. "/goal ...")
  * step_type  5  -> artifact/answer content the agent produced
  * step_type 33  -> tool calls (JSON with toolSummary/toolAction)

Project provenance comes from the ``trajectory_metadata_blob`` row, which
records the workspace the conversation was opened against as a ``file://``
URI (field 7, mirrored in field 1.1) plus the git branch (field 1.4).
Conversations started outside any workspace carry an explicit
``outside-of-project`` marker (field 18) instead. Sessions whose workspace
cannot be established are *never* relabelled as the invoked project: they are
only harvested under an explicit ``projects: "all"`` opt-in, and even then
keep an empty project rather than borrowing the caller's identity.

Databases are read through SQLite's online backup API into a private
temporary directory. A plain file copy is not sufficient — Antigravity keeps
these databases in WAL mode with a live writer attached, so the ``.db`` file
on its own can be missing recent commits or the schema entirely.

Heuristic by design: if Antigravity's schema changes, the walker degrades to
returning fewer strings — never to crashing the night (a session that yields
no user prompts is simply skipped, same as an empty transcript).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from skillopt_sleep.harvest import _detect_feedback, _is_meta_prompt
from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.types import SessionDigest

_USER_STEP_TYPES = {14}
_ARTIFACT_STEP_TYPES = {5}
_TOOL_STEP_TYPES = {33}

# trajectory_metadata_blob protobuf fields (empirically stable across the
# observed corpus; every lookup degrades to "unknown" if they move).
_META_WORKSPACE_FIELD = 7        # "file:///..." workspace root
_META_REPO_FIELD = 1             # submessage: .1 workspace uri, .4 git branch
_META_REPO_URI_SUBFIELD = 1
_META_REPO_BRANCH_SUBFIELD = 4
_META_CREATED_FIELD = 2          # submessage: .1 epoch seconds
_META_CREATED_SECONDS_SUBFIELD = 1
_META_PROJECT_MARKER_FIELD = 18  # "outside-of-project" when there is no workspace
_OUTSIDE_OF_PROJECT = "outside-of-project"

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

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

_READ_TIMEOUT = 5.0
_MAX_STRINGS_PER_BLOB = 400
_MAX_WALK_DEPTH = 6
_MIN_STRING_LEN = 16


# ── generic protobuf decoding ─────────────────────────────────────────────────

def _read_varint(buf: bytes, i: int) -> Tuple[Optional[int], int]:
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


def _proto_fields(buf: bytes) -> Dict[int, Any]:
    """Decode one protobuf level into {field_number: value}, first wins.

    Length-delimited fields yield ``bytes``; varints yield ``int``. Unknown
    wire types stop the scan rather than raising — a truncated or reshaped
    message simply contributes the fields decoded so far.
    """
    out: Dict[int, Any] = {}
    i, n = 0, len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        if tag is None:
            break
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _read_varint(buf, i)
            if val is None:
                break
            out.setdefault(field, val)
        elif wire == 2:
            ln, i = _read_varint(buf, i)
            if ln is None or ln < 0 or i + ln > n:
                break
            out.setdefault(field, buf[i:i + ln])
            i += ln
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        else:
            break
    return out


def _proto_strings(buf: bytes, depth: int = 0, out: Optional[List[str]] = None) -> List[str]:
    """Collect plausible UTF-8 string fields from a protobuf blob (schema-less)."""
    if out is None:
        out = []
    if depth > _MAX_WALK_DEPTH or len(out) > _MAX_STRINGS_PER_BLOB:
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
            try:
                text: Optional[str] = chunk.decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text is not None and len(text) >= _MIN_STRING_LEN and _looks_natural(text):
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
    # A nested message often decodes as valid UTF-8, which would otherwise
    # yield the inner text with its protobuf framing bytes glued to the front
    # ("\x12\x30Refactor the ..."). Real prose carries no control bytes, so
    # rejecting them here sends the chunk back through the walker instead.
    if _CONTROL_CHARS_RE.search(t):
        return False
    if t.startswith(("file:///", "http://", "https://")) and " " not in t:
        return False
    if " " not in t and len(t) > 40:  # long spaceless token: id/base64
        return False
    letters = sum(c.isalpha() or c.isspace() for c in t)
    return letters / max(1, len(t)) > 0.55


# ── path / provenance helpers ─────────────────────────────────────────────────

def _path_from_file_uri(uri: str) -> str:
    """Convert ``file:///C:/a%20b`` to a native path, cross-platform.

    ``urllib.request.url2pathname`` is platform-dependent, which would make the
    same database resolve differently on POSIX and Windows; the recorded URI is
    always absolute, so decode it directly instead.
    """
    if not uri.startswith("file://"):
        return ""
    rest = uri[len("file://"):]
    if not rest.startswith("/"):  # file://host/path — host form is not supported
        slash = rest.find("/")
        if slash < 0:
            return ""
        rest = rest[slash:]
    path = urllib.parse.unquote(rest)
    if re.match(r"^/[A-Za-z]:([/\\]|$)", path):  # /C:/... -> C:/...
        path = path[1:]
    return os.path.normpath(path) if path else ""


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.expanduser(path)))


def _is_workspace_ancestor(workspace: str, invoked: str) -> bool:
    """True when ``invoked`` is ``workspace`` or lives inside it."""
    if not workspace or not invoked:
        return False
    try:
        workspace_norm = _normalized_path(workspace)
        invoked_norm = _normalized_path(invoked)
        return os.path.commonpath([workspace_norm, invoked_norm]) == workspace_norm
    except (OSError, ValueError):  # different drives / unrelated roots
        return False


def _iso_utc(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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
        return datetime.fromisoformat(normalized).timestamp()
    except (TypeError, ValueError, OSError):
        return None


# ── snapshotting ──────────────────────────────────────────────────────────────

def _query(con: sqlite3.Connection) -> Dict[str, List[Any]]:
    """Pull the rows we need out of an open conversation database."""
    rows: Dict[str, List[Any]] = {
        "steps": con.execute(
            "SELECT idx, step_type, step_payload FROM steps ORDER BY idx"
        ).fetchall(),
    }
    try:
        rows["meta"] = con.execute(
            "SELECT data FROM trajectory_metadata_blob"
        ).fetchall()
    except sqlite3.Error:
        rows["meta"] = []  # older/reshaped stores: provenance stays unknown
    return rows


def _read_direct(path: str) -> Dict[str, List[Any]]:
    """Read through a read-only URI connection (raises on lock/corruption).

    SQLite attaches the ``-wal`` sidecar itself, so this read sees every
    committed transaction. Copying the bare ``.db`` — the previous approach —
    does not: with a live writer attached the copy can miss recent commits, or
    the schema entirely.
    """
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=_READ_TIMEOUT)
    try:
        return _query(con)
    finally:
        try:
            con.close()
        except sqlite3.Error:
            pass


def _read_via_backup(path: str, tmpdir: str) -> Dict[str, List[Any]]:
    """Read through SQLite's online backup API into a private temp directory.

    The fallback for databases that cannot be opened directly (an exclusive
    lock, or a read-only volume with no usable ``-shm``). The snapshot lands in
    the caller's ``TemporaryDirectory`` — never a predictable path — and is
    removed as soon as the rows are read.
    """
    snapshot = os.path.join(tmpdir, "snapshot.db")
    source = destination = None
    try:
        source = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=_READ_TIMEOUT)
        destination = sqlite3.connect(snapshot)
        source.backup(destination)
        return _query(destination)
    finally:
        for con in (destination, source):
            if con is not None:
                try:
                    con.close()
                except sqlite3.Error:
                    pass
        try:
            if os.path.exists(snapshot):
                os.unlink(snapshot)
        except OSError:
            pass


def _read_rows(path: str, tmpdir: str) -> Optional[Dict[str, List[Any]]]:
    """Read one conversation database consistently, without copying the file.

    Returns None when the database is locked beyond the timeout, corrupt, or
    not a database at all — a bad store skips its session instead of failing
    the night.
    """
    try:
        return _read_direct(path)
    except sqlite3.Error:
        pass
    try:
        return _read_via_backup(path, tmpdir)
    except sqlite3.Error:
        return None


def _provenance(meta_rows: List[Any]) -> Tuple[str, str, Optional[float]]:
    """Extract (workspace_path, git_branch, created_epoch) from the metadata row.

    ``workspace_path`` is "" both for conversations Antigravity marked
    ``outside-of-project`` and for stores whose provenance cannot be read; the
    caller treats the two identically (never attributable to a project).
    """
    for row in meta_rows or []:
        blob = row[0] if isinstance(row, (tuple, list)) else row
        if not isinstance(blob, (bytes, bytearray)):
            continue
        fields = _proto_fields(bytes(blob))

        marker = fields.get(_META_PROJECT_MARKER_FIELD)
        if isinstance(marker, bytes) and marker == _OUTSIDE_OF_PROJECT.encode():
            return "", "", _created_epoch(fields)

        uri = fields.get(_META_WORKSPACE_FIELD)
        branch = ""
        repo = fields.get(_META_REPO_FIELD)
        if isinstance(repo, bytes):
            inner = _proto_fields(repo)
            if not isinstance(uri, bytes):
                uri = inner.get(_META_REPO_URI_SUBFIELD)
            raw_branch = inner.get(_META_REPO_BRANCH_SUBFIELD)
            if isinstance(raw_branch, bytes):
                try:
                    branch = raw_branch.decode("utf-8")
                except UnicodeDecodeError:
                    branch = ""

        workspace = ""
        if isinstance(uri, bytes):
            try:
                workspace = _path_from_file_uri(uri.decode("utf-8"))
            except UnicodeDecodeError:
                workspace = ""

        return workspace, branch, _created_epoch(fields)
    return "", "", None


def _created_epoch(fields: Dict[int, Any]) -> Optional[float]:
    """Conversation start time, when the metadata carries a sane timestamp."""
    stamp = fields.get(_META_CREATED_FIELD)
    if isinstance(stamp, bytes):
        seconds = _proto_fields(stamp).get(_META_CREATED_SECONDS_SUBFIELD)
        if isinstance(seconds, int) and 0 < seconds < 4_102_444_800:  # < year 2100
            return float(seconds)
    return None


# ── per-database digestion ────────────────────────────────────────────────────

def _clean_user_prompt(text: str) -> str:
    t = text.strip()
    for prefix in ("/goal ", "/task ", "/ask "):
        if t.lower().startswith(prefix):
            t = t[len(prefix):]
    return t.strip()


def _sanitize(text: str) -> str:
    """Redact secrets and strip NULs before the text leaves the harvester.

    Every prompt, final and tool name passes through here, so nothing reaches
    the evidence log or a model-processing path un-redacted.
    """
    return str(redact_secrets(text)).replace("\x00", "").strip()


def _digest_with_provenance(
    path: str, tmpdir: str
) -> Optional[Tuple[SessionDigest, str]]:
    """Digest one database and report the workspace Antigravity recorded.

    Returns ``(digest, workspace)`` where ``workspace`` is "" when the
    conversation was started outside any project or its provenance could not be
    read. The database is snapshotted exactly once.
    """
    rows = _read_rows(path, tmpdir)
    if rows is None:
        return None

    workspace, branch, created = _provenance(rows.get("meta", []))

    prompts: List[str] = []
    finals: List[str] = []
    tools: List[str] = []
    for _idx, stype, payload in rows.get("steps", []):
        if isinstance(payload, (bytes, bytearray)):
            blob = bytes(payload)
        elif payload is None:
            continue
        else:
            blob = str(payload).encode("utf-8", "replace")
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
                p = _sanitize(_clean_user_prompt(max(strs, key=len)))
                if p and not _is_meta_prompt(p):
                    prompts.append(p)
        elif stype in _ARTIFACT_STEP_TYPES:
            strs = _proto_strings(blob)
            # prefer the artifact body over its ArtifactMetadata JSON envelope
            body = [s for s in strs if not s.lstrip().startswith("{")]
            if body or strs:
                final = _sanitize(max(body or strs, key=len))
                if final:
                    finals.append(final)
        elif stype in _TOOL_STEP_TYPES:
            for s in _proto_strings(blob):
                if not s.startswith("{"):
                    continue
                try:
                    obj = json.loads(s)
                except ValueError:
                    continue
                if not isinstance(obj, dict):
                    continue
                name = obj.get("toolSummary") or obj.get("toolAction")
                if not name:
                    continue
                clean = re.sub(r"[^A-Za-z0-9_.:-]+", "_", _sanitize(str(name)))[:80]
                if clean and clean not in tools:
                    tools.append(clean)

    if not prompts:
        return None

    modified = _mtime(path)
    ended = _iso_utc(modified) if modified is not None else ""
    started = _iso_utc(created) if created is not None else ended
    digest = SessionDigest(
        session_id=os.path.splitext(os.path.basename(path))[0],
        project="",
        git_branch=branch,
        started_at=started,
        ended_at=ended,
        user_prompts=prompts,
        assistant_finals=finals[-3:],
        tools_used=tools[:12],
        feedback_signals=_detect_feedback(" \n".join(prompts)),
        n_user_turns=len(prompts),
        n_assistant_turns=len(finals),
        raw_path=path,
    )
    return digest, workspace


def digest_antigravity_db(
    path: str, *, project: str = "", tmpdir: Optional[str] = None
) -> Optional[SessionDigest]:
    """Digest one conversation database, or None if it yields no user prompts.

    ``project`` overrides the label; when omitted the digest carries the
    workspace Antigravity recorded for the conversation.
    """
    if tmpdir is None:
        with tempfile.TemporaryDirectory(prefix="skillopt-agy-") as owned:
            return digest_antigravity_db(path, project=project, tmpdir=owned)
    result = _digest_with_provenance(path, tmpdir)
    if result is None:
        return None
    digest, workspace = result
    digest.project = project or workspace
    return digest


# ── scope selection ───────────────────────────────────────────────────────────

def _selected_projects(scope: Any, invoked_project: str) -> List[str]:
    if isinstance(scope, (list, tuple)):
        return [str(p) for p in scope if str(p).strip()]
    return [invoked_project] if invoked_project else []


def harvest_antigravity(
    conversations_dir: str,
    *,
    scope: Any = "invoked",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
) -> List[SessionDigest]:
    """Return Antigravity session digests for the selected workspace scope.

    ``scope="all"`` harvests every conversation, labelling each with the
    workspace Antigravity recorded (empty when the conversation was started
    outside a workspace). Any other scope keeps only conversations whose
    recorded workspace contains one of the selected projects; conversations
    with unknown or out-of-project provenance are skipped rather than being
    attributed to the invoked project.
    """
    if not os.path.isdir(conversations_dir):
        return []

    try:
        names = sorted(os.listdir(conversations_dir))
    except OSError:
        return []

    candidates: List[Tuple[str, float]] = []
    for name in names:
        if not name.endswith(".db"):
            continue
        path = os.path.join(conversations_dir, name)
        if not os.path.isfile(path):
            continue
        modified = _mtime(path)
        if modified is not None:
            candidates.append((path, modified))
    candidates.sort(key=lambda item: (-item[1], item[0]))

    since_epoch = _iso_epoch(since_iso)
    wanted = _selected_projects(scope, invoked_project)
    harvest_all = scope == "all"
    if not harvest_all and not wanted:
        return []  # nothing to scope against; never fall back to "everything"

    digests: List[SessionDigest] = []
    with tempfile.TemporaryDirectory(prefix="skillopt-agy-") as tmpdir:
        for path, modified in candidates:
            if since_epoch is not None and modified <= since_epoch:
                continue
            result = _digest_with_provenance(path, tmpdir)
            if result is None:
                continue
            digest, workspace = result
            if not harvest_all and not any(
                _is_workspace_ancestor(workspace, project) for project in wanted
            ):
                # Unknown, outside-of-project, or a different workspace: skip it
                # rather than relabelling it as the invoked project.
                continue
            digest.project = workspace
            digests.append(digest)
            if limit and len(digests) >= limit:
                break
    return digests

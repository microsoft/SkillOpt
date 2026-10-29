"""Read DeepSeek Harness JSONL session logs into ``SessionDigest`` records.

The DSH JSONL persistence backend stores one append-only event log per session.
This reader is deliberately read-only and privacy-bounded: it keeps human user
text, visible assistant text, short tool names, timestamps, and derived
positive/negative feedback signals.  It never persists reasoning, tool
arguments/results, request metadata, or feedback remarks themselves.

Malformed sessions are silently discarded as a whole.  DSH event sequences are
integrity-sensitive, so salvaging a suffix after a bad record could produce a
misleading conversation.  A bad file must not prevent other sessions from
being harvested.  When DSH retains multiple immutable format generations in
one session directory, only the numerically highest generation is considered;
an unsupported highest generation is never replaced by an older predecessor.
"""
from __future__ import annotations

import io
import json
import logging
import ntpath
import os
import posixpath
import re
import sys
from collections import deque
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional

from skillopt_sleep.harvest import _detect_feedback, _is_meta_prompt, _project_matches
from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.types import SessionDigest

_LOGGER = logging.getLogger("skillopt_sleep.harvest_dsh")
_LOG_NAME_RE = re.compile(r"^session(?:\.v([1-9][0-9]*))?\.jsonl(?:\.zstd)?$")
# DSH's current writer is v2.  v0/v1 are retained historical generations and
# use the same validated logical event boundary here.  A future generation is
# selected first and then discarded if it is outside this set.
_SUPPORTED_FORMAT_VERSIONS = frozenset({0, 1, 2})
_V2_HEADER_REQUIRED_KEYS = frozenset(
    {"type", "version", "id", "createdAt", "isSeeded", "delegationDepth"}
)
_V2_HEADER_ALLOWED_KEYS = _V2_HEADER_REQUIRED_KEYS | frozenset(
    {"cwd", "parentSession", "origin", "agentPreset"}
)
_PACKED_TYPES = {"text-chunks", "reasoning-chunks", "tool-call-chunks"}
_SURFACE_EVENT_TYPES = {"user/message", "assistant/message", "tool/result"}
_KNOWN_EVENT_TYPES = {
    # Current DSH lifecycle and presentation metadata. These records advance
    # the durable sequence but never contribute transcript content.
    "permission/preset",
    "sandbox/mode",
    "approval/policy",
    "approval/asked",
    "approval/decided",
    "agent/inbox/spliced",
    "turn/start",
    "turn/end",
    "step/start",
    "step/end",
    "session/title",
    "session/title-llm-request",
    "user/message",
    "assistant/chunk",
    "assistant/message",
    "assistant/attempt",
    "tool/call",
    "tool/result",
    "request/header",
    "request/context",
    "session/end-seed",
    "feedback/record",
    "web/deepseek-search-llm-request",
}
_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_.:-]+")

# There is no DSH replay producer in this change.  This stable, namespaced
# marker is reserved for a future producer so its sessions never feed the next
# harvest cycle.  Do not infer replay from ordinary natural-language prompts.
DSH_REPLAY_SENTINEL = "<skillopt_sleep_dsh_replay_v1>"


class _DshFormatError(ValueError):
    """Internal sentinel used to discard one invalid session silently."""


def _is_absolute_dsh_path(value: str) -> bool:
    """Recognize the POSIX and Windows absolute paths DSH can persist."""
    return posixpath.isabs(value) or ntpath.isabs(value)


@dataclass(frozen=True, slots=True)
class _SurfaceEntry:
    """Only digest content and the identity needed by later replacements."""

    seq: int
    kind: str
    text: str = ""
    tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Replacement:
    start: int
    end: int
    # Legacy replacements do not require provenance coverage.
    sources: Optional[frozenset[int]] = None


@dataclass(frozen=True, slots=True)
class _DshEvent:
    """Validated event projection; never holds a raw message or tool payload."""

    next_seq: int
    time: int
    surface: Optional[_SurfaceEntry] = None
    replacement: Optional[_Replacement] = None
    tool: str = ""
    feedback: tuple[str, ...] = ()
    inherited_seed_marker: bool = False


def _diagnostic(message: str, *args: object, progress: bool = False) -> None:
    rendered = message % args if args else message
    _LOGGER.debug("%s", rendered)
    if progress:
        print(f"[sleep] dsh: {rendered}", file=sys.stderr, flush=True)


def _parse_log_filename(filename: str) -> Optional[tuple[int, str]]:
    """Return ``(generation, encoding)`` for a canonical DSH log filename."""
    match = _LOG_NAME_RE.fullmatch(filename)
    if match is None:
        return None
    return (int(match.group(1) or 0), "zstd" if filename.endswith(".zstd") else "raw")


def _is_safe_int(value: Any) -> bool:
    return type(value) is int and 0 <= value <= 9_007_199_254_740_991


def _sanitize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        text = str(redact_secrets(value)).replace("\x00", "").strip()
    except Exception:
        return ""
    return "" if not text else text


def _sanitize_tool_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    return _TOOL_NAME_RE.sub("_", value)[:80]


def _dedup(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _iso_timestamp(value: Any) -> str:
    """Turn a DSH epoch-millisecond timestamp into a stable ISO string."""
    if not _is_safe_int(value):
        return ""
    try:
        return (
            datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (OverflowError, OSError, ValueError):
        return ""


def _iso_epoch(value: Optional[str]) -> Optional[float]:
    """Compare instants, interpreting offset-free Sleep checkpoints locally."""
    if not value:
        return None
    try:
        normalized = value[:-1] + "+00:00" if value[-1:] in {"Z", "z"} else value
        return datetime.fromisoformat(normalized).timestamp()
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def _utf16_units(value: str) -> Iterator[int]:
    raw = value.encode("utf-16-le", "surrogatepass")
    for offset in range(0, len(raw), 2):
        yield int.from_bytes(raw[offset : offset + 2], "little")


def _encode_segment(value: str) -> str:
    """Mirror DSH's injective safe-path encoding for ordinary Python strings."""
    if not value:
        raise _DshFormatError("empty segment")
    if value == ".":
        return "~002E"
    if value == "..":
        return "~002E~002E"
    pieces: list[str] = []
    for code in _utf16_units(value):
        char = chr(code)
        if char != "~" and (char.isascii() and (char.isalnum() or char in "._-")):
            pieces.append(char)
        else:
            pieces.append(f"~{code:04X}")
    return "".join(pieces)


def _project_key(cwd: str) -> str:
    """Mirror DSH's readable, intentionally lossy project directory key."""
    if not cwd:
        raise _DshFormatError("empty cwd")
    pieces: list[str] = []
    separator_run = False
    for code in _utf16_units(cwd):
        char = chr(code)
        if char in "/\\:":
            if not separator_run:
                pieces.append("-")
            separator_run = True
        elif char != "~" and (char.isascii() and (char.isalnum() or char in "._-")):
            pieces.append(char)
            separator_run = False
        else:
            pieces.append(f"~{code:04X}")
            separator_run = False
    slug = "".join(pieces).lstrip("-") or "root"
    return f"--{slug[:251]}--"


def _is_within(root: str, candidate: str) -> bool:
    try:
        return os.path.commonpath([root, candidate]) == root
    except ValueError:
        return False


def _is_candidate_path(root: str, path: str) -> bool:
    """Accept only DSH's fixed root/project/session/log layout."""
    if _parse_log_filename(os.path.basename(path)) is None:
        return False
    real_path = os.path.realpath(path)
    if not _is_within(root, real_path):
        return False
    try:
        parts = os.path.relpath(real_path, root).split(os.sep)
    except ValueError:
        return False
    return len(parts) == 3 and _parse_log_filename(parts[-1]) is not None


def _iter_plain_lines(path: str) -> Iterator[str]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            yield from handle
    except (OSError, UnicodeError) as exc:
        raise _DshFormatError("unreadable raw log") from exc


def _iter_zstd_lines(path: str) -> Iterator[str]:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise _DshFormatError("zstandard unavailable") from exc
    try:
        with open(path, "rb") as source:
            decoder = zstd.ZstdDecompressor()
            with decoder.stream_reader(source, read_across_frames=True) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8", newline="") as text:
                    yield from text
    except (OSError, UnicodeError, zstd.ZstdError, ValueError) as exc:
        raise _DshFormatError("unreadable zstd log") from exc


def _iter_records(path: str) -> Iterator[dict[str, Any]]:
    lines = _iter_zstd_lines(path) if path.endswith(".zstd") else _iter_plain_lines(path)
    saw_record = False
    with closing(lines):
        for line in lines:
            if not line.strip():
                continue
            # A DSH writer terminates every committed JSONL record.  Do not use
            # a possibly torn final line as a session event.
            if not line.endswith(("\n", "\r")):
                raise _DshFormatError("unterminated record")
            try:
                record = json.loads(line)
            except (TypeError, ValueError) as exc:
                raise _DshFormatError("invalid JSON record") from exc
            if not isinstance(record, dict):
                raise _DshFormatError("non-object record")
            saw_record = True
            yield record
    if not saw_record:
        raise _DshFormatError("empty session")


def _header_from_record(record: dict[str, Any], path: str, root: str) -> dict[str, Any]:
    if record.get("type") != "session":
        raise _DshFormatError("missing header")
    parsed_name = _parse_log_filename(os.path.basename(path))
    if parsed_name is None:
        raise _DshFormatError("non-canonical log name")
    filename_version, _encoding = parsed_name
    version = record.get("version")
    if type(version) is not int or version != filename_version or version not in _SUPPORTED_FORMAT_VERSIONS:
        raise _DshFormatError("unsupported format")
    if version >= 2 and (
        not _V2_HEADER_REQUIRED_KEYS.issubset(record) or not set(record).issubset(_V2_HEADER_ALLOWED_KEYS)
    ):
        raise _DshFormatError("invalid v2 header keys")
    session_id = record.get("id")
    created = record.get("createdAt")
    depth = record.get("delegationDepth")
    if not isinstance(session_id, str) or not session_id or not _is_safe_int(created) or not _is_safe_int(depth):
        raise _DshFormatError("invalid header")
    cwd = record.get("cwd")
    if cwd is not None and (not isinstance(cwd, str) or not cwd):
        raise _DshFormatError("invalid cwd")
    if version >= 2 and cwd is not None and not _is_absolute_dsh_path(cwd):
        raise _DshFormatError("invalid v2 cwd")
    parent = record.get("parentSession")
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise _DshFormatError("invalid parent session")
    if record.get("origin") not in {None, "subagent"}:
        raise _DshFormatError("invalid origin")
    if record.get("isSeeded") is not None and not isinstance(record.get("isSeeded"), bool):
        raise _DshFormatError("invalid seeded flag")
    if record.get("agentPreset") is not None and not isinstance(record.get("agentPreset"), str):
        raise _DshFormatError("invalid agent preset")
    seed_length = record.get("seedLength")
    if seed_length is not None and not _is_safe_int(seed_length):
        raise _DshFormatError("invalid seed length")
    if "sandboxMode" in record or "approvalPolicy" in record:
        raise _DshFormatError("retired header field")

    session_dir = os.path.dirname(path)
    project_dir = os.path.dirname(session_dir)
    expected_project = "_no-cwd" if cwd is None else _project_key(cwd)
    if os.path.normcase(os.path.basename(session_dir)) != os.path.normcase(_encode_segment(session_id)):
        raise _DshFormatError("session path mismatch")
    if os.path.normcase(os.path.basename(project_dir)) != os.path.normcase(expected_project):
        raise _DshFormatError("project path mismatch")
    if not _is_within(root, os.path.realpath(path)):
        raise _DshFormatError("path outside root")
    return record


def _packed_count(record: dict[str, Any]) -> int:
    row_type = record.get("type")
    if row_type not in _PACKED_TYPES or not _is_safe_int(record.get("seq0")) or not _is_safe_int(record.get("time0")):
        raise _DshFormatError("invalid packed row")
    data = record.get("data")
    if not isinstance(data, dict):
        raise _DshFormatError("invalid packed data")
    for key in ("turn", "step", "index"):
        if not _is_safe_int(data.get(key)):
            raise _DshFormatError("invalid packed position")
    values = data.get("texts") if row_type in {"text-chunks", "reasoning-chunks"} else data.get("args")
    if not isinstance(values, list) or len(values) < 3 or any(not isinstance(value, str) for value in values):
        raise _DshFormatError("invalid packed members")
    if row_type == "tool-call-chunks":
        if not isinstance(data.get("callId"), str) or not data.get("callId"):
            raise _DshFormatError("invalid packed tool call")
        if data.get("name") is not None and not isinstance(data.get("name"), str):
            raise _DshFormatError("invalid packed tool name")
    deltas = data.get("dt")
    if not isinstance(deltas, list) or any(not _is_safe_int(delta) for delta in deltas):
        raise _DshFormatError("invalid packed timing")
    # Current DSH writes a leading zero delta for the first member.  Accept the
    # equivalent n-1 representation too, because both reconstruct the same
    # event stream and older logs may omit that redundant first zero.
    if len(deltas) == len(values):
        if deltas[0] != 0:
            raise _DshFormatError("invalid packed first delta")
    elif len(deltas) != len(values) - 1:
        raise _DshFormatError("invalid packed delta count")
    return len(values)


def _text_blocks(content: Any) -> list[str]:
    if isinstance(content, str):
        return [_sanitize_text(content)]
    if not isinstance(content, list):
        return []
    return [
        _sanitize_text(block.get("text"))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def _tool_names_from_content(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    return [
        _sanitize_tool_name(block.get("name"))
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool-call"
    ]


def _human_user_text(record: dict[str, Any]) -> str:
    data = record["data"]
    if data.get("role") != "user":
        raise _DshFormatError("invalid user message")
    source = data.get("source")
    if not isinstance(source, dict) or source.get("kind") != "user":
        return ""
    text = "\n".join(part for part in _text_blocks(data.get("content")) if part).strip()
    return "" if _is_meta_prompt(text) else text


def _assistant_message(record: dict[str, Any]) -> tuple[str, list[str]]:
    data = record["data"]
    message = data.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise _DshFormatError("invalid assistant message")
    text = "\n".join(part for part in _text_blocks(message.get("content")) if part).strip()
    return text, _tool_names_from_content(message.get("content"))


def _tool_call_name(record: dict[str, Any]) -> str:
    data = record["data"]
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise _DshFormatError("invalid tool call")
    return _sanitize_tool_name(name)


def _feedback_signals(record: dict[str, Any]) -> list[str]:
    text = _sanitize_text(record["data"].get("text"))
    return _detect_feedback(text) if text else []


def _decode_source_event_seqs(value: Any, *, max_entries: int) -> list[int]:
    """Expand DSH v2's compact ``number | [start, end]`` sequence ranges."""
    if not isinstance(value, list):
        raise _DshFormatError("invalid surface sources")
    decoded: list[int] = []
    has_range = False
    for item in value:
        if _is_safe_int(item):
            if len(decoded) >= max_entries:
                raise _DshFormatError("too many surface sources")
            decoded.append(item)
            continue
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not _is_safe_int(item[0])
            or not _is_safe_int(item[1])
            or item[0] > item[1]
        ):
            raise _DshFormatError("invalid surface sources")
        start, end = item
        count = end - start + 1
        if count > max_entries - len(decoded):
            raise _DshFormatError("too many surface sources")
        decoded.extend(range(start, end + 1))
        has_range = True
    if has_range and any(later <= earlier for earlier, later in zip(decoded, decoded[1:])):
        raise _DshFormatError("non-increasing surface sources")
    if len(set(decoded)) != len(decoded):
        raise _DshFormatError("duplicate surface sources")
    return decoded


def _surface_replacement(record: dict[str, Any], version: int) -> Optional[_Replacement]:
    """Validate wire-format surface rules and normalize append/replace."""
    operation = record.get("surfaceOp")
    if operation is None:
        if version >= 2:
            raise _DshFormatError("missing v2 surface operation")
        operation = "append"

    # Preserve the currently supported v2 assistant contract explicitly.
    # Broadening this to replacement requires an upstream format sample.
    if version >= 2 and record["type"] == "assistant/message":
        if operation != "append" or record.get("sourceEventSeqs") is not None:
            raise _DshFormatError("v2 assistant surface must append without sources")

    source_seqs = record.get("sourceEventSeqs")
    if source_seqs is not None:
        source_seqs = (
            _decode_source_event_seqs(source_seqs, max_entries=record["seq"])
            if version >= 2
            else source_seqs
        )
        if (
            not isinstance(source_seqs, list)
            or any(not _is_safe_int(seq) for seq in source_seqs)
            or len(set(source_seqs)) != len(source_seqs)
        ):
            raise _DshFormatError("invalid surface sources")
    if operation == "append":
        return None
    if not isinstance(operation, dict) or operation.get("op") != "replace":
        raise _DshFormatError("invalid surface operation")
    start, end = operation.get("start"), operation.get("end")
    if not _is_safe_int(start) or not _is_safe_int(end):
        raise _DshFormatError("invalid surface replacement range")
    if version >= 2 and not source_seqs:
        raise _DshFormatError("missing v2 surface sources")
    return _Replacement(start, end, frozenset(source_seqs) if version >= 2 else None)


def _decode_event(record: dict[str, Any], expected_seq: int, version: int) -> _DshEvent:
    """Validate sequencing and harvest-relevant fields, then project safe data."""
    kind = record.get("type")
    if not isinstance(kind, str) or not kind:
        raise _DshFormatError("invalid event type")
    if kind in _PACKED_TYPES:
        if version >= 2:
            raise _DshFormatError("packed row in v2 session")
        count = _packed_count(record)
        if record["seq0"] != expected_seq:
            raise _DshFormatError("packed sequence gap")
        end_time = record["time0"] + sum(record["data"]["dt"])
        if not _is_safe_int(end_time) or not _is_safe_int(expected_seq + count - 1):
            raise _DshFormatError("packed event overflow")
        return _DshEvent(expected_seq + count, end_time)

    # Ignorable extensions have exactly the same envelope and sequence rules.
    if not _is_safe_int(record.get("seq")) or record["seq"] != expected_seq:
        raise _DshFormatError("non-contiguous sequence")
    if not _is_safe_int(record.get("time")) or not isinstance(record.get("data"), dict):
        raise _DshFormatError("invalid event envelope")
    if kind not in _KNOWN_EVENT_TYPES and record.get("ignorable") is not True:
        raise _DshFormatError("unknown event")

    next_seq, timestamp = expected_seq + 1, record["time"]
    if kind in _SURFACE_EVENT_TYPES:
        replacement = _surface_replacement(record, version)
        text, names = "", []
        if kind == "user/message":
            text = _human_user_text(record)
        elif kind == "assistant/message":
            text, names = _assistant_message(record)
        # Tool results and injected context still occupy the surface, but their
        # raw content is unnecessary even when later replacements target them.
        entry = _SurfaceEntry(expected_seq, kind, text, tuple(_dedup(names)))
        return _DshEvent(next_seq, timestamp, entry, replacement)
    if kind == "tool/call":
        return _DshEvent(next_seq, timestamp, tool=_tool_call_name(record))
    if kind == "feedback/record":
        return _DshEvent(next_seq, timestamp, feedback=tuple(_feedback_signals(record)))
    if kind == "session/end-seed":
        inherited = record["data"].get("inherited")
        if inherited is not None and inherited is not True:
            raise _DshFormatError("invalid inherited seed marker")
        return _DshEvent(next_seq, timestamp, inherited_seed_marker=inherited is True)
    return _DshEvent(next_seq, timestamp)


def _apply_surface(
    surface: list[_SurfaceEntry], entry: _SurfaceEntry, replacement: Optional[_Replacement],
) -> None:
    """Apply one normalized update without retaining raw or shadowed records."""
    if replacement is None:
        surface.append(entry)
        return
    try:
        start = next(index for index, item in enumerate(surface) if item.seq == replacement.start)
        end = next(index for index, item in enumerate(surface) if item.seq == replacement.end)
    except StopIteration as exc:
        raise _DshFormatError("surface replacement range is not visible") from exc
    if start > end:
        raise _DshFormatError("invalid surface replacement range")
    if replacement.sources is not None:
        if any(surface[index].seq not in replacement.sources for index in range(start, end + 1)):
            raise _DshFormatError("incomplete v2 surface sources")
    surface[start : end + 1] = [entry]


def _is_dsh_replay(digest: SessionDigest) -> bool:
    return bool(digest.user_prompts) and digest.user_prompts[0].lstrip().startswith(DSH_REPLAY_SENTINEL)


def digest_dsh_session(
    path: str,
    *,
    root: str,
    progress: bool = False,
    scope: Any = "all",
    invoked_project: str = "",
) -> Optional[SessionDigest]:
    """Parse one complete DSH session file, returning ``None`` on any failure."""
    records = _iter_records(path)
    try:
        header = _header_from_record(next(records), path, root)
        if header.get("origin") == "subagent":
            return None

        session_id = str(header["id"])
        project = str(header.get("cwd") or "")
        if (not project and scope != "all") or not _project_matches(project, scope, invoked_project):
            return None
        started_at = _iso_timestamp(header["createdAt"])
        ended_at = started_at
        user_prompts: list[str] = []
        assistant_finals: deque[str] = deque(maxlen=5)
        tools: dict[str, None] = {}
        feedback: dict[str, None] = {}
        surface: list[_SurfaceEntry] = []
        expected_seq = 0
        n_user = 0
        n_assistant = 0
        has_inherited_seed_marker = False

        for record in records:
            event = _decode_event(record, expected_seq, header["version"])
            expected_seq = event.next_seq
            ended_at = _iso_timestamp(event.time)
            if event.surface is not None:
                _apply_surface(surface, event.surface, event.replacement)
            if event.tool:
                tools[event.tool] = None
            feedback.update(dict.fromkeys(event.feedback))
            has_inherited_seed_marker = has_inherited_seed_marker or event.inherited_seed_marker

        if header["version"] >= 2 and bool(header["isSeeded"]) != has_inherited_seed_marker:
            raise _DshFormatError("v2 seeded header and inherited seed marker disagree")

        for entry in surface:
            if entry.kind == "user/message":
                text = entry.text
                if text:
                    user_prompts.append(text)
                    feedback.update(dict.fromkeys(_detect_feedback(text)))
                    n_user += 1
            elif entry.kind == "assistant/message":
                tools.update(dict.fromkeys(entry.tools))
                n_assistant += 1
                if entry.text:
                    assistant_finals.append(entry.text)
        if not user_prompts and not assistant_finals:
            return None

        digest = SessionDigest(
            session_id=session_id,
            project=project,
            started_at=started_at,
            ended_at=ended_at,
            user_prompts=user_prompts,
            assistant_finals=list(assistant_finals),
            tools_used=list(tools),
            files_touched=[],
            feedback_signals=list(feedback),
            n_user_turns=n_user,
            n_assistant_turns=n_assistant,
            raw_path=path,
        )
        return None if _is_dsh_replay(digest) else digest
    except (OSError, StopIteration, ValueError, TypeError) as exc:
        _diagnostic("Skipping DSH session file %s: %s", path, exc, progress=progress)
        return None
    finally:
        records.close()


def harvest_dsh(
    session_root: str,
    *,
    scope: Any = "all",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
    progress: bool = False,
) -> list[SessionDigest]:
    """Discover valid DSH session logs below one explicitly supplied root."""
    if not session_root:
        return []
    root = os.path.realpath(os.path.abspath(os.path.expanduser(session_root)))
    if not os.path.isdir(root):
        return []

    grouped: dict[str, list[tuple[int, str, str, float]]] = {}
    for directory, _dirs, files in os.walk(root, followlinks=False):
        for filename in files:
            parsed_name = _parse_log_filename(filename)
            if parsed_name is None:
                continue
            path = os.path.join(directory, filename)
            if not _is_candidate_path(root, path):
                continue
            try:
                version, encoding = parsed_name
                session_dir = os.path.dirname(path)
                grouped.setdefault(session_dir, []).append(
                    (version, encoding, path, os.path.getmtime(path))
                )
            except OSError:
                continue

    candidates: list[tuple[float, str]] = []
    for session_dir, entries in grouped.items():
        encodings = {entry[1] for entry in entries}
        if len(encodings) != 1:
            _diagnostic(
                "Skipping DSH session directory %s: mixed log encodings",
                session_dir,
                progress=progress,
            )
            continue
        highest_version = max(entry[0] for entry in entries)
        highest = [entry for entry in entries if entry[0] == highest_version]
        # A canonical session directory has at most one file for a generation
        # and encoding.  Ambiguous duplicates are safer to skip than guess at.
        if len(highest) != 1:
            _diagnostic(
                "Skipping DSH session directory %s: ambiguous files for highest generation v%d",
                session_dir,
                highest_version,
                progress=progress,
            )
            continue
        _version, _encoding, path, mtime = highest[0]
        if highest_version not in _SUPPORTED_FORMAT_VERSIONS:
            _diagnostic(
                "Skipping DSH session directory %s: selected file %s has unsupported highest "
                "generation v%d; supported generations are %s. Upgrade SkillOpt to support "
                "this DSH format.",
                session_dir,
                path,
                highest_version,
                ", ".join(str(version) for version in sorted(_SUPPORTED_FORMAT_VERSIONS)),
                progress=progress,
            )
            continue
        _diagnostic(
            "Selected DSH session file %s (highest generation v%d)",
            path,
            highest_version,
            progress=progress,
        )
        candidates.append((mtime, path))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    digests: list[SessionDigest] = []
    seen_ids: set[str] = set()
    since_epoch = _iso_epoch(since_iso)
    for _mtime, path in candidates:
        digest = digest_dsh_session(
            path, root=root, progress=progress, scope=scope, invoked_project=invoked_project,
        )
        if digest is None or digest.session_id in seen_ids:
            continue
        seen_ids.add(digest.session_id)
        if since_iso and digest.ended_at:
            ended_epoch = _iso_epoch(digest.ended_at)
            if since_epoch is not None and ended_epoch is not None:
                if ended_epoch < since_epoch:
                    continue
            elif digest.ended_at < since_iso:
                # Preserve best-effort behavior for malformed legacy cutoffs.
                continue
        digests.append(digest)
        if limit and len(digests) >= limit:
            break
    return digests

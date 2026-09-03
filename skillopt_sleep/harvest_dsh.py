"""Read DeepSeek Harness JSONL session logs into ``SessionDigest`` records.

The DSH JSONL persistence backend stores one append-only event log per session.
This reader is deliberately read-only and privacy-bounded: it keeps human user
text, visible assistant text, short tool names, timestamps, and derived
positive/negative feedback signals.  It never persists reasoning, tool
arguments/results, request metadata, or feedback remarks themselves.

Malformed sessions are silently discarded as a whole.  DSH event sequences are
integrity-sensitive, so salvaging a suffix after a bad record could produce a
misleading conversation.  A bad file must not prevent other sessions from
being harvested.
"""
from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional

from skillopt_sleep.harvest import _detect_feedback, _is_meta_prompt, _project_matches
from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.types import SessionDigest

_LOG_NAMES = {"session.jsonl", "session.jsonl.zstd"}
_PACKED_TYPES = {"text-chunks", "reasoning-chunks", "tool-call-chunks"}
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
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (OverflowError, OSError, ValueError):
        return ""


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
    if os.path.basename(path) not in _LOG_NAMES:
        return False
    real_path = os.path.realpath(path)
    if not _is_within(root, real_path):
        return False
    try:
        parts = os.path.relpath(real_path, root).split(os.sep)
    except ValueError:
        return False
    return len(parts) == 3 and parts[-1] in _LOG_NAMES


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
    for line in lines:
        if not line.strip():
            continue
        # A DSH writer terminates every committed JSONL record.  Do not use a
        # possibly torn final line as a session event.
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
    version = record.get("version")
    if version != 0:
        raise _DshFormatError("unsupported format")
    session_id = record.get("id")
    created = record.get("createdAt")
    depth = record.get("delegationDepth")
    if not isinstance(session_id, str) or not session_id or not _is_safe_int(created) or not _is_safe_int(depth):
        raise _DshFormatError("invalid header")
    cwd = record.get("cwd")
    if cwd is not None and (not isinstance(cwd, str) or not cwd):
        raise _DshFormatError("invalid cwd")
    parent = record.get("parentSession")
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise _DshFormatError("invalid parent session")
    if record.get("origin") not in {None, "subagent"}:
        raise _DshFormatError("invalid origin")
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


def _validate_event(record: dict[str, Any], expected_seq: int) -> None:
    event_type = record.get("type")
    if not isinstance(event_type, str) or event_type not in _KNOWN_EVENT_TYPES:
        if record.get("ignorable") is True:
            return
        raise _DshFormatError("unknown event")
    if record.get("seq") != expected_seq or not _is_safe_int(record.get("seq")):
        raise _DshFormatError("non-contiguous sequence")
    if not _is_safe_int(record.get("time")) or not isinstance(record.get("data"), dict):
        raise _DshFormatError("invalid event envelope")


def _is_append_surface(record: dict[str, Any]) -> bool:
    operation = record.get("surfaceOp")
    return operation is None or operation == "append"


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
    if not _is_append_surface(record):
        return ""
    data = record["data"]
    if data.get("role") != "user":
        raise _DshFormatError("invalid user message")
    source = data.get("source")
    if not isinstance(source, dict) or source.get("kind") != "user":
        return ""
    text = "\n".join(part for part in _text_blocks(data.get("content")) if part).strip()
    return "" if _is_meta_prompt(text) else text


def _assistant_message(record: dict[str, Any]) -> tuple[str, list[str]]:
    if not _is_append_surface(record):
        return "", []
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


def _is_dsh_replay(digest: SessionDigest) -> bool:
    return bool(digest.user_prompts) and digest.user_prompts[0].lstrip().startswith(DSH_REPLAY_SENTINEL)


def digest_dsh_session(path: str, *, root: str) -> Optional[SessionDigest]:
    """Parse one complete DSH session file, returning ``None`` on any failure."""
    try:
        records = _iter_records(path)
        header = _header_from_record(next(records), path, root)
        if header.get("origin") == "subagent":
            return None

        session_id = str(header["id"])
        project = str(header.get("cwd") or "")
        started_at = _iso_timestamp(header["createdAt"])
        ended_at = started_at
        user_prompts: list[str] = []
        assistant_finals: list[str] = []
        tools: list[str] = []
        feedback: list[str] = []
        expected_seq = 0
        n_user = 0
        n_assistant = 0

        for record in records:
            row_type = record.get("type")
            if row_type in _PACKED_TYPES:
                count = _packed_count(record)
                if record["seq0"] != expected_seq:
                    raise _DshFormatError("packed sequence gap")
                expected_seq += count
                # Packed chunks are intentionally not retained, but their final
                # timestamp is still the best session-end timestamp.
                deltas = record["data"]["dt"]
                ended_at = _iso_timestamp(record["time0"] + sum(deltas))
                continue

            _validate_event(record, expected_seq)
            if record.get("type") in _KNOWN_EVENT_TYPES:
                expected_seq += 1
                ended_at = _iso_timestamp(record["time"])
            # An ignorable extension has a normal event envelope and therefore
            # still occupies one sequence number.
            elif record.get("ignorable") is True:
                if record.get("seq") != expected_seq or not _is_safe_int(record.get("time")):
                    raise _DshFormatError("invalid ignorable event")
                expected_seq += 1
                ended_at = _iso_timestamp(record["time"])

            event_type = record.get("type")
            if event_type == "user/message":
                text = _human_user_text(record)
                if text:
                    user_prompts.append(text)
                    feedback.extend(_detect_feedback(text))
                    n_user += 1
            elif event_type == "assistant/message":
                text, names = _assistant_message(record)
                tools.extend(names)
                n_assistant += 1
                if text:
                    assistant_finals.append(text)
            elif event_type == "tool/call":
                tools.append(_tool_call_name(record))
            elif event_type == "feedback/record":
                feedback.extend(_feedback_signals(record))

        if not user_prompts and not assistant_finals:
            return None

        digest = SessionDigest(
            session_id=session_id,
            project=project,
            started_at=started_at,
            ended_at=ended_at,
            user_prompts=user_prompts,
            assistant_finals=assistant_finals[-5:],
            tools_used=_dedup(tools),
            files_touched=[],
            feedback_signals=_dedup(feedback),
            n_user_turns=n_user,
            n_assistant_turns=n_assistant,
            raw_path=path,
        )
        return None if _is_dsh_replay(digest) else digest
    except (OSError, StopIteration, _DshFormatError, ValueError, TypeError, json.JSONDecodeError):
        return None


def harvest_dsh(
    session_root: str,
    *,
    scope: Any = "all",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
) -> list[SessionDigest]:
    """Discover valid DSH session logs below one explicitly supplied root."""
    if not session_root:
        return []
    root = os.path.realpath(os.path.abspath(os.path.expanduser(session_root)))
    if not os.path.isdir(root):
        return []

    candidates: list[tuple[float, str]] = []
    for directory, _dirs, files in os.walk(root, followlinks=False):
        for filename in files:
            if filename not in _LOG_NAMES:
                continue
            path = os.path.join(directory, filename)
            if not _is_candidate_path(root, path):
                continue
            try:
                candidates.append((os.path.getmtime(path), path))
            except OSError:
                continue
    candidates.sort(key=lambda item: (-item[0], item[1]))

    digests: list[SessionDigest] = []
    seen_ids: set[str] = set()
    for _mtime, path in candidates:
        digest = digest_dsh_session(path, root=root)
        if digest is None or digest.session_id in seen_ids:
            continue
        seen_ids.add(digest.session_id)
        if not digest.project and scope != "all":
            continue
        if not _project_matches(digest.project, scope, invoked_project):
            continue
        if since_iso and digest.ended_at and digest.ended_at < since_iso:
            continue
        digests.append(digest)
        if limit and len(digests) >= limit:
            break
    return digests

"""SkillOpt-Sleep — Stage 5/6: staging and adoption.

Implements the Dreams safety contract: the cycle never mutates the user's
live CLAUDE.md / SKILL.md. It writes proposals + a human-readable report into
a staging directory; a separate, explicit `adopt` step copies them over the
live files after taking a backup.
"""
from __future__ import annotations

import base64
import getpass
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from skillopt_sleep.types import SleepReport

# A secret value may be quoted, braced (ODBC-style), or an unquoted scalar.
# Accept EOF as the terminator for quoted/braced values because diagnostics are
# often truncated precisely where a failing client was printing a credential.
# Doubled quote/brace characters are the escape convention used by SQL/ODBC.
_UNQUOTED_SECRET_VALUE = (
    r'''(?:[^\s"';&,)\]}]|[)\]}]+(?=[^\s"';&,)\]}]))+'''
)
_SECRET_VALUE = (
    r'''(?:"(?:\\(?:[^\r\n]|(?=\r?\n|$))|""|[^"\\\r\n])*'''
    r'''(?:"|(?=\r?\n|$))'''
    r'''|'(?:\\(?:[^\r\n]|(?=\r?\n|$))|''|[^'\\\r\n])*'''
    r'''(?:'|(?=\r?\n|$))'''
    r'''|\{(?:\\(?:[^\r\n]|(?=\r?\n|$))|}}|[^}\\\r\n])*'''
    r'''(?:}|(?=\r?\n|$))'''
    r'''|''' + _UNQUOTED_SECRET_VALUE + r''')'''
)

# Match both short labels (``token=``) and environment/connection-string names
# whose final component identifies a credential (``AZURE_CLIENT_SECRET=``).
_SECRET_NAME_BODY = (
    r"(?:(?:[A-Za-z0-9]+[_-])*(?:"
    r"api[_-]?key|access[_-]?token|refresh[_-]?token|token|"
    r"password|passwd|secret|secret[_-]?key|secret[_-]?access[_-]?key|"
    r"shared[_-]?access[_-]?key|private[_-]?key"
    r")|[A-Za-z0-9]*(?:"
    r"apikey|accesstoken|refreshtoken|clientsecret|secretkey|"
    r"secretaccesskey|sharedaccesskey|privatekey"
    r"))"
)
_SECRET_ASSIGNMENT_NAME = (
    r"(?<![A-Za-z0-9])"
    r"(" + _SECRET_NAME_BODY + r")"
    r"(?![A-Za-z0-9])"
)

_JSON_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?P<prefix>(?<![A-Za-z0-9])(?P<key_quote>[\"'])"
    + r"(?:" + _SECRET_NAME_BODY + r"|pwd|accountkey)"
    + r"(?P=key_quote)\s*:\s*)"
    + r"(?P<value>" + _SECRET_VALUE + r")"
)

_REDACTED_MARKER = re.compile(r"^\[REDACTED(?:_[A-Z_]+)?\]$")
_SECRET_MAPPING_KEY_SUFFIXES = (
    "apikey",
    "accesstoken",
    "refreshtoken",
    "token",
    "password",
    "passwd",
    "clientsecret",
    "secret",
    "secretkey",
    "secretaccesskey",
    "sharedaccesskey",
    "privatekey",
    "accountkey",
)


def _redact_json_assignment(match: re.Match[str]) -> str:
    """Keep JSON-like value quotes while replacing their complete contents."""
    value = match.group("value")
    quote = (
        value[:1] if value[:1] in {'"', "'"} else match.group("key_quote")
    )
    return f"{match.group('prefix')}{quote}[REDACTED]{quote}"


def _is_secret_mapping_key(key: Any) -> bool:
    """Recognize credential-bearing dict keys without flagging token budgets."""
    if not isinstance(key, str):
        return False
    stripped = key.strip()
    # PWD is conventionally the non-secret process working directory. Mixed or
    # lower-case ``Pwd`` remains a common database-password field.
    if stripped == "PWD":
        return False
    compact = re.sub(r"[^a-z0-9]", "", stripped.casefold())
    return compact in {"pwd", "sig", "authorization"} or compact.endswith(
        _SECRET_MAPPING_KEY_SUFFIXES
    )

# Secret patterns scrubbed from any free-text we persist to the staging dir
# (diagnostics, reports). Kept here so every on-disk artifact shares one
# redaction pass; harvest_codex reuses these for session text too.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_-]{10,}"), "[REDACTED_OPENAI_KEY]"),
    # Distinctive vendor token prefixes (low false-positive: these prefixes do
    # not occur in normal diagnostic prose).
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "[REDACTED_GOOGLE_KEY]"),
    # Bare JWT (three base64url segments) — e.g. a leaked bearer body without
    # the "Authorization:" prefix.
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
     "[REDACTED_JWT]"),
    (
        re.compile(
            r'''(?i)(Authorization:\s*Bearer\s+)'''
            r'''(?!\[REDACTED(?:_[A-Z_]+)?\])'''
            + _SECRET_VALUE
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r'''(?i)(Authorization:\s*Basic\s+)'''
            r'''(?!\[REDACTED(?:_[A-Z_]+)?\])'''
            + _SECRET_VALUE
        ),
        r"\1[REDACTED]",
    ),
    # Connection-string passwords. Handle quoted values (which may contain
    # semicolons) before the generic name=value rule below, and retain the key
    # plus all non-secret connection-string fields for useful diagnostics.
    (
        re.compile(
            r'''(?i)(\bPassword\s*=\s*)'''
            r'''(?!\[REDACTED(?:_[A-Z_]+)?\])'''
            + _SECRET_VALUE
        ),
        r"\1[REDACTED_DB_PASS]",
    ),
    # ODBC commonly abbreviates Password as Pwd. Keep the conventional
    # all-uppercase PWD working-directory variable intact.
    (
        re.compile(
            r"((?<![A-Za-z0-9])(?:Pwd|pwd)\s*=\s*)"
            r"(?!\[REDACTED(?:_[A-Z_]+)?\])"
            + _SECRET_VALUE
        ),
        r"\1[REDACTED_DB_PASS]",
    ),
    # Upper-case PWD is normally a process working-directory variable, but
    # after a semicolon it is the canonical ODBC connection-string password.
    (
        re.compile(
            r"((?<=;)\s*PWD\s*=\s*)"
            r"(?!\[REDACTED(?:_[A-Z_]+)?\])"
            + _SECRET_VALUE
        ),
        r"\1[REDACTED_DB_PASS]",
    ),
    (
        re.compile(
            r"(?i)" + _SECRET_ASSIGNMENT_NAME
            + r"(\s*[:=]\s*)(?!\[REDACTED(?:_[A-Z_]+)?\])"
            + _SECRET_VALUE
        ),
        r"\1\2[REDACTED]",
    ),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|token|password|secret)\b(\s+)"
            r"(?!\[REDACTED(?:_[A-Z_]+)?\])"
            r"(?=[^\s\"';&,)\]}]{6,}(?:[\s,;&\"')\]}]|$))"
            r"(?:(?=[^\s\"';&,)\]}]*(?:\d|[_./+=:@-]))"
            r"|(?=[A-Za-z]{16,}(?:[\s,;&\"')\]}]|$)))"
            r"[^\s\"';&,)\]}]+"
        ),
        r"\1\2[REDACTED]",
    ),
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    # Azure SAS tokens (URL query param: ?sig=<base64>&...)
    (
        re.compile(r"(?i)(\bsig\s*=\s*)[A-Za-z0-9%+/]{10,}"),
        r"\1[REDACTED_SAS_SIG]",
    ),
    # Azure Storage account keys (base64, typically 88 chars)
    (
        re.compile(
            r'''(?i)(\bAccountKey\s*=\s*)'''
            r'''(?!\[REDACTED(?:_[A-Z_]+)?\])'''
            + _SECRET_VALUE
        ),
        r"\1[REDACTED_STORAGE_KEY]",
    ),
)


def redact_secrets(value: Any) -> Any:
    """Scrub secret-looking substrings (API keys, bearer tokens, private keys)
    from a string, or recursively from the string leaves of a list/dict.

    Used before writing backend stderr / optimizer replies / task responses to
    on-disk diagnostics: those are surfaced for debugging, but the underlying
    text (e.g. a codex 401 stderr dump) can carry credentials. Non-string
    scalars pass through unchanged.
    """
    if isinstance(value, str):
        out = _JSON_SECRET_ASSIGNMENT.sub(_redact_json_assignment, value)
        for pattern, replacement in _SECRET_PATTERNS:
            out = pattern.sub(replacement, out)
        return out
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if _is_secret_mapping_key(key):
                if isinstance(item, str) and _REDACTED_MARKER.fullmatch(item):
                    redacted[key] = item
                else:
                    redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    return value


def json_safe(value: Any) -> Any:
    """Replace non-finite floats recursively so persisted JSON stays standard."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


class StagingError(ValueError):
    """A proposal could not be staged safely (bad name, bad target, collision)."""


class StagingRecoveryError(StagingError):
    """A transaction failed and could not be rolled back without data loss."""

    def __init__(
        self,
        message: str,
        *,
        primary: Optional[BaseException] = None,
        recovery_errors: Sequence[str] = (),
    ) -> None:
        self.primary = primary
        self.recovery_errors = tuple(recovery_errors)
        detail = "; ".join(self.recovery_errors)
        super().__init__(f"{message}: {detail}" if detail else message)


@dataclass
class SkillProposal:
    """One skill's proposed document plus the live file it would replace."""

    skill_name: str
    proposed_skill: str
    live_skill_path: str
    # ``None`` lets low-level callers snapshot the baseline at staging time.
    # The cycle supplies the exact raw-byte hash it read before consolidation,
    # closing the otherwise-unchecked read -> model call -> staging window.
    live_sha256: Optional[str] = None
    live_realpath: str = ""


def _filesystem_key(value: str) -> str:
    """Conservative collision key for case/normalisation-insensitive filesystems."""
    return unicodedata.normalize("NFC", value).casefold()


def _path_identity_key(value: str) -> str:
    """Platform path identity key; distinct POSIX spellings stay distinct."""
    return os.path.normcase(os.path.normpath(value))


def _is_link_or_junction(path: str) -> bool:
    isjunction = getattr(os.path, "isjunction", None)
    if os.path.islink(path) or bool(isjunction and isjunction(path)):
        return True
    if os.name == "nt" and os.path.lexists(path):
        try:
            return bool(getattr(os.lstat(path), "st_reparse_tag", 0))
        except OSError:
            return True
    return False


def _safe_skill_name(name: object) -> str:
    """Return a skill name usable as a single path segment, else ""."""
    if not isinstance(name, str):
        return ""
    candidate = name.strip()
    if not candidate or candidate in {os.curdir, os.pardir}:
        return ""
    if candidate.startswith("~") or os.path.isabs(candidate):
        return ""
    if os.path.splitdrive(candidate)[0]:
        return ""
    separators = {"/", "\\", os.sep, os.altsep or os.sep}
    if any(sep in candidate for sep in separators):
        return ""
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in candidate):
        return ""
    # The name becomes a filename, so reject what Windows cannot store. Without
    # this the write fails with an OSError from deep inside staging instead of
    # a StagingError naming the offending skill.
    if any(ch in candidate for ch in ':*?"<>|'):
        return ""
    if candidate[-1] in {".", " "}:
        return ""
    return candidate


def _safe_live_path(path: object) -> str:
    """Return an absolute, traversal-free ``*.md`` target path, else ""."""
    if not isinstance(path, str) or not path.strip():
        return ""
    raw = path.strip()
    if raw.startswith("~"):
        return ""
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        return ""
    # Reject traversal on the RAW input, before normalising. Normalising first
    # would silently resolve "/live/../../etc/SKILL.md" into "/etc/SKILL.md"
    # and then accept it, because no ".." survives the collapse -- turning a
    # traversal guard into a traversal helper.
    if any(part == os.pardir for part in raw.replace("\\", "/").split("/")):
        return ""
    # Only then normalise, so a caller is not forced to hand over an already
    # canonical string. The old form demanded input == normpath(input), which
    # rejected benign duplicate separators and every forward-slash absolute
    # path on Windows (normpath rewrites those to backslashes, so a safe path
    # never matched itself).
    candidate = os.path.normpath(raw)
    if not os.path.isabs(candidate):
        return ""
    if not candidate.endswith(".md"):
        return ""
    return candidate


def _live_target_within_roots(live: str, roots: Iterable[str]) -> bool:
    """Return True when ``live`` resolves inside one of ``roots``.

    ``_safe_live_path`` only proves a target is absolute, traversal-free and
    ``*.md``; it accepts any such path on the machine. Containment is a separate
    question and the manifest is not a trust boundary -- a tampered
    ``live_skill_path`` with self-consistent pins otherwise redirects an adopt
    onto an arbitrary file. Callers pass the roots recorded when the night was
    staged and must run this AFTER the existing ``realpath(live) == live``
    identity check, so a symlinked ancestor cannot make an outside target look
    contained.
    """
    real_live = os.path.realpath(live)
    for root in roots:
        if not isinstance(root, str) or not root.strip():
            continue
        real_root = os.path.realpath(os.path.abspath(os.path.expanduser(root)))
        if _path_is_within(real_live, real_root):
            return True
    return False


def proposal_filename(skill_name: str) -> str:
    """Staged filename for one skill's proposal (unique per skill name)."""
    return f"proposed_SKILL.{skill_name}.md"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_live_path(path: str) -> str:
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return ""


def _sha256_file_bytes(path: str) -> str:
    """Hash a regular live file's raw bytes; ``""`` means it is absent."""
    data, _mode, _file_id = _file_snapshot(path)
    return _bytes_sha256(data)


def _fsync_directory(path: str) -> None:
    """Durably publish directory-entry changes where the platform supports it."""
    if os.name == "nt":
        # Python exposes no portable way to open a Windows directory for
        # FlushFileBuffers. The WAL makes recovery deterministic there; POSIX
        # additionally gets rename/unlink durability through directory fsync.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path or ".", flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_parent(path: str) -> None:
    _fsync_directory(os.path.dirname(path) or ".")


def _unlink_fsync(path: str) -> None:
    os.unlink(path)
    _fsync_parent(path)


def _write_atomic_bytes(
    path: str,
    data: bytes,
    *,
    create_parents: bool = True,
    mode: Optional[int] = None,
) -> None:
    """Write raw bytes atomically, optionally restoring an exact file mode."""
    directory = os.path.dirname(path) or "."
    if create_parents:
        os.makedirs(directory, exist_ok=True)
    existing_mode = mode
    if existing_mode is None and os.path.exists(path):
        existing_mode = stat.S_IMODE(os.stat(path).st_mode)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".md")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            if existing_mode is not None:
                if hasattr(os, "fchmod"):
                    os.fchmod(f.fileno(), existing_mode)
                else:
                    os.chmod(tmp, existing_mode)
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_parent(path)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            # Preserve the operation's primary error; a private temp may be
            # left for manual cleanup, but no caller mutation is hidden.
            pass
        raise


def _write_atomic(path: str, text: str, *, create_parents: bool = True) -> None:
    """Write ``text`` to ``path`` atomically, so review never sees half a file."""
    _write_atomic_bytes(
        path,
        text.encode("utf-8"),
        create_parents=create_parents,
    )


def _write_new_bytes(path: str, data: bytes, *, mode: Optional[int] = None) -> None:
    """Atomically publish a complete new file and never replace an existing one."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-new-", suffix=".md")
    published = False
    failed = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            if mode is not None:
                if hasattr(os, "fchmod"):
                    os.fchmod(handle.fileno(), mode)
                else:
                    os.chmod(tmp, mode)
            os.fsync(handle.fileno())
        # Hard-link publication is an atomic no-replace operation on every
        # supported local filesystem. If the filesystem cannot hard-link, fail
        # before any live target is mutated rather than weakening immutability.
        os.link(tmp, path)
        published = True
        _fsync_directory(directory)
    except BaseException:
        failed = True
        if published:
            try:
                _unlink_fsync(path)
            except OSError:
                pass
        raise
    finally:
        try:
            os.unlink(tmp)
            _fsync_directory(directory)
        except FileNotFoundError:
            pass
        except OSError:
            if not failed:
                raise


def _remove_private_temp_aliases(path: str) -> None:
    """Remove crash-left hard-link publication temps for exactly ``path``."""
    if not os.path.lexists(path):
        return
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink <= 1:
        return
    directory = os.path.dirname(path) or "."
    file_id = (info.st_dev, info.st_ino)
    removed = False
    for entry in os.scandir(directory):
        if not entry.name.startswith(".tmp-new-") or entry.path == path:
            continue
        try:
            candidate = entry.stat(follow_symlinks=False)
            if (
                stat.S_ISREG(candidate.st_mode)
                and (candidate.st_dev, candidate.st_ino) == file_id
            ):
                os.unlink(entry.path)
                removed = True
        except FileNotFoundError:
            continue
    if removed:
        _fsync_directory(directory)


def _artifact_snapshot(path: str) -> Optional[tuple[bytes, int]]:
    """Return bytes/mode for an existing artifact, or ``None`` when absent."""
    if not os.path.lexists(path):
        return None
    if _is_link_or_junction(path) or not os.path.isfile(path):
        raise StagingError(f"staging artifact path is not a regular file: {path}")
    with open(path, "rb") as handle:
        data = handle.read()
    return data, stat.S_IMODE(os.stat(path).st_mode)


def _write_artifact_batch(artifacts: Sequence[tuple[str, str]]) -> None:
    """Publish a group of artifacts atomically per file, rolling back as a set.

    Callers put the manifest last. A crash can therefore leave only an
    unadoptable directory; an ordinary write failure additionally restores or
    removes every artifact touched by this call.
    """
    snapshots = {path: _artifact_snapshot(path) for path, _text in artifacts}
    written: List[str] = []
    try:
        for path, text in artifacts:
            # Record the undo before the call: a filesystem wrapper can commit
            # a replace and then report a late error (for example on close).
            written.append(path)
            _write_atomic(path, text)
    except BaseException as primary:
        recovery_errors: List[str] = []
        for path in reversed(written):
            try:
                snapshot = snapshots[path]
                if snapshot is None:
                    if os.path.lexists(path):
                        _unlink_fsync(path)
                else:
                    data, mode = snapshot
                    _write_atomic_bytes(path, data, mode=mode)
            except BaseException as exc:
                recovery_errors.append(
                    f"could not restore staging artifact {path}: "
                    f"{type(exc).__name__}: {exc}"
                )
        if recovery_errors:
            raise StagingRecoveryError(
                "staging artifact write failed and rollback was incomplete",
                primary=primary,
                recovery_errors=recovery_errors,
            ) from primary
        raise


def skill_proposal_rows(proposals: Iterable[SkillProposal]) -> List[Dict[str, Any]]:
    """Validate proposals and return their manifest rows, in input order.

    Raises :class:`StagingError` on an unusable skill name, an unsafe live target
    path, or a collision on the skill name, the staged filename, or the live
    path: a night must never stage two skills into one file or point a proposal
    at the wrong one.

    Staged filenames are compared case-insensitively. Skill names are
    case-sensitive, so ``Research`` and ``research`` are two different skills on
    a case-sensitive filesystem — but their proposal files land in one staging
    directory, and on macOS and Windows that directory is case-insensitive, so
    the second write silently replaces the first and the manifest then points a
    surviving filename at another skill's content. Refusing the pair is the
    conservative reading of the promise above.
    """
    rows: List[Dict[str, Any]] = []
    seen_paths: Dict[str, str] = {}
    seen_files: Dict[str, str] = {}
    for proposal in proposals:
        name = _safe_skill_name(proposal.skill_name)
        if not name:
            raise StagingError(f"unsafe skill name for staging: {proposal.skill_name!r}")
        if not isinstance(proposal.proposed_skill, str):
            raise StagingError(
                f"proposed skill content for {name!r} must be text, "
                f"got {type(proposal.proposed_skill).__name__}"
            )
        requested_live = _safe_live_path(proposal.live_skill_path)
        live = _canonical_live_path(requested_live) if requested_live else ""
        if not live:
            raise StagingError(
                f"unsafe live skill path for {name!r}: {proposal.live_skill_path!r}"
            )
        parent = os.path.dirname(live)
        if os.path.basename(live) != "SKILL.md":
            raise StagingError(
                f"live skill path for {name!r} must be a SKILL.md file: {live}"
            )
        if _filesystem_key(os.path.basename(parent)) != _filesystem_key(name):
            raise StagingError(
                f"live skill path for {name!r} is not {name}/SKILL.md: {live}"
            )
        if any(row["skill_name"] == name for row in rows):
            raise StagingError(f"duplicate skill name in staging fan-out: {name!r}")
        proposed_file = proposal_filename(name)
        # casefold, not lower: it folds Unicode pairs lower() leaves distinct,
        # which is the comparison a case-insensitive filesystem actually makes.
        file_key = _filesystem_key(proposed_file)
        if file_key in seen_files:
            raise StagingError(
                f"skills {seen_files[file_key]!r} and {name!r} stage to the same "
                f"file on a case-insensitive filesystem: {proposed_file}"
            )
        # Same reasoning for the live target: /x/A.md and /x/a.md are one file
        # on macOS and Windows, so an exact-string check lets two skills
        # overwrite each other's live document. casefold rather than
        # os.path.normcase: normcase only folds case on Windows, so it is a
        # no-op on the macOS box where the collision is just as real.
        live_key = _filesystem_key(live)
        if live_key in seen_paths:
            raise StagingError(
                f"skills {seen_paths[live_key]!r} and {name!r} target the same file: {live}"
            )
        seen_paths[live_key] = name
        seen_files[file_key] = name
        rows.append({
            "skill_name": name,
            "proposed_file": proposed_file,
            "live_skill_path": live,
            "sha256": _sha256_text(proposal.proposed_skill),
        })
    return rows


def _prepare_skill_proposals(
    proposals: Iterable[SkillProposal],
) -> tuple[List[SkillProposal], List[Dict[str, Any]]]:
    """Validate proposals and pin the exact live state each one was derived from."""
    materialized = list(proposals)
    rows = skill_proposal_rows(materialized)
    for row, proposal in zip(rows, materialized):
        name = row["skill_name"]
        if not proposal.proposed_skill.strip():
            raise StagingError(f"proposed skill content for {name!r} is empty")

        live = row["live_skill_path"]
        actual_realpath = _canonical_live_path(live)
        if not actual_realpath:
            raise StagingError(f"could not canonicalize live skill path for {name!r}")
        if proposal.live_realpath:
            # This is the identity captured with the baseline read. Resolving
            # it again would follow a symlink created later and make an
            # ancestor swap look unchanged.
            expected_realpath = _safe_live_path(proposal.live_realpath)
            if (
                not expected_realpath
                or _path_identity_key(expected_realpath)
                != _path_identity_key(actual_realpath)
            ):
                raise StagingError(
                    f"live skill canonical target for {name!r} changed during consolidation"
                )

        actual_sha256 = _sha256_file_bytes(live)
        expected_sha256 = proposal.live_sha256
        if expected_sha256 is None:
            expected_sha256 = actual_sha256
        elif expected_sha256 != "" and not _valid_sha256_pin(expected_sha256):
            raise StagingError(f"invalid live baseline sha256 for {name!r}")
        if actual_sha256 != expected_sha256:
            raise StagingError(
                f"live skill for {name!r} changed during consolidation; "
                "discard and rerun this night"
            )

        row["live_sha256"] = expected_sha256
        row["live_realpath"] = actual_realpath
    return materialized, rows


def _prepare_legacy_proposal(
    *,
    label: str,
    proposed_file: str,
    proposed_text: str,
    live_path: str,
    live_sha256: Optional[str],
    live_realpath: str,
) -> Dict[str, Any]:
    """Validate and pin one legacy SKILL.md/CLAUDE.md proposal."""
    if not isinstance(proposed_text, str) or (
        label == "skill" and not proposed_text.strip()
    ):
        raise StagingError(f"legacy {label} proposal must be valid text")
    requested = _safe_live_path(live_path)
    canonical = _canonical_live_path(requested) if requested else ""
    if not canonical:
        raise StagingError(f"unsafe legacy {label} live path: {live_path!r}")
    expected_basename = "SKILL.md" if label == "skill" else "CLAUDE.md"
    if os.path.basename(canonical) != expected_basename:
        raise StagingError(
            f"legacy {label} target must be {expected_basename}: {canonical}"
        )
    actual_realpath = _canonical_live_path(canonical)
    if live_realpath:
        # Compare the identity captured by the original baseline read without
        # resolving it again: doing so would bless a symlink/junction swap
        # that happened after the caller captured the path.
        expected_realpath = _safe_live_path(live_realpath)
        if not expected_realpath or (
            _path_identity_key(expected_realpath)
            != _path_identity_key(actual_realpath)
        ):
            raise StagingError(
                f"legacy {label} canonical target changed during consolidation"
            )
    actual_sha256 = _sha256_file_bytes(canonical)
    expected_sha256 = actual_sha256 if live_sha256 is None else live_sha256
    if expected_sha256 != "" and not _valid_sha256_pin(expected_sha256):
        raise StagingError(f"invalid legacy {label} live baseline sha256")
    if actual_sha256 != expected_sha256:
        raise StagingError(
            f"legacy {label} changed during consolidation; discard and rerun this night"
        )
    return {
        "proposed_file": proposed_file,
        "live_path": canonical,
        "sha256": _sha256_text(proposed_text),
        "live_sha256": expected_sha256,
        "live_realpath": actual_realpath,
    }


def write_skill_proposals(
    out_dir: str, proposals: Iterable[SkillProposal]
) -> List[Dict[str, Any]]:
    """Stage one uniquely named proposal file per skill; return manifest rows.

    Every proposal is validated before anything is written, so a rejected
    fan-out leaves no partial files behind.
    """
    # Materialise once. The signature accepts any Iterable, so a generator is
    # legal input — and it would otherwise be drained by the validation pass.
    proposals, rows = _prepare_skill_proposals(proposals)
    if not rows:
        return rows
    os.makedirs(out_dir, exist_ok=True)
    _write_artifact_batch([
        (os.path.join(out_dir, row["proposed_file"]), proposal.proposed_skill)
        for row, proposal in zip(rows, proposals)
    ])
    return rows


def _ts_dir() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def staging_root(project: str) -> str:
    return os.path.join(project, ".skillopt-sleep", "staging")


def _ensure_staging_root(project: str) -> str:
    """Create the private staging root without following an injected alias."""
    root = staging_root(project)
    project_real = _canonical_live_path(project)
    expected_real = os.path.join(project_real, ".skillopt-sleep", "staging")
    if os.path.lexists(root) and (
        _is_link_or_junction(root) or not os.path.isdir(root)
    ):
        raise StagingError(f"staging root is unsafe: {root}")
    os.makedirs(root, exist_ok=True)
    if _path_identity_key(_canonical_live_path(root)) != _path_identity_key(expected_real):
        raise StagingError(f"staging root passes through a symlink: {root}")
    return root


def new_staging_dir(project: str) -> str:
    """Atomically reserve a staging path, unique across concurrent runs."""
    root = _ensure_staging_root(project)
    base = os.path.join(root, _ts_dir())
    out, i = base, 2
    while True:
        try:
            os.mkdir(out)
            _fsync_parent(out)
            _fsync_directory(out)
            return out
        except FileExistsError:
            out = f"{base}-{i}"
            i += 1


_STAGING_DIR_RE = re.compile(r"^(\d{8}-\d{6})(?:-(\d+))?$")
_MANIFEST_SCHEMA = "skillopt-sleep-staging"
_MANIFEST_VERSION = 2
_LATEST_FILENAME = ".latest"


def _latest_pointer_path(root: str) -> str:
    return os.path.join(root, _LATEST_FILENAME)


def _published_night_from_pointer(root: str) -> Optional[str]:
    """Return the atomically published night, or ``None`` for an old/invalid root.
    """
    pointer = _latest_pointer_path(root)
    if not os.path.lexists(pointer):
        return None
    try:
        info = os.lstat(pointer)
        if (
            _is_link_or_junction(pointer)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            return None
        raw, _mode, file_id = _file_snapshot(pointer)
        current = os.lstat(pointer)
        if file_id != (current.st_dev, current.st_ino) or current.st_nlink != 1:
            return None
        if raw is None:
            return None
        name = raw.decode("utf-8").strip()
    except (OSError, UnicodeError, StagingError):
        return None
    if not _STAGING_DIR_RE.fullmatch(name) or os.path.basename(name) != name:
        return None
    night = os.path.join(root, name)
    manifest = os.path.join(night, "manifest.json")
    if (
        _is_link_or_junction(night)
        or not os.path.isdir(night)
        or _is_link_or_junction(manifest)
        or not os.path.isfile(manifest)
    ):
        return None
    try:
        expected = os.path.join(os.path.realpath(root), name)
        if _path_identity_key(os.path.realpath(night)) != _path_identity_key(
            expected
        ):
            return None
    except (OSError, ValueError):
        return None
    return night


def _publish_latest(root: str, out: str) -> None:
    """Atomically make ``out`` the last successfully published staging night."""
    root_abs = os.path.abspath(root)
    out_abs = os.path.abspath(out)
    name = os.path.basename(out_abs)
    if (
        not _STAGING_DIR_RE.fullmatch(name)
        or _path_identity_key(os.path.dirname(out_abs))
        != _path_identity_key(root_abs)
        or _is_link_or_junction(out_abs)
        or not os.path.isdir(out_abs)
        or _path_identity_key(os.path.realpath(out_abs))
        != _path_identity_key(os.path.join(os.path.realpath(root_abs), name))
    ):
        raise StagingError(f"staging directory is not a safe reserved night: {out}")
    manifest = os.path.join(out_abs, "manifest.json")
    if _is_link_or_junction(manifest) or not os.path.isfile(manifest):
        raise StagingError(f"cannot publish a staging night without a manifest: {out}")
    pointer = _latest_pointer_path(root_abs)
    if os.path.lexists(pointer):
        info = os.lstat(pointer)
        if (
            _is_link_or_junction(pointer)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            raise StagingError(f"latest-staging pointer is unsafe: {pointer}")
    _write_atomic_bytes(pointer, f"{name}\n".encode("utf-8"), mode=0o600)


def _staging_order(path: str) -> tuple:
    """Order nights by the publication marker, which adoption never mutates."""
    manifest_path = os.path.join(path, "manifest.json")
    try:
        published_ns = os.stat(manifest_path, follow_symlinks=False).st_mtime_ns
    except OSError:
        published_ns = 0
    match = _STAGING_DIR_RE.fullmatch(os.path.basename(path))
    if match:
        return (published_ns, 1, match.group(1), int(match.group(2) or 1))
    return (published_ns, 0, "", 0)


def latest_staging(project: str) -> Optional[str]:
    root = staging_root(project)
    if _is_link_or_junction(root) or not os.path.isdir(root):
        return None
    published = _published_night_from_pointer(root)
    if published is not None:
        return published
    subs = []
    for entry in os.listdir(root):
        if not _STAGING_DIR_RE.fullmatch(entry):
            continue
        path = os.path.join(root, entry)
        if _is_link_or_junction(path) or not os.path.isdir(path):
            continue
        manifest_path = os.path.join(path, "manifest.json")
        if _is_link_or_junction(manifest_path) or not os.path.isfile(manifest_path):
            continue
        subs.append(path)
    subs.sort(key=_staging_order, reverse=True)
    for p in subs:
        # Only adoptable folders count: a no-tasks night leaves evidence.jsonl
        # but no manifest, and adopt() needs the manifest.
        return p
    return None


def write_staging(
    project: str,
    *,
    report: SleepReport,
    proposed_skill: Optional[str],
    proposed_memory: Optional[str],
    live_skill_path: str,
    live_memory_path: str,
    live_skill_sha256: Optional[str] = None,
    live_memory_sha256: Optional[str] = None,
    live_skill_realpath: str = "",
    live_memory_realpath: str = "",
    report_md: str,
    out_dir: str = "",
    skill_proposals: Iterable[SkillProposal] = (),
    skill_roots: Iterable[str] = (),
) -> str:
    """Write proposals + report into staging/<ts>/ and return that path.

    ``out_dir`` lets the cycle pre-create the night's staging folder at cycle
    START, so incremental artifacts (evidence.jsonl) accumulate in the same
    place the report lands.

    ``skill_proposals`` stages one extra uniquely named file and manifest row per
    skill for a multi-skill night. Left empty, the staging layout and manifest
    are exactly the legacy single-proposal ones.
    """
    root = _ensure_staging_root(project)
    out = out_dir or new_staging_dir(project)
    if out_dir:
        out_abs = os.path.abspath(out)
        root_abs = os.path.abspath(root)
        if (
            not _STAGING_DIR_RE.fullmatch(os.path.basename(out_abs))
            or _path_identity_key(os.path.dirname(out_abs))
            != _path_identity_key(root_abs)
        ):
            raise StagingError(f"staging directory is unsafe: {out}")
        if os.path.lexists(out) and (
            _is_link_or_junction(out) or not os.path.isdir(out)
        ):
            raise StagingError(f"staging directory is unsafe: {out}")
        os.makedirs(out, exist_ok=True)

    proposals, skill_rows = _prepare_skill_proposals(skill_proposals)
    if proposed_skill is not None and skill_rows:
        legacy_live = _safe_live_path(live_skill_path)
        legacy_real = _canonical_live_path(legacy_live) if legacy_live else ""
        if not legacy_real:
            raise StagingError(
                f"unsafe managed live skill path in mixed staging: {live_skill_path!r}"
            )
        legacy_key = _filesystem_key(legacy_real)
        for row in skill_rows:
            if _filesystem_key(row["live_realpath"]) == legacy_key:
                raise StagingError(
                    f"managed and per-skill proposals target the same live file: "
                    f"{legacy_real}"
                )

    legacy: Dict[str, Dict[str, Any]] = {}
    if proposed_skill is not None:
        legacy["skill"] = _prepare_legacy_proposal(
            label="skill",
            proposed_file="proposed_SKILL.md",
            proposed_text=proposed_skill,
            live_path=live_skill_path,
            live_sha256=live_skill_sha256,
            live_realpath=live_skill_realpath,
        )
        live_skill_path = legacy["skill"]["live_path"]
    if proposed_memory is not None:
        legacy["memory"] = _prepare_legacy_proposal(
            label="memory",
            proposed_file="proposed_CLAUDE.md",
            proposed_text=proposed_memory,
            live_path=live_memory_path,
            live_sha256=live_memory_sha256,
            live_realpath=live_memory_realpath,
        )
        live_memory_path = legacy["memory"]["live_path"]

    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "schema_version": _MANIFEST_VERSION,
        "live_skill_path": live_skill_path,
        "live_memory_path": live_memory_path,
        # PyPI v0.2.0 adopted these top-level flags without integrity pins.
        # Keep them false so an old runtime fails closed on a new manifest.
        "has_skill": False,
        "has_memory": False,
        "has_managed_skill": proposed_skill is not None,
        "has_managed_memory": proposed_memory is not None,
        "accepted": report.accepted,
    }
    if skill_rows:
        manifest["skills"] = skill_rows
        # The roots the fan-out actually resolved from. Recorded so adoption can
        # re-check containment instead of trusting each row's live path.
        recorded_roots = [
            os.path.abspath(os.path.expanduser(str(root)))
            for root in skill_roots
            if isinstance(root, str) and str(root).strip()
        ]
        if not recorded_roots:
            # Low-level callers may not know the search roots. Every live target
            # is <root>/<name>/SKILL.md, so the root each resolved path sits in
            # is derivable here -- at stage time, from paths we just resolved
            # ourselves, never from the manifest we are about to trust later.
            recorded_roots = [
                os.path.dirname(os.path.dirname(os.path.abspath(str(row["live_skill_path"]))))
                for row in skill_rows
                if str(row.get("live_skill_path") or "").strip()
            ]
        manifest["skill_roots"] = list(dict.fromkeys(recorded_roots))
    if legacy:
        manifest["legacy"] = legacy
    artifacts: List[tuple[str, str]] = [
        (
            os.path.join(out, row["proposed_file"]),
            proposal.proposed_skill,
        )
        for row, proposal in zip(skill_rows, proposals)
    ]
    if proposed_skill is not None:
        artifacts.append((os.path.join(out, "proposed_SKILL.md"), proposed_skill))
    if proposed_memory is not None:
        artifacts.append((os.path.join(out, "proposed_CLAUDE.md"), proposed_memory))
    artifacts.extend([
        (
            os.path.join(out, "report.json"),
            json.dumps(
                json_safe(report.to_dict()),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
        ),
        (os.path.join(out, "report.md"), report_md),
        # The manifest is the publication marker and must always be last.
        (
            os.path.join(out, "manifest.json"),
            json.dumps(manifest, ensure_ascii=False, indent=2),
        ),
    ])
    _write_artifact_batch(artifacts)
    _publish_latest(root, out)
    return out


@dataclass
class AdoptedSkill:
    """Receipt for one adopted skill: where it landed and what changed."""

    skill_name: str
    live_skill_path: str
    sha256_before: str          # "" when no live file existed yet
    sha256_after: str
    backup_path: str = ""       # "" when there was nothing to back up


@dataclass
class _TransactionTarget:
    """Fully pinned mutation used by the durable adoption transaction."""

    key: str
    live_path: str
    expected_realpath: str
    expected_basename: str
    proposed_bytes: bytes
    proposed_sha256: str
    original_bytes: Optional[bytes]
    original_mode: Optional[int]
    original_file_id: Optional[tuple[int, int]]
    baseline_sha256: str
    backup_path: str
    created_dirs: tuple[str, ...] = ()
    # Persisted identities for directories this transaction actually created.
    # ``None`` is fail-closed: recovery may leave an empty directory behind,
    # but it must never remove a path whose ownership was not durably recorded.
    created_dir_ids: tuple[Optional[tuple[int, int]], ...] = ()


_WAL_FILENAME = ".adopt-transaction.json"
_WAL_VERSION = 2


def _bytes_sha256(data: Optional[bytes]) -> str:
    return hashlib.sha256(data).hexdigest() if data is not None else ""


def _modes_match(actual: Optional[int], expected: Optional[int]) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if os.name == "nt":
        # Windows chmod/stat expose only the portable read-only distinction.
        return bool(actual & stat.S_IWRITE) == bool(expected & stat.S_IWRITE)
    return actual == expected


def _b64(data: Optional[bytes]) -> Optional[str]:
    return base64.b64encode(data).decode("ascii") if data is not None else None


def _from_b64(value: object, *, field: str) -> Optional[bytes]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise StagingError(f"transaction WAL {field} must be base64 text or null")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as exc:
        raise StagingError(f"transaction WAL {field} is invalid base64") from exc


def _file_snapshot(
    path: str,
) -> tuple[Optional[bytes], Optional[int], Optional[tuple[int, int]]]:
    """Read a regular file and its identity without following a final symlink."""
    if not os.path.lexists(path):
        return None, None, None
    if _is_link_or_junction(path) or not os.path.isfile(path):
        raise StagingError(f"target is not a regular file: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(path, flags)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise StagingError(f"target is not a regular file: {path}")
            chunks: List[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            info = os.fstat(fd)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_size",
                "st_mtime_ns",
                "st_nlink",
            )
            if any(getattr(before, field) != getattr(info, field) for field in stable_fields):
                raise StagingError(f"target changed while it was being read: {path}")
        finally:
            os.close(fd)
    except OSError as exc:
        raise StagingError(f"could not snapshot target: {path}") from exc
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise StagingError(f"target changed while it was being read: {path}") from exc
    file_id = (info.st_dev, info.st_ino)
    if (
        (current.st_dev, current.st_ino) != file_id
        or current.st_mode != info.st_mode
        or current.st_size != info.st_size
        or current.st_mtime_ns != info.st_mtime_ns
        or current.st_nlink != info.st_nlink
    ):
        raise StagingError(f"target changed while it was being read: {path}")
    return b"".join(chunks), stat.S_IMODE(info.st_mode), file_id


def _canonical_staging_dir(staging_dir: str) -> str:
    """Return one stable absolute staging identity for locks, WAL, and receipts."""
    absolute = os.path.abspath(staging_dir)
    if _is_link_or_junction(absolute) or not os.path.isdir(absolute):
        raise StagingError(f"staging directory is unsafe: {staging_dir}")
    canonical = os.path.realpath(absolute)
    if not os.path.isdir(canonical):
        raise StagingError(f"staging directory is unsafe: {staging_dir}")
    return canonical


def _manifest_schema_version(manifest: Dict[str, Any]) -> int:
    """Validate the explicit schema, while retaining read support for old nights."""
    schema_present = "schema" in manifest or "schema_version" in manifest
    if not schema_present:
        return 1
    if (
        manifest.get("schema") != _MANIFEST_SCHEMA
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != _MANIFEST_VERSION
    ):
        raise StagingError("staging manifest has an unsupported schema version")
    return _MANIFEST_VERSION


def staged_skills(staging_dir: str) -> List[Dict[str, Any]]:
    """Manifest rows for the per-skill proposals staged in ``staging_dir``."""
    if _is_link_or_junction(staging_dir) or not os.path.isdir(staging_dir):
        raise StagingError(f"staging directory is unsafe: {staging_dir}")
    manifest_path = os.path.join(staging_dir, "manifest.json")
    if _is_link_or_junction(manifest_path):
        raise StagingError("staging manifest must not be a symlink")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StagingError(f"cannot read staging manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise StagingError("staging manifest must be a JSON object")
    _manifest_schema_version(manifest)
    if "skills" not in manifest:
        return []
    rows = manifest["skills"]
    if not isinstance(rows, list):
        raise StagingError("staging manifest 'skills' must be a list")
    if any(not isinstance(row, dict) for row in rows):
        raise StagingError("every staging manifest 'skills' row must be an object")
    return rows


def staged_skill_roots(staging_dir: str) -> List[str]:
    """The skills roots recorded when this night was staged."""
    manifest_path = os.path.join(staging_dir, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StagingError(f"cannot read staging manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise StagingError("staging manifest must be a JSON object")
    roots = manifest.get("skill_roots")
    if not isinstance(roots, list) or not roots:
        raise StagingError(
            "staging manifest is missing 'skill_roots'; it was written by an older "
            "version that could not confine adoption. Discard and restage this night."
        )
    out: List[str] = []
    for root in roots:
        if not isinstance(root, str) or not root.strip():
            raise StagingError("staging manifest 'skill_roots' must be non-empty strings")
        if not os.path.isabs(root):
            raise StagingError(f"staging manifest 'skill_roots' entry is not absolute: {root}")
        out.append(root)
    return out


def _selected_rows(
    rows: Sequence[Dict[str, Any]], skill_names: Optional[Sequence[str]]
) -> List[Dict[str, Any]]:
    """Rows for the reviewed subset, in manifest order, or every row."""
    if skill_names is None:
        return list(rows)
    wanted = [str(n).strip() for n in skill_names]
    if not wanted:
        return []
    known = {str(row.get("skill_name", "")) for row in rows}
    unknown = [n for n in wanted if n not in known]
    if unknown:
        raise StagingError(f"no staged proposal for: {', '.join(sorted(unknown))}")
    duplicates = {n for n in wanted if wanted.count(n) > 1}
    if duplicates:
        raise StagingError(f"skill selected twice: {', '.join(sorted(duplicates))}")
    chosen = set(wanted)
    return [row for row in rows if str(row.get("skill_name", "")) in chosen]


def _revalidate_selected_skill_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    all_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    """Re-run uniqueness and live-target checks at adoption time.

    Staging already refused collisions, but the manifest can be edited between
    staging and adopt. A tampered pair that shares a skill name, a staged
    filename, or a live target must fail here with no writes. The check runs
    against every staged row, not only the selection, so adopting one skill
    cannot hide a sibling that now points at the same file. Live paths are
    also compared by realpath so a symlink cannot hide a second claim on one
    file, and a live path that exists as something other than a file is
    refused rather than overwritten.
    """
    universe = list(all_rows) if all_rows is not None else list(rows)
    skill_proposal_rows([
        SkillProposal(
            str(row.get("skill_name") or ""),
            "",
            str(row.get("live_skill_path") or ""),
        )
        for row in universe
    ])
    seen_real: Dict[str, str] = {}
    seen_files: Dict[tuple[int, int], str] = {}
    for row in universe:
        name = _safe_skill_name(row.get("skill_name"))
        live = _safe_live_path(row.get("live_skill_path"))
        if not name or not live:
            continue
        expected_real = _safe_live_path(row.get("live_realpath"))
        live_pin = row.get("live_sha256")
        if not expected_real or (
            live_pin != "" and not _valid_sha256_pin(live_pin)
        ):
            raise StagingError(
                f"staged skill {name!r} is missing live baseline pins; "
                "discard and restage this night"
            )
        if _is_link_or_junction(live):
            raise StagingError(f"live skill path for {name!r} is a symlink: {live}")
        parent = os.path.dirname(live)
        if _is_link_or_junction(parent):
            raise StagingError(
                f"live skill parent directory for {name!r} is a symlink: {parent}"
            )
        if os.path.basename(live) != "SKILL.md":
            raise StagingError(
                f"live skill path for {name!r} must be a SKILL.md file: {live}"
            )
        real = _canonical_live_path(live)
        if not real or _path_identity_key(real) != _path_identity_key(expected_real):
            raise StagingError(
                f"live skill canonical target for {name!r} changed since staging"
            )
        key = _filesystem_key(real)
        if key in seen_real:
            raise StagingError(
                f"skills {seen_real[key]!r} and {name!r} target the same file: {live}"
            )
        seen_real[key] = name
        if os.path.lexists(live) and not os.path.isfile(live):
            raise StagingError(
                f"live skill path for {name!r} exists and is not a file: {live}"
            )
        if os.path.isfile(live) and not _is_link_or_junction(live):
            info = os.stat(live, follow_symlinks=False)
            file_id = (info.st_dev, info.st_ino)
            if file_id in seen_files:
                raise StagingError(
                    f"skills {seen_files[file_id]!r} and {name!r} target the "
                    f"same file through a hard link: {live}"
                )
            seen_files[file_id] = name


def _valid_sha256_pin(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _adopt_target_ok(
    label: str,
    live: str,
    expected_realpath: str,
    *,
    expected_basename: str,
) -> None:
    """Refuse targets that moved, create dirs, or traverse any symlink."""
    if _is_link_or_junction(live):
        raise StagingError(f"live target for {label} is a symlink: {live}")
    if os.path.lexists(live):
        info = os.lstat(live)
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise StagingError(f"live target for {label} has multiple hard links")
    parent = os.path.dirname(live)
    if _is_link_or_junction(parent):
        raise StagingError(
            f"live parent directory for {label} is a symlink: {parent}"
        )
    if not os.path.isdir(parent):
        raise StagingError(
            f"live parent directory for {label} does not exist: {parent}"
        )
    current_realpath = _canonical_live_path(live)
    if (
        not current_realpath
        or _path_identity_key(current_realpath)
        != _path_identity_key(expected_realpath)
    ):
        raise StagingError(
            f"live canonical target for {label} changed since staging"
        )
    # Genuine staged rows store the canonical path itself. Any difference now
    # means an ancestor was replaced by a symlink/junction after review.
    if _path_identity_key(current_realpath) != _path_identity_key(live):
        raise StagingError(
            f"live path for {label} passes through a symlink or junction: {live}"
        )
    if os.path.basename(live) != expected_basename:
        raise StagingError(
            f"live target for {label} must be {expected_basename}: {live}"
        )


def _adopt_live_target_ok(
    name: str,
    live: str,
    expected_realpath: str,
    roots: Iterable[str] = (),
) -> None:
    _adopt_target_ok(
        repr(name),
        live,
        expected_realpath,
        expected_basename="SKILL.md",
    )
    parent = os.path.dirname(live)
    if _filesystem_key(os.path.basename(parent)) != _filesystem_key(name):
        raise StagingError(
            f"live skill path for {name!r} is not {name}/SKILL.md: {live}"
        )
    # Shape is not location. Run this only after ``_adopt_target_ok`` has proved
    # ``realpath(live) == live``, so a symlinked ancestor cannot fake containment.
    if roots and not _live_target_within_roots(live, roots):
        raise StagingError(
            f"live skill path for {name!r} is outside the skills roots recorded "
            f"when this night was staged: {live}"
        )


def _planned_live_directories(
    label: str,
    live: str,
    expected_realpath: str,
    *,
    expected_basename: str,
) -> tuple[str, ...]:
    """Validate an absent legacy target and return missing parents outer-first."""
    if os.path.basename(live) != expected_basename:
        raise StagingError(f"live target for {label} must be {expected_basename}: {live}")
    current_realpath = _canonical_live_path(live)
    if (
        not current_realpath
        or _path_identity_key(current_realpath) != _path_identity_key(expected_realpath)
        or _path_identity_key(current_realpath) != _path_identity_key(live)
    ):
        raise StagingError(
            f"live path for {label} changed or passes through a symlink/junction"
        )
    missing: List[str] = []
    current = os.path.dirname(live)
    while not os.path.lexists(current):
        missing.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            raise StagingError(f"no existing ancestor for live target {label}")
        current = parent
    info = os.lstat(current)
    if _is_link_or_junction(current) or not stat.S_ISDIR(info.st_mode):
        raise StagingError(f"live ancestor for {label} is unsafe: {current}")
    return tuple(reversed(missing))


def _read_receipt_file(
    path: str,
) -> tuple[
    List[Dict[str, Any]],
    Optional[bytes],
    Optional[int],
    Optional[tuple[int, int]],
]:
    if not os.path.lexists(path):
        return [], None, None, None
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise StagingError(f"cannot inspect adoption receipt path: {exc}") from exc
    if (
        _is_link_or_junction(path)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise StagingError("adoption receipt path is not a private regular file")
    try:
        original, mode, file_id = _file_snapshot(path)
        if original is None:
            raise StagingError("adoption receipt disappeared while being read")
        current = os.lstat(path)
        if (
            file_id != (current.st_dev, current.st_ino)
            or current.st_nlink != 1
        ):
            raise StagingError("adoption receipt identity changed while being read")
        payload = json.loads(original.decode("utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise StagingError(f"cannot read existing adoption receipt: {exc}") from exc
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise StagingError("existing adoption receipt must be a list of objects")
    return payload, original, mode, file_id


def _read_existing_receipts(
    path: str,
    staging_dir: str,
) -> tuple[
    List[Dict[str, Any]],
    Optional[bytes],
    Optional[int],
    Optional[tuple[int, int]],
]:
    """Load and verify the append-only per-skill adoption ledger."""
    payload, original, mode, file_id = _read_receipt_file(path)
    seen: set[str] = set()
    schema = {
        "skill_name",
        "live_skill_path",
        "sha256_before",
        "sha256_after",
        "backup_path",
    }
    for row in payload:
        if set(row) != schema:
            raise StagingError("existing adoption receipt has an invalid schema")
        name = _safe_skill_name(row.get("skill_name"))
        if not name or row.get("skill_name") != name or name in seen:
            raise StagingError("existing adoption receipt has invalid or duplicate skills")
        seen.add(name)
        live = _safe_live_path(row.get("live_skill_path"))
        before = row.get("sha256_before")
        after = row.get("sha256_after")
        backup = row.get("backup_path")
        if (
            not live
            or os.path.basename(live) != "SKILL.md"
            or _filesystem_key(os.path.basename(os.path.dirname(live)))
            != _filesystem_key(name)
            or (before != "" and not _valid_sha256_pin(before))
        ):
            raise StagingError(f"existing adoption receipt for {name!r} is invalid")
        if not _valid_sha256_pin(after) or not isinstance(backup, str):
            raise StagingError(f"existing adoption receipt for {name!r} is invalid")
        if before == "":
            if backup:
                raise StagingError(
                    f"existing adoption receipt for {name!r} has an unexpected backup"
                )
            continue
        expected = os.path.join(
            staging_dir, "backup", "skills", name, "SKILL.md"
        )
        if _path_identity_key(backup) != _path_identity_key(expected):
            raise StagingError(
                f"existing adoption receipt for {name!r} has an invalid backup path"
            )
        backup_sha256 = _immutable_backup_sha256(expected, staging_dir)
        if backup_sha256 is None:
            raise StagingError(
                f"immutable backup for previously adopted skill {name!r} is missing"
            )
        if backup_sha256 != before:
            raise StagingError(
                f"immutable backup for previously adopted skill {name!r} changed"
            )
    return payload, original, mode, file_id


def pending_staged_skills(staging_dir: str) -> List[Dict[str, Any]]:
    """Return fan-out rows not already recorded in the validated adoption ledger."""
    staging_dir = _canonical_staging_dir(staging_dir)
    if os.path.lexists(_wal_path(staging_dir)):
        raise StagingError(
            "an interrupted adoption must be recovered before listing pending skills"
        )
    rows = staged_skills(staging_dir)
    receipt_path = os.path.join(staging_dir, "adopted_skills.json")
    receipts, _raw, _mode, _file_id = _read_existing_receipts(
        receipt_path, staging_dir
    )
    adopted = {str(row["skill_name"]) for row in receipts}
    return [row for row in rows if str(row.get("skill_name")) not in adopted]


def has_staged_managed(staging_dir: str) -> bool:
    """Return whether a validated managed skill or memory proposal is present."""
    staging_dir = _canonical_staging_dir(staging_dir)
    if os.path.lexists(_wal_path(staging_dir)):
        raise StagingError(
            "an interrupted adoption must be recovered before listing managed proposals"
        )
    return bool(_legacy_rows(_load_manifest(staging_dir)))


def has_pending_staged_managed(staging_dir: str) -> bool:
    """Return whether a validated managed proposal target remains unadopted."""
    staging_dir = _canonical_staging_dir(staging_dir)
    if os.path.lexists(_wal_path(staging_dir)):
        raise StagingError(
            "an interrupted adoption must be recovered before listing managed proposals"
        )
    rows = _legacy_rows(_load_manifest(staging_dir))
    if not rows:
        return False
    receipts, _raw, _mode, _file_id = _read_legacy_receipts(
        os.path.join(staging_dir, "adopted_legacy.json"),
        staging_dir,
    )
    adopted = {str(row["target"]) for row in receipts}
    return any(label not in adopted for label in rows)


def _target_lock_paths(live_paths: Sequence[str]) -> List[str]:
    """Stable per-target lock names shared by separate staging nights."""
    if hasattr(os, "getuid"):
        identity = str(os.getuid())
    else:
        user_material = f"{getpass.getuser()}|{os.path.expanduser('~')}"
        identity = hashlib.sha256(user_material.encode("utf-8")).hexdigest()[:16]
    root = os.path.join(tempfile.gettempdir(), f"skillopt-sleep-adopt-{identity}")
    try:
        os.mkdir(root, 0o700)
        _fsync_parent(root)
    except FileExistsError:
        pass
    info = os.lstat(root)
    if _is_link_or_junction(root) or not stat.S_ISDIR(info.st_mode):
        raise StagingError(f"adoption lock root is unsafe: {root}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise StagingError(f"adoption lock root has the wrong owner: {root}")
    if os.name != "nt" and stat.S_IMODE(info.st_mode) != 0o700:
        raise StagingError(
            f"adoption lock root permissions are unsafe; expected 0700: {root}"
        )
    return [
        os.path.join(
            root,
            hashlib.sha256(path.encode("utf-8")).hexdigest() + ".lock",
        )
        for path in sorted({_filesystem_key(path) for path in live_paths})
    ]


@contextmanager
def _exclusive_create_locks(paths: Sequence[str]):
    """Acquire cross-platform fail-closed locks with atomic file creation."""
    acquired: List[tuple[str, int, tuple[int, int]]] = []
    try:
        for path in paths:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as exc:
                raise StagingError(
                    "skill adoption is already in progress or a stale lock exists: "
                    f"{path}"
                ) from exc
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            acquired.append((path, fd, identity))
            os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(fd)
        yield
    finally:
        for path, fd, identity in reversed(acquired):
            try:
                os.close(fd)
            except OSError:
                # The operation inside the lock has a definitive result; a
                # cleanup error must not turn a committed adoption into a
                # reported failure. A surviving lock fails closed next time.
                pass
            try:
                info = os.lstat(path)
                if (info.st_dev, info.st_ino) == identity:
                    os.unlink(path)
            except OSError:
                pass


def _wal_path(staging_dir: str) -> str:
    return os.path.join(staging_dir, _WAL_FILENAME)


def _target_wal_row(target: _TransactionTarget) -> Dict[str, Any]:
    created_dir_ids = target.created_dir_ids or tuple(
        None for _path in target.created_dirs
    )
    return {
        "key": target.key,
        "live_path": target.live_path,
        "expected_realpath": target.expected_realpath,
        "expected_basename": target.expected_basename,
        "proposed_sha256": target.proposed_sha256,
        "original_b64": _b64(target.original_bytes),
        "original_mode": target.original_mode,
        "original_file_id": (
            list(target.original_file_id)
            if target.original_file_id is not None
            else None
        ),
        "baseline_sha256": target.baseline_sha256,
        "backup_path": target.backup_path,
        "created_dirs": list(target.created_dirs),
        "created_dir_ids": [
            list(file_id) if file_id is not None else None
            for file_id in created_dir_ids
        ],
    }


def _transaction_wal(
    *,
    kind: str,
    targets: Sequence[_TransactionTarget],
    receipt_path: str,
    receipt_original: Optional[bytes],
    receipt_mode: Optional[int],
    receipt_file_id: Optional[tuple[int, int]],
    receipt_after: bytes,
) -> Dict[str, Any]:
    return {
        "version": _WAL_VERSION,
        "kind": kind,
        "targets": [_target_wal_row(target) for target in targets],
        "receipt": {
            "path": receipt_path,
            "original_b64": _b64(receipt_original),
            "original_mode": receipt_mode,
            "original_file_id": (
                list(receipt_file_id) if receipt_file_id is not None else None
            ),
            "baseline_sha256": _bytes_sha256(receipt_original),
            "proposed_sha256": hashlib.sha256(receipt_after).hexdigest(),
        },
    }


def _write_transaction_wal(staging_dir: str, wal: Dict[str, Any]) -> None:
    path = _wal_path(staging_dir)
    if os.path.lexists(path):
        raise StagingError(
            f"an adoption recovery journal already exists; recover it first: {path}"
        )
    payload = json.dumps(wal, ensure_ascii=False, indent=2).encode("utf-8")
    _write_new_bytes(path, payload, mode=0o600)


def _rewrite_transaction_wal(
    staging_dir: str,
    *,
    expected_wal: Dict[str, Any],
    replacement_wal: Dict[str, Any],
) -> None:
    """Durably add post-creation identities without accepting another journal."""
    current = _read_transaction_wal(staging_dir)
    if current != expected_wal:
        raise StagingError(
            "adoption recovery journal changed during directory creation"
        )
    payload = json.dumps(
        replacement_wal, ensure_ascii=False, indent=2
    ).encode("utf-8")
    _write_atomic_bytes(
        _wal_path(staging_dir), payload, create_parents=False, mode=0o600
    )


def _read_transaction_wal(staging_dir: str) -> Optional[Dict[str, Any]]:
    path = _wal_path(staging_dir)
    if not os.path.lexists(path):
        return None
    _remove_private_temp_aliases(path)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise StagingError(f"cannot inspect adoption recovery journal: {exc}") from exc
    if (
        _is_link_or_junction(path)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise StagingError(f"adoption recovery journal is unsafe: {path}")
    try:
        raw, _mode, _file_id = _file_snapshot(path)
        if raw is None:
            raise StagingError("adoption recovery journal disappeared")
        current = os.lstat(path)
        if (
            _file_id != (current.st_dev, current.st_ino)
            or current.st_nlink != 1
        ):
            raise StagingError("adoption recovery journal identity changed")
        payload = json.loads(raw.decode("utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise StagingError(f"cannot read adoption recovery journal: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"version", "kind", "targets", "receipt"}
        or type(payload.get("version")) is not int
        or payload.get("version") not in {1, _WAL_VERSION}
    ):
        raise StagingError("adoption recovery journal has an unsupported format")
    return payload


def _remove_transaction_wal(
    staging_dir: str, *, expected_wal: Optional[Dict[str, Any]] = None
) -> None:
    path = _wal_path(staging_dir)
    if expected_wal is not None:
        current = _read_transaction_wal(staging_dir)
        if current != expected_wal:
            raise StagingError("adoption recovery journal changed before commit")
    if os.path.lexists(path):
        _remove_private_temp_aliases(path)
        info = os.lstat(path)
        if (
            _is_link_or_junction(path)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            raise StagingError(f"adoption recovery journal is unsafe: {path}")
        _unlink_fsync(path)


def _path_is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath(
            [os.path.abspath(path), os.path.abspath(root)]
        ) == os.path.abspath(root)
    except ValueError:
        return False


def _existing_path_is_canonical_staging_descendant(
    path: str, staging_dir: str
) -> bool:
    """Reject a staging descendant reached through a symlink or junction.

    The staging root itself may be supplied through a symlink, so compare the
    resolved candidate with the same relative path beneath the resolved root.
    """
    try:
        relative = os.path.relpath(path, staging_dir)
    except ValueError:
        return False
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return False
    expected_real = os.path.join(os.path.realpath(staging_dir), relative)
    return _path_identity_key(os.path.realpath(path)) == _path_identity_key(expected_real)


def _immutable_backup_snapshot(
    path: str, staging_dir: str, *, repair_temp_aliases: bool = False
) -> Optional[tuple[str, tuple[int, int]]]:
    """Hash one derived backup only when its path and inode are immutable-safe."""
    if not _existing_path_is_canonical_staging_descendant(path, staging_dir):
        return None
    if repair_temp_aliases:
        _remove_private_temp_aliases(path)
    try:
        before = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        return None
    data, _mode, file_id = _file_snapshot(path)
    try:
        after = os.lstat(path)
    except OSError:
        return None
    if file_id != (after.st_dev, after.st_ino) or after.st_nlink != 1:
        return None
    return _bytes_sha256(data), (after.st_dev, after.st_ino)


def _immutable_backup_sha256(path: str, staging_dir: str) -> Optional[str]:
    snapshot = _immutable_backup_snapshot(path, staging_dir)
    return snapshot[0] if snapshot is not None else None


def _decode_wal_targets(
    staging_dir: str, wal: Dict[str, Any]
) -> List[_TransactionTarget]:
    kind = wal.get("kind")
    if kind not in {"skills", "legacy"}:
        raise StagingError("adoption recovery journal has an invalid transaction kind")
    raw_targets = wal.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise StagingError("adoption recovery journal has no targets")
    targets: List[_TransactionTarget] = []
    seen: set[str] = set()
    seen_live: set[str] = set()
    seen_live_collisions: set[str] = set()
    target_schema = {
        "key",
        "live_path",
        "expected_realpath",
        "expected_basename",
        "proposed_sha256",
        "original_b64",
        "original_mode",
        "original_file_id",
        "baseline_sha256",
        "backup_path",
        "created_dirs",
    }
    if wal.get("version") == _WAL_VERSION:
        target_schema.add("created_dir_ids")
    for index, row in enumerate(raw_targets):
        if not isinstance(row, dict):
            raise StagingError("adoption recovery journal target must be an object")
        if set(row) != target_schema:
            raise StagingError("adoption recovery journal target has an invalid schema")
        key = row.get("key")
        live = _safe_live_path(row.get("live_path"))
        expected_realpath = _safe_live_path(row.get("expected_realpath"))
        expected_basename = row.get("expected_basename")
        proposed_sha256 = row.get("proposed_sha256")
        baseline_sha256 = row.get("baseline_sha256")
        backup_path = row.get("backup_path")
        raw_created_dirs = row.get("created_dirs", [])
        raw_created_dir_ids = row.get("created_dir_ids", [])
        if not isinstance(key, str) or not key or key in seen:
            raise StagingError("adoption recovery journal has invalid target keys")
        seen.add(key)
        if not live or not expected_realpath:
            raise StagingError("adoption recovery journal has an unsafe live path")
        live_key = _path_identity_key(live)
        collision_key = _filesystem_key(live)
        if live_key in seen_live or collision_key in seen_live_collisions:
            raise StagingError("adoption recovery journal repeats a live target")
        seen_live.add(live_key)
        seen_live_collisions.add(collision_key)
        if _path_identity_key(expected_realpath) != live_key:
            raise StagingError("adoption recovery journal target identity is invalid")
        if expected_basename not in {"SKILL.md", "CLAUDE.md"}:
            raise StagingError("adoption recovery journal has an unsafe target basename")
        if not _valid_sha256_pin(proposed_sha256):
            raise StagingError("adoption recovery journal has an invalid proposal hash")
        if baseline_sha256 != "" and not _valid_sha256_pin(baseline_sha256):
            raise StagingError("adoption recovery journal has an invalid baseline hash")
        original = _from_b64(row.get("original_b64"), field=f"targets[{index}]")
        if _bytes_sha256(original) != baseline_sha256:
            raise StagingError("adoption recovery journal baseline bytes do not match")
        original_mode = row.get("original_mode")
        if original_mode is not None and (
            type(original_mode) is not int
            or original_mode < 0
            or original_mode > 0o7777
        ):
            raise StagingError("adoption recovery journal has an invalid file mode")
        raw_file_id = row.get("original_file_id")
        file_id = None
        if raw_file_id is not None:
            if not (
                isinstance(raw_file_id, list)
                and len(raw_file_id) == 2
                and all(type(value) is int and value >= 0 for value in raw_file_id)
            ):
                raise StagingError("adoption recovery journal has an invalid file id")
            file_id = (raw_file_id[0], raw_file_id[1])
        if original is None:
            if (
                original_mode is not None
                or file_id is not None
                or baseline_sha256 != ""
                or backup_path != ""
            ):
                raise StagingError(
                    "adoption recovery journal absent target has metadata"
                )
        elif original_mode is None or file_id is None or not backup_path:
            raise StagingError(
                "adoption recovery journal existing target is missing metadata"
            )
        if not isinstance(backup_path, str):
            raise StagingError("adoption recovery journal has an invalid backup path")
        if backup_path and not _path_is_within(backup_path, staging_dir):
            raise StagingError("adoption recovery journal backup escapes staging")
        if kind == "skills":
            name = _safe_skill_name(os.path.basename(os.path.dirname(live)))
            expected_key = f"skill {name!r}" if name else ""
            expected_backup = os.path.join(
                staging_dir, "backup", "skills", name, "SKILL.md"
            ) if name else ""
            if (
                expected_basename != "SKILL.md"
                or key != expected_key
                or raw_created_dirs
            ):
                raise StagingError("adoption recovery journal skill target is invalid")
        else:
            label = "skill" if expected_basename == "SKILL.md" else "memory"
            expected_key = f"legacy {label}"
            expected_backup = os.path.join(staging_dir, "backup", expected_basename)
            if key != expected_key:
                raise StagingError("adoption recovery journal legacy target is invalid")
        derived_backup = expected_backup if original is not None else ""
        if _path_identity_key(backup_path) != _path_identity_key(derived_backup):
            raise StagingError("adoption recovery journal backup is not derived")
        if not isinstance(raw_created_dirs, list) or any(
            not isinstance(path, str) or not os.path.isabs(path)
            for path in raw_created_dirs
        ):
            raise StagingError("adoption recovery journal has invalid created directories")
        if raw_created_dirs:
            if _path_identity_key(raw_created_dirs[-1]) != _path_identity_key(
                os.path.dirname(live)
            ) or any(
                _path_identity_key(os.path.dirname(child))
                != _path_identity_key(parent)
                for parent, child in zip(raw_created_dirs, raw_created_dirs[1:])
            ):
                raise StagingError(
                    "adoption recovery journal created directories are not derived"
                )
        if wal.get("version") == _WAL_VERSION:
            if (
                not isinstance(raw_created_dir_ids, list)
                or len(raw_created_dir_ids) != len(raw_created_dirs)
            ):
                raise StagingError(
                    "adoption recovery journal created directory identities are invalid"
                )
            created_dir_ids: List[Optional[tuple[int, int]]] = []
            for raw_id in raw_created_dir_ids:
                if raw_id is None:
                    created_dir_ids.append(None)
                elif (
                    isinstance(raw_id, list)
                    and len(raw_id) == 2
                    and all(type(value) is int and value >= 0 for value in raw_id)
                ):
                    created_dir_ids.append((raw_id[0], raw_id[1]))
                else:
                    raise StagingError(
                        "adoption recovery journal created directory identity is invalid"
                    )
        else:
            created_dir_ids = [None] * len(raw_created_dirs)
        targets.append(_TransactionTarget(
            key=key,
            live_path=live,
            expected_realpath=expected_realpath,
            expected_basename=expected_basename,
            proposed_bytes=b"",
            proposed_sha256=proposed_sha256,
            original_bytes=original,
            original_mode=original_mode,
            original_file_id=file_id,
            baseline_sha256=baseline_sha256,
            backup_path=backup_path,
            created_dirs=tuple(raw_created_dirs),
            created_dir_ids=tuple(created_dir_ids),
        ))
    return targets


def _decode_wal_receipt(
    staging_dir: str,
    wal: Dict[str, Any],
) -> tuple[
    str,
    Optional[bytes],
    Optional[int],
    Optional[tuple[int, int]],
    str,
    str,
]:
    row = wal.get("receipt")
    if not isinstance(row, dict):
        raise StagingError("adoption recovery journal receipt must be an object")
    if set(row) != {
        "path",
        "original_b64",
        "original_mode",
        "original_file_id",
        "baseline_sha256",
        "proposed_sha256",
    }:
        raise StagingError("adoption recovery journal receipt has an invalid schema")
    path = row.get("path")
    kind = wal.get("kind")
    if kind not in {"skills", "legacy"}:
        raise StagingError("adoption recovery journal has an invalid transaction kind")
    expected_name = (
        "adopted_skills.json" if kind == "skills" else "adopted_legacy.json"
    )
    expected_path = os.path.join(staging_dir, expected_name)
    if not isinstance(path, str) or _path_identity_key(path) != _path_identity_key(expected_path):
        raise StagingError("adoption recovery journal has an invalid receipt path")
    original = _from_b64(row.get("original_b64"), field="receipt")
    mode = row.get("original_mode")
    if mode is not None and (
        type(mode) is not int or mode < 0 or mode > 0o7777
    ):
        raise StagingError("adoption recovery journal has an invalid receipt mode")
    raw_file_id = row.get("original_file_id")
    file_id = None
    if raw_file_id is not None:
        if not (
            isinstance(raw_file_id, list)
            and len(raw_file_id) == 2
            and all(type(value) is int and value >= 0 for value in raw_file_id)
        ):
            raise StagingError(
                "adoption recovery journal has an invalid receipt file id"
            )
        file_id = (raw_file_id[0], raw_file_id[1])
    if original is None and (mode is not None or file_id is not None):
        raise StagingError("adoption recovery journal absent receipt has metadata")
    if original is not None and (mode is None or file_id is None):
        raise StagingError(
            "adoption recovery journal existing receipt is missing metadata"
        )
    baseline = row.get("baseline_sha256")
    proposed = row.get("proposed_sha256")
    if _bytes_sha256(original) != baseline or not _valid_sha256_pin(proposed):
        raise StagingError("adoption recovery journal has invalid receipt hashes")
    return path, original, mode, file_id, baseline, proposed


def _ensure_safe_directory(path: str, staging_dir: str) -> None:
    """Create one staging descendant and reject symlinks and junction escapes."""
    if not _path_is_within(path, staging_dir):
        raise StagingError(f"backup directory escapes staging: {path}")
    created = False
    try:
        os.mkdir(path, 0o700)
        created = True
    except FileExistsError:
        pass
    info = os.lstat(path)
    if _is_link_or_junction(path) or not stat.S_ISDIR(info.st_mode):
        raise StagingError(f"backup directory is unsafe: {path}")
    staging_real = os.path.realpath(staging_dir)
    path_real = os.path.realpath(path)
    if not _path_is_within(path_real, staging_real):
        raise StagingError(f"backup directory escapes staging through a junction: {path}")
    if created:
        _fsync_parent(path)
        _fsync_directory(path)


def _prepare_backup_parent(backup_path: str, staging_dir: str) -> None:
    relative = os.path.relpath(os.path.dirname(backup_path), staging_dir)
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise StagingError(f"backup path escapes staging: {backup_path}")
    current = staging_dir
    for component in relative.split(os.sep):
        if not component or component == os.curdir:
            continue
        current = os.path.join(current, component)
        _ensure_safe_directory(current, staging_dir)


def _recover_target(
    target: _TransactionTarget,
    *,
    expected_proposal_file_id: Optional[tuple[int, int]] = None,
) -> None:
    if target.original_bytes is None and not os.path.lexists(target.live_path):
        current_realpath = _canonical_live_path(target.live_path)
        if (
            not current_realpath
            or _path_identity_key(current_realpath)
            != _path_identity_key(target.expected_realpath)
            or _path_identity_key(current_realpath)
            != _path_identity_key(target.live_path)
        ):
            raise StagingError(
                f"recovery conflict for {target.key}: target path identity changed"
            )
        return
    _adopt_target_ok(
        target.key,
        target.live_path,
        target.expected_realpath,
        expected_basename=target.expected_basename,
    )
    current, current_mode, current_file_id = _file_snapshot(target.live_path)
    current_sha256 = _bytes_sha256(current)
    if current_sha256 == target.baseline_sha256:
        if current is not None and not _modes_match(
            current_mode, target.original_mode
        ):
            raise StagingError(
                f"recovery conflict for {target.key}: baseline mode changed"
            )
        return
    if current_sha256 != target.proposed_sha256:
        raise StagingError(
            f"recovery conflict for {target.key}: live content is neither baseline nor proposal"
        )
    if (
        expected_proposal_file_id is not None
        and current_file_id != expected_proposal_file_id
    ):
        raise StagingError(
            f"recovery conflict for {target.key}: proposal identity changed"
        )
    proposal_mode = target.original_mode if target.original_mode is not None else 0o600
    if not _modes_match(current_mode, proposal_mode):
        raise StagingError(
            f"recovery conflict for {target.key}: proposal mode changed"
        )
    # Narrow the rollback check/use window. Cooperative adopters also hold the
    # target lock; an uncooperative editor is detected whenever it wins before
    # this final read.
    final, final_mode, final_file_id = _file_snapshot(target.live_path)
    if (
        _bytes_sha256(final) != target.proposed_sha256
        or final_file_id != current_file_id
        or not _modes_match(final_mode, current_mode)
    ):
        raise StagingError(f"recovery conflict for {target.key}: live content changed")
    if target.original_bytes is None:
        _unlink_fsync(target.live_path)
    else:
        _write_atomic_bytes(
            target.live_path,
            target.original_bytes,
            create_parents=False,
            mode=target.original_mode,
        )


def _recover_receipt(
    path: str,
    original: Optional[bytes],
    original_mode: Optional[int],
    _original_file_id: Optional[tuple[int, int]],
    baseline_sha256: str,
    proposed_sha256: str,
) -> None:
    current, current_mode, current_file_id = _file_snapshot(path)
    current_sha256 = _bytes_sha256(current)
    if current_sha256 == baseline_sha256:
        if current is not None and not _modes_match(current_mode, original_mode):
            raise StagingError("recovery conflict: adoption receipt mode changed")
        return
    if current_sha256 != proposed_sha256:
        raise StagingError(
            "recovery conflict: adoption receipt is neither baseline nor transaction receipt"
        )
    proposal_mode = original_mode if original_mode is not None else 0o600
    if not _modes_match(current_mode, proposal_mode):
        raise StagingError("recovery conflict: adoption receipt mode changed")
    final, final_mode, final_file_id = _file_snapshot(path)
    if (
        _bytes_sha256(final) != proposed_sha256
        or final_file_id != current_file_id
        or not _modes_match(final_mode, current_mode)
    ):
        raise StagingError("recovery conflict: adoption receipt changed")
    if original is None:
        _unlink_fsync(path)
    else:
        _write_atomic_bytes(path, original, mode=original_mode)


def _cleanup_transaction_backups(
    targets: Sequence[_TransactionTarget], staging_dir: str
) -> List[str]:
    errors: List[str] = []
    for target in reversed(targets):
        path = target.backup_path
        if not path or not os.path.lexists(path):
            continue
        try:
            snapshot = _immutable_backup_snapshot(
                path, staging_dir, repair_temp_aliases=True
            )
            if snapshot is None:
                raise StagingError(f"transaction backup is not a regular file: {path}")
            backup_sha256, file_id = snapshot
            if backup_sha256 != target.baseline_sha256:
                raise StagingError(f"transaction backup changed during recovery: {path}")
            current = os.lstat(path)
            if (
                (current.st_dev, current.st_ino) != file_id
                or current.st_nlink != 1
                or not _existing_path_is_canonical_staging_descendant(
                    path, staging_dir
                )
            ):
                raise StagingError(f"transaction backup identity changed: {path}")
            _unlink_fsync(path)
        except BaseException as exc:
            errors.append(f"backup {path}: {type(exc).__name__}: {exc}")
    return errors


def _cleanup_created_directories(
    targets: Sequence[_TransactionTarget],
) -> List[str]:
    """Remove only empty directories whose creation identity was journaled."""
    identities: Dict[str, tuple[str, Optional[tuple[int, int]]]] = {}
    ordered: List[str] = []
    for target in targets:
        ids = target.created_dir_ids or tuple(None for _path in target.created_dirs)
        for path, file_id in zip(target.created_dirs, ids):
            key = _path_identity_key(path)
            existing = identities.get(key)
            if existing is not None and existing[1] != file_id:
                return [f"created directory {path}: journal identities disagree"]
            if existing is None:
                identities[key] = (path, file_id)
                ordered.append(key)

    errors: List[str] = []
    for key in reversed(ordered):
        path, expected_id = identities[key]
        if expected_id is None or not os.path.lexists(path):
            # Old journals and crashes before the identity rewrite fail closed.
            continue
        try:
            info = os.lstat(path)
            if (
                _is_link_or_junction(path)
                or not stat.S_ISDIR(info.st_mode)
                or (info.st_dev, info.st_ino) != expected_id
            ):
                raise StagingError(
                    f"created directory identity changed during recovery: {path}"
                )
            os.rmdir(path)
            _fsync_parent(path)
        except BaseException as exc:
            errors.append(
                f"created directory {path}: {type(exc).__name__}: {exc}"
            )
    return errors


def _recover_transaction_locked(
    staging_dir: str,
    wal: Dict[str, Any],
    *,
    expected_proposal_file_ids: Optional[
        Dict[str, tuple[int, int]]
    ] = None,
) -> List[str]:
    """Idempotently roll back one WAL while its staging and target locks are held."""
    try:
        targets = _decode_wal_targets(staging_dir, wal)
        receipt = _decode_wal_receipt(staging_dir, wal)
    except BaseException as exc:
        return [f"invalid recovery journal: {type(exc).__name__}: {exc}"]
    errors: List[str] = []
    for target in reversed(targets):
        try:
            _recover_target(
                target,
                expected_proposal_file_id=(
                    expected_proposal_file_ids or {}
                ).get(target.key),
            )
        except BaseException as exc:
            errors.append(f"target {target.key}: {type(exc).__name__}: {exc}")
    try:
        _recover_receipt(*receipt)
    except BaseException as exc:
        errors.append(f"receipt: {type(exc).__name__}: {exc}")
    if errors:
        return errors
    errors.extend(_cleanup_transaction_backups(targets, staging_dir))
    if errors:
        return errors
    errors.extend(_cleanup_created_directories(targets))
    if errors:
        return errors
    try:
        _remove_transaction_wal(staging_dir, expected_wal=wal)
    except BaseException as exc:
        errors.append(f"journal removal: {type(exc).__name__}: {exc}")
    return errors


def _verify_published_targets(
    targets: Sequence[_TransactionTarget],
    published_file_ids: Dict[str, tuple[int, int]],
) -> None:
    """Verify the complete live set immediately around receipt publication.

    Portable filesystems do not expose a conditional replace operation, so the
    final check-to-commit microgap is unavoidable. Rechecking the whole set on
    both sides of receipt publication nevertheless detects edits made while
    later targets or the receipt itself were being written.
    """
    for target in targets:
        _adopt_target_ok(
            target.key,
            target.live_path,
            target.expected_realpath,
            expected_basename=target.expected_basename,
        )
        current, current_mode, current_file_id = _file_snapshot(target.live_path)
        proposal_mode = (
            target.original_mode if target.original_mode is not None else 0o600
        )
        if (
            _bytes_sha256(current) != target.proposed_sha256
            or not _modes_match(current_mode, proposal_mode)
            or current_file_id != published_file_ids.get(target.key)
        ):
            raise StagingError(
                f"live target for {target.key} changed after publication"
            )


def _execute_transaction_locked(
    staging_dir: str,
    *,
    kind: str,
    targets: Sequence[_TransactionTarget],
    receipt_path: str,
    receipt_original: Optional[bytes],
    receipt_mode: Optional[int],
    receipt_file_id: Optional[tuple[int, int]],
    receipt_after: bytes,
) -> None:
    wal = _transaction_wal(
        kind=kind,
        targets=targets,
        receipt_path=receipt_path,
        receipt_original=receipt_original,
        receipt_mode=receipt_mode,
        receipt_file_id=receipt_file_id,
        receipt_after=receipt_after,
    )
    _write_transaction_wal(staging_dir, wal)
    published_file_ids: Dict[str, tuple[int, int]] = {}
    try:
        planned_directories: List[str] = []
        seen_directories: set[str] = set()
        for target in targets:
            for directory in target.created_dirs:
                key = _path_identity_key(directory)
                if key not in seen_directories:
                    seen_directories.add(key)
                    planned_directories.append(directory)
        for directory in planned_directories:
            try:
                os.mkdir(directory, 0o700)
                _fsync_parent(directory)
                _fsync_directory(directory)
            except FileExistsError as exc:
                raise StagingError(
                    f"live directory appeared during adoption: {directory}"
                ) from exc
            info = os.lstat(directory)
            if _is_link_or_junction(directory) or not stat.S_ISDIR(info.st_mode):
                raise StagingError(
                    f"live directory changed during adoption: {directory}"
                )
            created_id = (info.st_dev, info.st_ino)
            for target in targets:
                ids = list(
                    target.created_dir_ids
                    or tuple(None for _path in target.created_dirs)
                )
                for index, path in enumerate(target.created_dirs):
                    if _path_identity_key(path) == _path_identity_key(directory):
                        ids[index] = created_id
                target.created_dir_ids = tuple(ids)
            replacement_wal = _transaction_wal(
                kind=kind,
                targets=targets,
                receipt_path=receipt_path,
                receipt_original=receipt_original,
                receipt_mode=receipt_mode,
                receipt_file_id=receipt_file_id,
                receipt_after=receipt_after,
            )
            previous_wal = wal
            wal = replacement_wal
            _rewrite_transaction_wal(
                staging_dir,
                expected_wal=previous_wal,
                replacement_wal=replacement_wal,
            )

        for target in targets:
            if target.backup_path:
                _prepare_backup_parent(target.backup_path, staging_dir)
                _write_new_bytes(
                    target.backup_path,
                    target.original_bytes or b"",
                    mode=target.original_mode,
                )
                backup_sha256 = _immutable_backup_sha256(
                    target.backup_path, staging_dir
                )
                if backup_sha256 != target.baseline_sha256:
                    raise StagingError(
                        f"immutable backup for {target.key} was not published safely"
                    )
            _adopt_target_ok(
                target.key,
                target.live_path,
                target.expected_realpath,
                expected_basename=target.expected_basename,
            )
            current, current_mode, current_file_id = _file_snapshot(target.live_path)
            if (
                _bytes_sha256(current) != target.baseline_sha256
                or current_file_id != target.original_file_id
                or not _modes_match(current_mode, target.original_mode)
            ):
                raise StagingError(
                    f"live target for {target.key} changed bytes, mode, or identity; "
                    "discard and rerun this night"
                )
            # This final snapshot narrows, but cannot portably eliminate, the
            # pre-replace editor race: Python exposes no cross-platform
            # compare-and-swap rename. The whole published set is checked again
            # on both sides of receipt publication below.
            _write_atomic(
                target.live_path,
                target.proposed_bytes.decode("utf-8"),
                create_parents=False,
            )
            current, current_mode, current_file_id = _file_snapshot(
                target.live_path
            )
            proposal_mode = (
                target.original_mode if target.original_mode is not None else 0o600
            )
            if (
                _bytes_sha256(current) != target.proposed_sha256
                or not _modes_match(current_mode, proposal_mode)
            ):
                raise StagingError(
                    f"live target for {target.key} changed during publication"
                )
            if current_file_id is None:
                raise StagingError(
                    f"live target for {target.key} disappeared during publication"
                )
            published_file_ids[target.key] = current_file_id
        _verify_published_targets(targets, published_file_ids)
        receipt_current, receipt_current_mode, receipt_current_file_id = (
            _file_snapshot(receipt_path)
        )
        if (
            _bytes_sha256(receipt_current) != _bytes_sha256(receipt_original)
            or receipt_current_file_id != receipt_file_id
            or not _modes_match(receipt_current_mode, receipt_mode)
        ):
            raise StagingError(
                "adoption receipt changed bytes, mode, or identity during adoption"
            )
        _write_atomic(
            receipt_path,
            receipt_after.decode("utf-8"),
            create_parents=False,
        )
        receipt_current, receipt_current_mode, _receipt_file_id = _file_snapshot(
            receipt_path
        )
        receipt_proposal_mode = receipt_mode if receipt_mode is not None else 0o600
        if (
            _bytes_sha256(receipt_current) != hashlib.sha256(receipt_after).hexdigest()
            or not _modes_match(receipt_current_mode, receipt_proposal_mode)
        ):
            raise StagingError("adoption receipt changed during publication")
        _verify_published_targets(targets, published_file_ids)
        # WAL unlink + parent fsync is the transaction commit point.
        _remove_transaction_wal(staging_dir, expected_wal=wal)
    except BaseException as primary:
        recovery_errors = _recover_transaction_locked(
            staging_dir,
            wal,
            expected_proposal_file_ids=published_file_ids,
        )
        if recovery_errors:
            if not os.path.lexists(_wal_path(staging_dir)):
                try:
                    _write_transaction_wal(staging_dir, wal)
                except BaseException as exc:
                    recovery_errors.append(
                        f"could not restore recovery journal: {type(exc).__name__}: {exc}"
                    )
            raise StagingRecoveryError(
                "adoption failed and automatic rollback was incomplete; "
                f"journal and backups retained at {staging_dir}",
                primary=primary,
                recovery_errors=recovery_errors,
            ) from primary
        raise


@contextmanager
def _adoption_locks(staging_dir: str, live_paths: Sequence[str]):
    staging_lock = os.path.join(staging_dir, ".adopt-skills.lock")
    with _exclusive_create_locks([staging_lock]):
        wal = _read_transaction_wal(staging_dir)
        recovery_paths: List[str] = []
        if wal is not None:
            recovery_paths = [
                target.live_path for target in _decode_wal_targets(staging_dir, wal)
            ]
        all_paths = list(live_paths) + recovery_paths
        with _exclusive_create_locks(_target_lock_paths(all_paths)):
            if wal is not None:
                recovery_errors = _recover_transaction_locked(staging_dir, wal)
                if recovery_errors:
                    raise StagingRecoveryError(
                        "cannot recover an interrupted adoption; journal and backups retained",
                        recovery_errors=recovery_errors,
                    )
            yield


def _recover_before_manifest(staging_dir: str) -> None:
    """Recover a WAL without depending on a still-readable staging manifest."""
    if _is_link_or_junction(staging_dir) or not os.path.isdir(staging_dir):
        raise StagingError(f"staging directory is unsafe: {staging_dir}")
    staging_lock = os.path.join(staging_dir, ".adopt-skills.lock")
    with _exclusive_create_locks([staging_lock]):
        wal = _read_transaction_wal(staging_dir)
        if wal is None:
            return
        targets = _decode_wal_targets(staging_dir, wal)
        with _exclusive_create_locks(
            _target_lock_paths([target.live_path for target in targets])
        ):
            recovery_errors = _recover_transaction_locked(staging_dir, wal)
            if recovery_errors:
                raise StagingRecoveryError(
                    "cannot recover an interrupted adoption; journal and backups retained",
                    recovery_errors=recovery_errors,
                )


def adopt_skills(
    staging_dir: str, skill_names: Optional[Sequence[str]] = None
) -> List[AdoptedSkill]:
    """Durably adopt a reviewed per-skill subset with restart-safe rollback."""
    staging_dir = _canonical_staging_dir(staging_dir)
    _recover_before_manifest(staging_dir)
    initial_all_rows = staged_skills(staging_dir)
    initial_rows = _selected_rows(initial_all_rows, skill_names)
    if not initial_rows:
        return []
    skill_roots = staged_skill_roots(staging_dir)
    initial_live_paths: List[str] = []
    for row in initial_rows:
        live = _safe_live_path(row.get("live_skill_path"))
        if not live:
            raise StagingError(
                f"unsafe live skill path: {row.get('live_skill_path')!r}"
            )
        initial_live_paths.append(live)

    with _adoption_locks(staging_dir, initial_live_paths):
        all_rows = staged_skills(staging_dir)
        rows = _selected_rows(all_rows, skill_names)
        locked_paths = {_path_identity_key(path) for path in initial_live_paths}
        current_paths = {
            _path_identity_key(str(row.get("live_skill_path") or ""))
            for row in rows
        }
        if locked_paths != current_paths:
            raise StagingError("staging manifest changed while adoption was locking")
        _revalidate_selected_skill_rows(rows, all_rows=all_rows)

        receipt_path = os.path.join(staging_dir, "adopted_skills.json")
        existing_receipts, receipt_original, receipt_mode, receipt_file_id = (
            _read_existing_receipts(receipt_path, staging_dir)
        )
        already_adopted = {
            str(row["skill_name"]) for row in existing_receipts
        }

        targets: List[_TransactionTarget] = []
        receipts: List[AdoptedSkill] = []
        backup_dir = os.path.join(staging_dir, "backup", "skills")
        for row in rows:
            name = _safe_skill_name(row.get("skill_name"))
            if not name:
                raise StagingError(
                    f"unsafe staged skill name: {row.get('skill_name')!r}"
                )
            if name in already_adopted:
                raise StagingError(
                    f"staged skill {name!r} was already adopted from this night"
                )
            live = _safe_live_path(row.get("live_skill_path"))
            expected_realpath = _safe_live_path(row.get("live_realpath"))
            live_pin = row.get("live_sha256")
            if not live or not expected_realpath or (
                live_pin != "" and not _valid_sha256_pin(live_pin)
            ):
                raise StagingError(
                    f"staged skill {name!r} is missing safe live baseline pins; "
                    "discard and restage this night"
                )
            _adopt_live_target_ok(name, live, expected_realpath, skill_roots)

            expected_file = proposal_filename(name)
            if row.get("proposed_file") != expected_file:
                raise StagingError(
                    f"unsafe staged proposal filename for {name!r}: "
                    f"{row.get('proposed_file')!r}; expected {expected_file!r}"
                )
            staged = os.path.join(staging_dir, expected_file)
            if _is_link_or_junction(staged) or not os.path.isfile(staged):
                raise StagingError(
                    f"staged proposal for {name!r} is missing or a symlink: {staged}"
                )
            try:
                with open(staged, "rb") as handle:
                    proposed_bytes = handle.read()
                proposed_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise StagingError(
                    f"staged proposal for {name!r} must be valid UTF-8"
                ) from exc
            except OSError as exc:
                raise StagingError(
                    f"could not read staged proposal for {name!r}: {staged}"
                ) from exc
            pin = row.get("sha256")
            if not _valid_sha256_pin(pin):
                raise StagingError(
                    f"staged proposal for {name!r} is missing a sha256 pin"
                )
            if hashlib.sha256(proposed_bytes).hexdigest() != pin:
                raise StagingError(
                    f"staged proposal for {name!r} does not match its manifest sha256"
                )
            if not proposed_bytes.strip():
                raise StagingError(f"staged proposal for {name!r} is empty")

            original, original_mode, original_file_id = _file_snapshot(live)
            if _bytes_sha256(original) != live_pin:
                raise StagingError(
                    f"live skill for {name!r} changed since staging; "
                    "discard and rerun this night"
                )
            backup_path = (
                os.path.join(backup_dir, name, "SKILL.md")
                if original is not None
                else ""
            )
            if backup_path and os.path.lexists(backup_path):
                raise StagingError(
                    f"immutable backup already exists for {name!r}: {backup_path}"
                )
            targets.append(_TransactionTarget(
                key=f"skill {name!r}",
                live_path=live,
                expected_realpath=expected_realpath,
                expected_basename="SKILL.md",
                proposed_bytes=proposed_bytes,
                proposed_sha256=pin,
                original_bytes=original,
                original_mode=original_mode,
                original_file_id=original_file_id,
                baseline_sha256=live_pin,
                backup_path=backup_path,
            ))
            receipts.append(AdoptedSkill(
                skill_name=name,
                live_skill_path=live,
                sha256_before=live_pin,
                sha256_after=pin,
                backup_path=backup_path,
            ))

        combined_receipts = existing_receipts + [
            receipt.__dict__ for receipt in receipts
        ]
        receipt_after = json.dumps(
            combined_receipts,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        _execute_transaction_locked(
            staging_dir,
            kind="skills",
            targets=targets,
            receipt_path=receipt_path,
            receipt_original=receipt_original,
            receipt_mode=receipt_mode,
            receipt_file_id=receipt_file_id,
            receipt_after=receipt_after,
        )
        return receipts


def _load_manifest(staging_dir: str) -> Dict[str, Any]:
    path = os.path.join(staging_dir, "manifest.json")
    if _is_link_or_junction(path):
        raise StagingError("staging manifest must not be a symlink")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StagingError(f"cannot read staging manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise StagingError("staging manifest must be a JSON object")
    _manifest_schema_version(payload)
    return payload


def _legacy_rows(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    version = _manifest_schema_version(manifest)
    if version == _MANIFEST_VERSION:
        if (
            manifest.get("has_skill") is not False
            or manifest.get("has_memory") is not False
        ):
            raise StagingError(
                "versioned staging manifest has unsafe legacy compatibility flags"
            )
        skill_flag = manifest.get("has_managed_skill")
        memory_flag = manifest.get("has_managed_memory")
        if type(skill_flag) is not bool or type(memory_flag) is not bool:
            raise StagingError("versioned staging manifest has invalid managed flags")
        flags = (("skill", skill_flag), ("memory", memory_flag))
        if not skill_flag and not memory_flag:
            return {}
    else:
        flags = (("skill", bool(manifest.get("has_skill"))),
                 ("memory", bool(manifest.get("has_memory"))))
    raw = manifest.get("legacy")
    if not isinstance(raw, dict):
        raise StagingError(
            "legacy staging manifest is missing integrity pins; discard and restage"
        )
    rows: Dict[str, Dict[str, Any]] = {}
    for label, present in flags:
        if not present:
            continue
        row = raw.get(label)
        if not isinstance(row, dict):
            raise StagingError(
                f"legacy {label} staging row is missing; discard and restage"
            )
        rows[label] = row
    if not rows:
        return {}
    return rows


def _read_legacy_receipts(
    path: str,
    staging_dir: str,
) -> tuple[
    List[Dict[str, Any]],
    Optional[bytes],
    Optional[int],
    Optional[tuple[int, int]],
]:
    payload, original, mode, file_id = _read_receipt_file(path)
    seen: set[str] = set()
    schema = {
        "target",
        "live_path",
        "sha256_before",
        "sha256_after",
        "backup_path",
    }
    for row in payload:
        if set(row) != schema:
            raise StagingError("existing legacy adoption receipt has an invalid schema")
        label = row.get("target")
        live = _safe_live_path(row.get("live_path"))
        before = row.get("sha256_before")
        after = row.get("sha256_after")
        backup = row.get("backup_path")
        if label not in {"skill", "memory"} or label in seen:
            raise StagingError("existing legacy adoption receipt is invalid")
        seen.add(label)
        expected_basename = "SKILL.md" if label == "skill" else "CLAUDE.md"
        if (
            not live
            or os.path.basename(live) != expected_basename
            or (before != "" and not _valid_sha256_pin(before))
        ):
            raise StagingError("existing legacy adoption receipt is invalid")
        if not _valid_sha256_pin(after) or not isinstance(backup, str):
            raise StagingError("existing legacy adoption receipt is invalid")
        if before == "":
            if backup:
                raise StagingError("existing legacy adoption receipt has an unexpected backup")
            continue
        expected = os.path.join(
            staging_dir,
            "backup",
            "SKILL.md" if label == "skill" else "CLAUDE.md",
        )
        if _path_identity_key(backup) != _path_identity_key(expected):
            raise StagingError("existing legacy adoption receipt has an invalid backup")
        backup_sha256 = _immutable_backup_sha256(expected, staging_dir)
        if backup_sha256 is None:
            raise StagingError("immutable legacy backup is missing")
        if backup_sha256 != before:
            raise StagingError("immutable legacy backup changed")
    return payload, original, mode, file_id


def adopt(staging_dir: str) -> List[str]:
    """Durably adopt the pinned legacy SKILL.md/CLAUDE.md proposal pair."""
    staging_dir = _canonical_staging_dir(staging_dir)
    _recover_before_manifest(staging_dir)
    initial_manifest = _load_manifest(staging_dir)
    initial_rows = _legacy_rows(initial_manifest)
    if not initial_rows:
        return []
    initial_paths: List[str] = []
    for row in initial_rows.values():
        live = _safe_live_path(row.get("live_path"))
        if not live:
            raise StagingError("legacy staging row has an unsafe live path")
        initial_paths.append(live)

    with _adoption_locks(staging_dir, initial_paths):
        manifest = _load_manifest(staging_dir)
        rows = _legacy_rows(manifest)
        current_paths = {
            _path_identity_key(str(row.get("live_path") or ""))
            for row in rows.values()
        }
        if current_paths != {_path_identity_key(path) for path in initial_paths}:
            raise StagingError("legacy staging manifest changed while adoption was locking")

        receipt_path = os.path.join(staging_dir, "adopted_legacy.json")
        existing, receipt_original, receipt_mode, receipt_file_id = (
            _read_legacy_receipts(receipt_path, staging_dir)
        )
        already = {str(row["target"]) for row in existing}
        targets: List[_TransactionTarget] = []
        new_receipts: List[Dict[str, Any]] = []
        updated: List[str] = []
        for label in ("skill", "memory"):
            row = rows.get(label)
            if row is None:
                continue
            if label in already:
                raise StagingError(f"legacy {label} was already adopted from this night")
            expected_file = (
                "proposed_SKILL.md" if label == "skill" else "proposed_CLAUDE.md"
            )
            expected_basename = "SKILL.md" if label == "skill" else "CLAUDE.md"
            live = _safe_live_path(row.get("live_path"))
            expected_realpath = _safe_live_path(row.get("live_realpath"))
            live_pin = row.get("live_sha256")
            proposal_pin = row.get("sha256")
            if (
                row.get("proposed_file") != expected_file
                or not live
                or not expected_realpath
                or (live_pin != "" and not _valid_sha256_pin(live_pin))
                or not _valid_sha256_pin(proposal_pin)
            ):
                raise StagingError(
                    f"legacy {label} staging pins are invalid; discard and restage"
                )
            created_dirs = _planned_live_directories(
                f"legacy {label}",
                live,
                expected_realpath,
                expected_basename=expected_basename,
            )
            if not created_dirs:
                _adopt_target_ok(
                    f"legacy {label}",
                    live,
                    expected_realpath,
                    expected_basename=expected_basename,
                )
            staged = os.path.join(staging_dir, expected_file)
            if _is_link_or_junction(staged) or not os.path.isfile(staged):
                raise StagingError(f"legacy {label} proposal is missing or a symlink")
            try:
                with open(staged, "rb") as handle:
                    proposed = handle.read()
                proposed.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise StagingError(f"legacy {label} proposal must be valid UTF-8") from exc
            if (
                label == "skill" and not proposed.strip()
            ) or hashlib.sha256(proposed).hexdigest() != proposal_pin:
                raise StagingError(f"legacy {label} proposal does not match its sha256")
            original, original_mode, original_file_id = _file_snapshot(live)
            if _bytes_sha256(original) != live_pin:
                raise StagingError(
                    f"legacy {label} changed since staging; discard and restage"
                )
            backup_path = (
                os.path.join(staging_dir, "backup", expected_basename)
                if original is not None
                else ""
            )
            if backup_path and os.path.lexists(backup_path):
                raise StagingError(f"immutable legacy backup already exists: {backup_path}")
            targets.append(_TransactionTarget(
                key=f"legacy {label}",
                live_path=live,
                expected_realpath=expected_realpath,
                expected_basename=expected_basename,
                proposed_bytes=proposed,
                proposed_sha256=proposal_pin,
                original_bytes=original,
                original_mode=original_mode,
                original_file_id=original_file_id,
                baseline_sha256=live_pin,
                backup_path=backup_path,
                created_dirs=created_dirs,
            ))
            new_receipts.append({
                "target": label,
                "live_path": live,
                "sha256_before": live_pin,
                "sha256_after": proposal_pin,
                "backup_path": backup_path,
            })
            updated.append(live)

        receipt_after = json.dumps(
            existing + new_receipts,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        _execute_transaction_locked(
            staging_dir,
            kind="legacy",
            targets=targets,
            receipt_path=receipt_path,
            receipt_original=receipt_original,
            receipt_mode=receipt_mode,
            receipt_file_id=receipt_file_id,
            receipt_after=receipt_after,
        )
        return updated

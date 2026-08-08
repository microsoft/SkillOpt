"""SkillOpt-Sleep — Stage 5/6: staging and adoption.

Implements the Dreams safety contract: the cycle never mutates the user's
live CLAUDE.md / SKILL.md. It writes proposals + a human-readable report into
a staging directory; a separate, explicit `adopt` step copies them over the
live files after taking a backup.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

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


class StagingError(ValueError):
    """A proposal could not be staged safely (bad name, bad target, collision)."""


@dataclass
class SkillProposal:
    """One skill's proposed document plus the live file it would replace."""

    skill_name: str
    proposed_skill: str
    live_skill_path: str


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


def proposal_filename(skill_name: str) -> str:
    """Staged filename for one skill's proposal (unique per skill name)."""
    return f"proposed_SKILL.{skill_name}.md"


def _write_atomic(path: str, text: str) -> None:
    """Write ``text`` to ``path`` atomically, so review never sees half a file."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
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
        live = _safe_live_path(proposal.live_skill_path)
        if not live:
            raise StagingError(
                f"unsafe live skill path for {name!r}: {proposal.live_skill_path!r}"
            )
        if any(row["skill_name"] == name for row in rows):
            raise StagingError(f"duplicate skill name in staging fan-out: {name!r}")
        proposed_file = proposal_filename(name)
        # casefold, not lower: it folds Unicode pairs lower() leaves distinct,
        # which is the comparison a case-insensitive filesystem actually makes.
        file_key = proposed_file.casefold()
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
        live_key = live.casefold()
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
        })
    return rows


def write_skill_proposals(
    out_dir: str, proposals: Iterable[SkillProposal]
) -> List[Dict[str, Any]]:
    """Stage one uniquely named proposal file per skill; return manifest rows.

    Every proposal is validated before anything is written, so a rejected
    fan-out leaves no partial files behind.
    """
    # Materialise once. The signature accepts any Iterable, so a generator is
    # legal input — and it would otherwise be drained by the validation pass,
    # leaving the write loop with nothing to iterate and returning a full set
    # of manifest rows for files that were never created.
    proposals = list(proposals)
    rows = skill_proposal_rows(proposals)
    if not rows:
        return rows
    os.makedirs(out_dir, exist_ok=True)
    for row, proposal in zip(rows, proposals):
        _write_atomic(os.path.join(out_dir, row["proposed_file"]), proposal.proposed_skill)
    return rows


def _ts_dir() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def staging_root(project: str) -> str:
    return os.path.join(project, ".skillopt-sleep", "staging")


def new_staging_dir(project: str) -> str:
    """A staging path that is unique even for two runs in the same second."""
    base = os.path.join(staging_root(project), _ts_dir())
    out, i = base, 2
    while os.path.exists(out):
        out = f"{base}-{i}"
        i += 1
    return out


def latest_staging(project: str) -> Optional[str]:
    root = staging_root(project)
    if not os.path.isdir(root):
        return None
    subs = sorted(
        (os.path.join(root, d) for d in os.listdir(root)),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    for p in subs:
        # Only adoptable folders count: a no-tasks night leaves evidence.jsonl
        # but no manifest, and adopt() needs the manifest.
        if os.path.exists(os.path.join(p, "manifest.json")):
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
    report_md: str,
    out_dir: str = "",
    skill_proposals: Iterable[SkillProposal] = (),
) -> str:
    """Write proposals + report into staging/<ts>/ and return that path.

    ``out_dir`` lets the cycle pre-create the night's staging folder at cycle
    START, so incremental artifacts (evidence.jsonl) accumulate in the same
    place the report lands.

    ``skill_proposals`` stages one extra uniquely named file and manifest row per
    skill for a multi-skill night. Left empty, the staging layout and manifest
    are exactly the legacy single-proposal ones.
    """
    out = out_dir or os.path.join(staging_root(project), _ts_dir())
    os.makedirs(out, exist_ok=True)

    skill_rows = write_skill_proposals(out, skill_proposals)

    manifest = {
        "live_skill_path": live_skill_path,
        "live_memory_path": live_memory_path,
        "has_skill": proposed_skill is not None,
        "has_memory": proposed_memory is not None,
        "accepted": report.accepted,
    }
    if skill_rows:
        manifest["skills"] = skill_rows
    if proposed_skill is not None:
        with open(os.path.join(out, "proposed_SKILL.md"), "w", encoding="utf-8") as f:
            f.write(proposed_skill)
    if proposed_memory is not None:
        with open(os.path.join(out, "proposed_CLAUDE.md"), "w", encoding="utf-8") as f:
            f.write(proposed_memory)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out


def _backup(path: str, backup_dir: str) -> None:
    if os.path.exists(path):
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))


def adopt(staging_dir: str) -> List[str]:
    """Copy staged proposals over the live files, backing up first.

    Returns the list of live paths that were updated.
    """
    with open(os.path.join(staging_dir, "manifest.json")) as f:
        manifest = json.load(f)
    backup_dir = os.path.join(staging_dir, "backup")
    updated: List[str] = []

    if manifest.get("has_skill"):
        live = manifest["live_skill_path"]
        os.makedirs(os.path.dirname(live), exist_ok=True)
        _backup(live, backup_dir)
        shutil.copy2(os.path.join(staging_dir, "proposed_SKILL.md"), live)
        updated.append(live)
    if manifest.get("has_memory"):
        live = manifest["live_memory_path"]
        os.makedirs(os.path.dirname(live), exist_ok=True)
        _backup(live, backup_dir)
        shutil.copy2(os.path.join(staging_dir, "proposed_CLAUDE.md"), live)
        updated.append(live)
    return updated

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
import time
from typing import Any, List, Optional

from skillopt_sleep.types import SleepReport

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


def redact_secrets(value: Any) -> Any:
    """Scrub secret-looking substrings (API keys, bearer tokens, private keys)
    from a string, or recursively from the string leaves of a list/dict.

    Used before writing backend stderr / optimizer replies / task responses to
    on-disk diagnostics: those are surfaced for debugging, but the underlying
    text (e.g. a codex 401 stderr dump) can carry credentials. Non-string
    scalars pass through unchanged.
    """
    if isinstance(value, str):
        out = value
        for pattern, replacement in _SECRET_PATTERNS:
            out = pattern.sub(replacement, out)
        return out
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    if isinstance(value, dict):
        return {k: redact_secrets(v) for k, v in value.items()}
    return value


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
) -> str:
    """Write proposals + report into staging/<ts>/ and return that path.

    ``out_dir`` lets the cycle pre-create the night's staging folder at cycle
    START, so incremental artifacts (evidence.jsonl) accumulate in the same
    place the report lands.
    """
    out = out_dir or os.path.join(staging_root(project), _ts_dir())
    os.makedirs(out, exist_ok=True)

    manifest = {
        "live_skill_path": live_skill_path,
        "live_memory_path": live_memory_path,
        "has_skill": proposed_skill is not None,
        "has_memory": proposed_memory is not None,
        "accepted": report.accepted,
    }
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

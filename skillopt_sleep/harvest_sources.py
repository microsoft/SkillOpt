"""Source selection for SkillOpt-Sleep transcript harvesting."""
from __future__ import annotations

from typing import Optional

from skillopt_sleep.harvest import harvest
from skillopt_sleep.harvest_codex import harvest_codex
from skillopt_sleep.harvest_copilot import harvest_copilot
from skillopt_sleep.harvest_copilot_cli import harvest_copilot_cli
from skillopt_sleep.harvest_cursor import harvest_cursor
from skillopt_sleep.harvest_opencode import harvest_opencode
from skillopt_sleep.harvest_pi import harvest_pi
from skillopt_sleep.types import SessionDigest

SUPPORTED_SOURCES = (
    "claude",
    "codex",
    "copilot",
    "copilot_cli",
    "cursor",
    "pi",
    "opencode",
    "auto",
)


def normalize_transcript_source(value: object) -> str:
    """Canonical name of a configured ``transcript_source``.

    Unset or empty means ``claude``; otherwise case and surrounding
    whitespace are ignored. The harvester, its per-source cursor and the
    tasks document must all use this one value: a cursor keyed by the raw
    spelling (``<project>::Codex``) is not the one the harvested source
    owns, and skips its older sessions (#312).
    """
    supported = ", ".join(SUPPORTED_SOURCES)
    if value is None:
        return "claude"
    if not isinstance(value, str):
        raise ValueError(
            f"transcript_source must be a string, not {type(value).__name__} "
            f"{value!r}. Supported sources: {supported}"
        )
    source = (value or "claude").strip().lower()
    if source not in SUPPORTED_SOURCES:
        raise ValueError(
            f"Unknown transcript source '{value}'. Supported sources: {supported}"
        )
    return source


def harvest_for_config(
    cfg,
    *,
    since_iso: Optional[str] = None,
    limit: int = 0,
    stats: Optional[dict] = None,
) -> list[SessionDigest]:
    """Harvest with the configured source.

    ``stats`` is filled in place by the Claude source (see
    `skillopt_sleep.harvest.harvest`); the other sources have no repeat
    filter to report yet and leave it untouched.
    """
    source = normalize_transcript_source(cfg.get("transcript_source", "claude"))
    scope = cfg.get("projects", "invoked")
    invoked_project = cfg.get("invoked_project", "")

    if source == "codex":
        return harvest_codex(
            cfg.codex_archived_sessions_dir,
            scope=scope,
            invoked_project=invoked_project,
            since_iso=since_iso,
            limit=limit,
        )
    if source == "copilot":
        return harvest_copilot(
            cfg.vscode_workspace_storage,
            scope=scope,
            invoked_project=invoked_project,
            since_iso=since_iso,
            limit=limit,
        )
    if source == "copilot_cli":
        return harvest_copilot_cli(
            cfg.copilot_cli_session_store,
            scope=scope,
            invoked_project=invoked_project,
            since_iso=since_iso,
            limit=limit,
        )
    if source == "cursor":
        return harvest_cursor(
            cfg.cursor_projects_dir,
            scope=scope,
            invoked_project=invoked_project,
            since_iso=since_iso,
            limit=limit,
        )
    if source == "pi":
        return harvest_pi(
            cfg.pi_sessions_dir,
            scope=scope,
            invoked_project=invoked_project,
            since_iso=since_iso,
            limit=limit,
        )
    if source == "opencode":
        return harvest_opencode(
            cfg.opencode_db_path,
            scope=scope,
            invoked_project=invoked_project,
            since_iso=since_iso,
            limit=limit,
        )
    if source == "auto":
        codex_digests = harvest_codex(
            cfg.codex_archived_sessions_dir,
            scope=scope,
            invoked_project=invoked_project,
            since_iso=since_iso,
            limit=limit,
        )
        if codex_digests:
            return codex_digests

    return harvest(
        cfg.transcripts_dir,
        scope=scope,
        invoked_project=invoked_project,
        since_iso=since_iso,
        limit=limit,
        stats=stats,
    )

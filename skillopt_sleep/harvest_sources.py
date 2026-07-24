"""Source selection for SkillOpt-Sleep transcript harvesting."""
from __future__ import annotations

import logging
import os
from typing import Optional

from skillopt_sleep.harvest import harvest
from skillopt_sleep.harvest_codex import harvest_codex
from skillopt_sleep.harvest_cursor import harvest_cursor
from skillopt_sleep.types import SessionDigest

_log = logging.getLogger(__name__)


# \u2500\u2500 Egress policy (F10) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n# Digests whose project path falls under any egress_deny_dirs entry are not\n# sent to external LLM backends for mining (they still contribute to the\n# heuristic miner locally).  Configure via config[\"egress_deny_dirs\"] = [...].\n\ndef _egress_allowed(digest: SessionDigest, cfg) -> bool:\n    \"\"\"Return False if this digest's project is under a deny-dir.\"\"\"\n    deny_dirs = cfg.get(\"egress_deny_dirs\") or []\n    if not deny_dirs:\n        return True\n    project = os.path.abspath(digest.project or \"\")\n    for deny in deny_dirs:\n        deny_abs = os.path.abspath(deny)\n        if project == deny_abs or project.startswith(deny_abs + os.sep):\n            _log.info(\n                \"egress blocked for session from %s (matches egress_deny_dirs)\", project\n            )\n            return False\n    return True\n\n\ndef _filter_egress(digests: list[SessionDigest], cfg) -> list[SessionDigest]:\n    \"\"\"Remove digests that are not allowed to be sent to external backends.\"\"\"\n    return [d for d in digests if _egress_allowed(d, cfg)]


def harvest_for_config(cfg, *, since_iso: Optional[str] = None, limit: int = 0) -> list[SessionDigest]:
    source = cfg.get("transcript_source", "claude")
    scope = cfg.get("projects", "invoked")
    invoked_project = cfg.get("invoked_project", "")

    if source == "codex":
        raw = harvest_codex(
            cfg.codex_archived_sessions_dir,
            scope=scope,
            invoked_project=invoked_project,
            since_iso=since_iso,
            limit=limit,
        )
        return _filter_egress(raw, cfg)
    if source == "cursor":
        raw = harvest_cursor(
            cfg.cursor_projects_dir,
            scope=scope,
            invoked_project=invoked_project,
            since_iso=since_iso,
            limit=limit,
        )
        return _filter_egress(raw, cfg)
    if source == "auto":
        codex_digests = harvest_codex(
            cfg.codex_archived_sessions_dir,
            scope=scope,
            invoked_project=invoked_project,
            since_iso=since_iso,
            limit=limit,
        )
        if codex_digests:
            return _filter_egress(codex_digests, cfg)

    raw = harvest(
        cfg.transcripts_dir,
        scope=scope,
        invoked_project=invoked_project,
        since_iso=since_iso,
        limit=limit,
    )
    return _filter_egress(raw, cfg)

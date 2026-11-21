"""Resolve OpenClaw locations without assuming a specific OS user."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _configured(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value.lower() in {"", "undefined", "null"}:
        return ""
    return value


def _system_home() -> Path:
    configured = _configured("HOME")
    return Path(configured) if configured else Path.home()


def _expand_user_path(value: str | Path) -> str:
    text = str(value)
    if text == "~":
        return str(_system_home())
    if text.startswith("~/") or text.startswith("~\\"):
        return str(_system_home() / text[2:])
    return text


def _absolute_path(value: str | Path) -> Path:
    """Match OpenClaw's lexical absolute paths without resolving symlinks."""
    return Path(os.path.abspath(_expand_user_path(value)))


def resolve_openclaw_path(value: str | Path) -> Path:
    """Resolve a user path with OpenClaw's HOME fallback and no realpath."""
    return _absolute_path(value)


def openclaw_home() -> Path:
    """Return the home used by OpenClaw path defaults."""
    configured = _configured("OPENCLAW_HOME")
    if configured:
        return resolve_openclaw_path(configured)
    home = _configured("HOME")
    return resolve_openclaw_path(home) if home else resolve_openclaw_path(Path.home())


def openclaw_state_dir() -> Path:
    """Match OpenClaw's state-dir and named-profile conventions."""
    configured = _configured("OPENCLAW_STATE_DIR")
    if configured:
        return resolve_openclaw_path(configured)
    profile = _configured("OPENCLAW_PROFILE")
    valid_profile = bool(_PROFILE_NAME_RE.fullmatch(profile))
    if not valid_profile or profile.lower() == "default":
        profile = ""
    dirname = f".openclaw-{profile}" if profile else ".openclaw"
    return openclaw_home() / dirname


def openclaw_agents_dir() -> Path:
    return openclaw_state_dir() / "agents"


def openclaw_workspace_dir() -> Path:
    return openclaw_state_dir() / "workspace"


def apply_openclaw_defaults(overrides: dict[str, Any]) -> dict[str, Any]:
    """Fill blank OpenClaw paths without overriding operator choices."""
    resolved = dict(overrides)
    if not resolved.get("claude_home"):
        resolved["claude_home"] = str(openclaw_agents_dir())
    if not resolved.get("invoked_project"):
        resolved["invoked_project"] = str(openclaw_workspace_dir())
    return resolved

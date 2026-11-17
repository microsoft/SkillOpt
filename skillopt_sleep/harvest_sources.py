"""Source selection for SkillOpt-Sleep transcript harvesting."""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional

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

_CUSTOM_SOURCES: Dict[str, Callable] = {}


def register_source(name: str, fn: Callable) -> None:
    """Register a custom transcript source in-memory (#314)."""
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("Source name cannot be blank")
    _CUSTOM_SOURCES[key] = fn


def unregister_source(name: str) -> None:
    """Unregister a custom transcript source."""
    _CUSTOM_SOURCES.pop((name or "").strip().lower(), None)


def load_module_callable(spec: str) -> Any:
    """Import and return a callable or class from a 'pkg.mod:attr' spec (#314)."""
    spec = (spec or "").strip()
    if ":" not in spec:
        raise ValueError(
            f"Expected format 'module:attribute' (e.g. 'my_pkg.adapters:harvest_func'), got {spec!r}"
        )
    mod_name, attr_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name.strip())
    attr = getattr(mod, attr_name.strip(), None)
    if attr is None:
        raise AttributeError(f"Module {mod_name!r} has no attribute {attr_name!r}")
    return attr


def load_entry_point_sources() -> Dict[str, Any]:
    """Discover transcript sources registered via entry points (#314)."""
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return {}
    try:
        eps = entry_points(group="skillopt_sleep.sources")
    except TypeError:
        eps = entry_points().get("skillopt_sleep.sources", [])
    out: Dict[str, Any] = {}
    for ep in eps:
        out[ep.name.strip().lower()] = ep.load
    return out


def get_supported_sources() -> List[str]:
    """Return all known source names: built-ins, in-memory registered, and entry points."""
    names = set(SUPPORTED_SOURCES)
    names.update(_CUSTOM_SOURCES.keys())
    names.update(load_entry_point_sources().keys())
    return sorted(names)


def normalize_transcript_source(value: object) -> str:
    """Canonical name of a configured ``transcript_source``.

    Unset or empty means ``claude``; otherwise case and surrounding
    whitespace are ignored. The harvester, its per-source cursor and the
    tasks document must all use this one value: a cursor keyed by the raw
    spelling (``<project>::Codex``) is not the one the harvested source
    owns, and skips its older sessions (#312).
    """
    if value is None:
        return "claude"
    if not isinstance(value, str):
        supported = ", ".join(get_supported_sources())
        raise ValueError(
            f"transcript_source must be a string, not {type(value).__name__} "
            f"{value!r}. Supported sources: {supported}"
        )
    source = (value or "claude").strip().lower()
    known = get_supported_sources()
    if source not in known and ":" not in source:
        raise ValueError(
            f"Unknown transcript source '{value}'. Supported sources: {', '.join(known)}"
        )
    return source


def _invoke_source_fn(
    fn: Any,
    cfg: Any,
    *,
    since_iso: Optional[str] = None,
    limit: int = 0,
    stats: Optional[dict] = None,
) -> list[SessionDigest]:
    scope = cfg.get("projects", "invoked")
    invoked_project = cfg.get("invoked_project", "")
    try:
        return fn(
            cfg,
            since_iso=since_iso,
            limit=limit,
            scope=scope,
            invoked_project=invoked_project,
        )
    except TypeError:
        try:
            return fn(cfg, since_iso=since_iso, limit=limit)
        except TypeError:
            return fn(cfg)


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
    source_mod = cfg.get("source_module")
    if source_mod:
        fn = load_module_callable(source_mod)
        return _invoke_source_fn(fn, cfg, since_iso=since_iso, limit=limit, stats=stats)

    raw_source = cfg.get("transcript_source", "claude")
    if isinstance(raw_source, str) and ":" in raw_source:
        fn = load_module_callable(raw_source)
        return _invoke_source_fn(fn, cfg, since_iso=since_iso, limit=limit, stats=stats)

    source = normalize_transcript_source(raw_source)
    scope = cfg.get("projects", "invoked")
    invoked_project = cfg.get("invoked_project", "")

    if source in _CUSTOM_SOURCES:
        return _invoke_source_fn(
            _CUSTOM_SOURCES[source],
            cfg,
            since_iso=since_iso,
            limit=limit,
            stats=stats,
        )

    ep_sources = load_entry_point_sources()
    if source in ep_sources:
        fn = ep_sources[source]()
        return _invoke_source_fn(fn, cfg, since_iso=since_iso, limit=limit, stats=stats)

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

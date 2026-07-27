"""Resolution of the ``mem0_*`` trainer config keys.

The single rule this module exists to enforce: **an API key present in the
environment must never, by itself, cause data to leave the machine.** Uploading
is opt-in via ``mem0_enabled``, and a key is only consulted once that flag is
explicitly true. A ``MEM0_API_KEY`` set for some unrelated application is
therefore inert here.

Namespacing
-----------
Memories are scoped to a namespace derived from the *project root* rather than
from a config name, so two checkouts that happen to share a config label do not
read each other's memories. The project path is hashed rather than sent
verbatim: the namespace has to be stable and unique, and it does not need to be
legible to the remote service.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

# Ceiling on any single stored payload. Applied after redaction, so the cap is
# measured on exactly the text that would be transmitted.
DEFAULT_MAX_CHARS = 4000
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_RETRIEVAL_LIMIT = 5


@dataclass(frozen=True)
class Mem0Settings:
    """Fully resolved mem0 configuration for one training run."""

    enabled: bool = False
    api_key: str = ""
    namespace: str = "skillopt"
    retrieval_enabled: bool = True
    retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_chars: int = DEFAULT_MAX_CHARS
    project_root: str = ""

    @property
    def usable(self) -> bool:
        """True only when uploading was explicitly enabled *and* a key exists."""
        return bool(self.enabled and self.api_key)


def project_namespace(project_root: str, env_name: str = "") -> str:
    """A stable, project-specific namespace.

    Stable across runs of the same project (same absolute root → same digest)
    and distinct across projects, which is what keeps unrelated experiments
    from sharing a memory pool.
    """
    root = os.path.abspath(project_root or os.getcwd())
    digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:12]
    env_part = (env_name or "default").strip().replace(" ", "_")
    return f"skillopt:{env_part}:{digest}"


def _as_bool(value: object, default: bool = False) -> bool:
    """YAML may hand us a real bool; ``--cfg-options`` hands us a string."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def resolve_settings(cfg: dict | None, env: dict | None = None) -> Mem0Settings:
    """Build :class:`Mem0Settings` from a flat trainer config.

    Parameters
    ----------
    cfg : dict | None
        Flat trainer config. ``mem0_enabled`` is the master switch.
    env : dict | None
        Environment mapping, defaulting to :data:`os.environ`. Only consulted
        for the API key, and only when ``mem0_enabled`` is true.
    """
    cfg = cfg or {}
    env = os.environ if env is None else env

    enabled = _as_bool(cfg.get("mem0_enabled"), False)
    if not enabled:
        # Return the inert default rather than reading the key at all, so a
        # disabled run cannot even accidentally hold a credential.
        return Mem0Settings(enabled=False)

    api_key = str(cfg.get("mem0_api_key") or env.get("MEM0_API_KEY", "") or "")

    project_root = os.path.abspath(str(cfg.get("out_root") or os.getcwd()))
    namespace = str(cfg.get("mem0_namespace") or "").strip()
    if not namespace:
        namespace = project_namespace(project_root, str(cfg.get("env") or ""))

    return Mem0Settings(
        enabled=True,
        api_key=api_key,
        namespace=namespace,
        retrieval_enabled=_as_bool(cfg.get("mem0_retrieval_enabled"), True),
        retrieval_limit=_as_int(cfg.get("mem0_retrieval_limit"), DEFAULT_RETRIEVAL_LIMIT),
        timeout_seconds=_as_float(cfg.get("mem0_timeout_seconds"), DEFAULT_TIMEOUT_SECONDS),
        max_chars=_as_int(cfg.get("mem0_max_chars"), DEFAULT_MAX_CHARS),
        project_root=project_root,
    )

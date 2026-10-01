"""Resolution of the ``mem0_*`` trainer config keys.

The single rule this module exists to enforce: **an API key present in the
environment must never, by itself, cause data to leave the machine.** Uploading
is opt-in via ``mem0_enabled``, and a key is only consulted once that flag is
explicitly true. A ``MEM0_API_KEY`` set for some unrelated application is
therefore inert here.

Namespacing
-----------
Memories are scoped to a namespace derived from a *stable project identity*, so
that a later run of the same project reads what earlier runs wrote. That
identity must not come from ``out_root``: the train/eval CLIs default it to
``outputs/skillopt_<env>_<model>_<timestamp>``, so deriving from it would mint a
fresh namespace on every run and cross-run retrieval would never return
anything. Identity is resolved as: an explicit ``mem0_namespace``, else the
enclosing git repository root, else the current working directory.

The path is hashed rather than sent verbatim: the namespace has to be stable and
unique, and it does not need to be legible to the remote service.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
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


def git_repo_root(start: str | None = None) -> str:
    """Absolute path of the enclosing git work tree, or ``""`` if there is none.

    Preferred identity source: it is the same for every run of a checkout no
    matter which output directory a run happens to write to, and no matter which
    subdirectory the command was invoked from.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if out.returncode != 0:
        return ""
    return os.path.abspath(out.stdout.strip()) if out.stdout.strip() else ""


def project_identity(start: str | None = None) -> str:
    """The stable path this project is identified by.

    Never derived from ``out_root`` — see the module docstring for why.
    """
    return git_repo_root(start) or os.path.abspath(start or os.getcwd())


def project_namespace(project_root: str, env_name: str = "") -> str:
    """Hash *project_root* into a namespace.

    Stable across runs of the same project (same absolute path → same digest)
    and distinct across projects, which is what keeps unrelated experiments from
    sharing a memory pool. Callers should pass :func:`project_identity`, not a
    run output directory.
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

    # Identity for namespacing: deliberately NOT out_root, which is per-run.
    identity = project_identity()
    namespace = str(cfg.get("mem0_namespace") or "").strip()
    if not namespace:
        namespace = project_namespace(identity, str(cfg.get("env") or ""))

    # out_root still anchors *path redaction* — collapsing the run directory out
    # of stored text is exactly what it is right for.
    project_root = os.path.abspath(str(cfg.get("out_root") or identity))

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

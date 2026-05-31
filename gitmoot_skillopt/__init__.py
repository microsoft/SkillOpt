"""Gitmoot-specific SkillOpt integration package."""

from __future__ import annotations

try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover - Python 3.10+ always has it.
    version = None  # type: ignore[assignment]


def package_version() -> str:
    """Return the installed package version when available."""
    if version is None:
        return "0.0.0"
    try:
        return version("gitmoot-skillopt")
    except Exception:  # noqa: BLE001 - metadata is optional in source checkouts.
        return "0.0.0"


__all__ = ["package_version"]

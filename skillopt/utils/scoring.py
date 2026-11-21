"""Scoring and hashing utilities."""
from __future__ import annotations

import hashlib


def compute_score(results: list) -> tuple[float, float]:
    """Compute hard and soft accuracy from a list of episode results.

    Accepts both plain dicts and duck-typed result instances.
    Missing, null, or invalid metric values safely default to 0.0.
    hard may be continuous (0.0-1.0) when using smoothed reward.
    """
    if not results:
        return 0.0, 0.0

    def _extract(r: object, key: str) -> float:
        if r is None:
            return 0.0
        val = (
            getattr(r, key, None)
            if hasattr(r, key)
            else (r.get(key) if isinstance(r, dict) else None)
        )
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    hard = sum(_extract(r, "hard") for r in results) / len(results)
    soft = sum(_extract(r, "soft") for r in results) / len(results)
    return hard, soft


def skill_hash(content: str) -> str:
    """Return a short deterministic hash of skill content (for caching)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]

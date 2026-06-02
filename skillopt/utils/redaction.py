"""Secret redaction helpers for persisted SkillOpt artifacts."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"

_SECRET_KEY_RE = re.compile(
    r"(?:^|[_-])(api[_-]?key|access[_-]?token|auth[_-]?token|bearer|client[_-]?secret|credential|password|secret|token)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE_RES = [
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9][A-Za-z0-9._-]{12,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[=:]\s*[^\s,;]{8,}"),
]


def is_secret_key(key: str) -> bool:
    """Return True when a mapping key is likely to contain a credential."""
    return bool(_SECRET_KEY_RE.search(str(key)))


def redact_string(value: str) -> str:
    """Redact credential-like substrings in free-form text."""
    redacted = value
    for pattern in _SECRET_VALUE_RES:
        redacted = pattern.sub(REDACTED, redacted)
    return redacted


def redact_secrets(value: Any) -> Any:
    """Recursively redact secrets from JSON-serializable artifacts.

    Mapping values are replaced wholesale when the key name is credential-like.
    Free-form strings are scanned for common token patterns. Other scalar values
    are preserved.
    """
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if is_secret_key(str(key)) and item not in (None, "") else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, str):
        return redact_string(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_secrets(item) for item in value]
    return value

"""Redaction applied to every payload before it leaves the process.

Mem0 is a third-party hosted service, so anything the trainer stores can leave
the machine. This module is the single choke point: :func:`redact_for_upload`
is called on every string in every payload built by
:mod:`skillopt.memory.mem0_backend`, so there is exactly one place to audit.

What gets removed
-----------------
* **Credentials** — vendor API keys, bearer/basic tokens, JWTs, private keys,
  and ``key = value`` style assignments for api-key/token/password/secret.
* **Filesystem identity** — the project root, the user's home directory, and
  ``/home/<user>`` / ``/Users/<user>`` / ``C:\\Users\\<user>`` prefixes, all of
  which carry the operating-system username.

What deliberately stays
-----------------------
Relative paths and file *names* survive: a skill that says "edit
``configs/train.yaml``" is still useful after redaction, and the portion
removed is the machine-specific prefix rather than the structure. Redaction
that destroyed the text would defeat the point of storing it.

The secret patterns intentionally mirror
``skillopt_sleep/staging.py::_SECRET_PATTERNS``. They are duplicated rather
than imported because ``skillopt_sleep`` is decoupled from the research
package by design (``pyproject.toml``: "the open-source Sleep tool (decoupled,
zero research dep)"); importing across that boundary to save a few lines would
couple two packages the project keeps apart on purpose.
"""
from __future__ import annotations

import os
import re
from typing import Any

# Credential patterns — mirrored from skillopt_sleep/staging.py (see module
# docstring for why this is a copy and not an import).
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_-]{10,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"\bm0-[A-Za-z0-9_-]{16,}\b"), "[REDACTED_MEM0_KEY]"),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
        "[REDACTED_JWT]",
    ),
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

# Home-directory prefixes carry the OS username, which is personally
# identifying. Replaced with "~" so the trailing structure stays readable.
_HOME_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"/home/[^/\s\"']+"), "~"),
    (re.compile(r"/Users/[^/\s\"']+"), "~"),
    (re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\\s\"']+"), "~"),
)

_PROJECT_PLACEHOLDER = "<project>"


def redact_secrets(value: Any) -> Any:
    """Scrub credential-looking substrings from every string leaf of *value*.

    Strings are rewritten; lists and dicts are walked recursively; other
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


def redact_paths(value: Any, project_root: str = "") -> Any:
    """Replace machine-identifying path prefixes in every string leaf.

    *project_root*, when given, is collapsed to ``<project>`` first so that a
    path inside the run directory does not also match a home-directory rule.
    """
    if isinstance(value, str):
        out = value
        root = (project_root or "").rstrip("/\\")
        if root:
            out = out.replace(root, _PROJECT_PLACEHOLDER)
        for pattern, replacement in _HOME_PATTERNS:
            out = pattern.sub(replacement, out)
        return out
    if isinstance(value, list):
        return [redact_paths(v, project_root) for v in value]
    if isinstance(value, dict):
        return {k: redact_paths(v, project_root) for k, v in value.items()}
    return value


def redact_for_upload(value: Any, project_root: str = "") -> Any:
    """Full outbound redaction: credentials first, then path identity.

    This is the only function callers should need. Order matters — a secret
    embedded in a path (``/home/alice/.config/sk-abc123``) must be scrubbed as
    a credential before the home prefix is collapsed, or the key would survive
    inside the shortened string.
    """
    return redact_paths(redact_secrets(value), project_root)


def default_project_root(cfg: dict | None = None) -> str:
    """Best-effort absolute project root, used to anchor path redaction."""
    if cfg:
        root = cfg.get("out_root")
        if root:
            return os.path.abspath(str(root))
    return os.path.abspath(os.getcwd())

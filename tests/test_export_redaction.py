"""Boundary tests for structured-output redaction (mapping-key aware).

The maintainer flagged that ``_redact_deep`` recurses only into values and
loses the mapping-key context, so ``{"api_key": "plain-secret"}`` survives.
The structured export boundaries now use the mapping-key-aware
``redact_secrets`` instead. These tests pin that behavior.
"""

from __future__ import annotations

from skillopt_sleep.__main__ import _redact_deep
from skillopt_sleep.staging import redact_secrets


def test_redact_secrets_is_mapping_key_aware():
    """redact_secrets redacts by mapping key (api_key/token), not just content."""
    data = {"api_key": "top-secret", "nested": {"token": "deep-secret"}}
    out = redact_secrets(data)
    assert "top-secret" not in str(out)
    assert "deep-secret" not in str(out)
    assert not _contains(out, "top-secret")
    assert not _contains(out, "deep-secret")


def test_redact_deep_is_mapping_key_aware():
    """_redact_deep now delegates to the key-aware redact_secrets, so a secret
    value under a secret-named key is redacted (the old value-losing bug)."""
    data = {"api_key": "top-secret", "content": "keep me"}
    out = _redact_deep(data)
    assert "top-secret" not in str(out), "secret still leaked"
    assert "keep me" in str(out), "diagnostic content lost"


def test_report_md_is_redacted_as_string():
    """Markdown output is scrubbed of secret-looking assignments/tokens."""
    md = (
        "reason: the API key is sk-abcdefghijklmnop123456 and "
        "a secret assignment api_key=plain-secret-value appears"
    )
    out = redact_secrets(md)
    assert "sk-abcdefghijklmnop123456" not in out
    assert "plain-secret-value" not in out


def _contains(obj, needle: str) -> bool:
    if isinstance(obj, str):
        return needle in obj
    if isinstance(obj, dict):
        return any(_contains(v, needle) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains(i, needle) for i in obj)
    return False

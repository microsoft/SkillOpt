"""Tests for persisted artifact redaction."""
from __future__ import annotations

import json

from skillopt.engine.trainer import _redact_cfg, _save_history, _save_runtime_state
from skillopt.utils.redaction import REDACTED, redact_secrets


def test_redact_secrets_recurses_keys_and_strings() -> None:
    payload = {
        "api_key": "sk-1234567890abcdef",
        "nested": {
            "client_secret": "very-secret-value",
            "message": "Authorization: Bearer abcdefghijklmnop",
        },
        "items": [{"access_token": "ghp_abcdefghijklmnopqrstuvwxyz"}],
    }

    redacted = redact_secrets(payload)

    assert redacted["api_key"] == REDACTED
    assert redacted["nested"]["client_secret"] == REDACTED
    assert REDACTED in redacted["nested"]["message"]
    assert redacted["items"][0]["access_token"] == REDACTED


def test_trainer_config_history_and_runtime_state_are_redacted(tmp_path) -> None:
    cfg = {
        "optimizer": "openai",
        "openai_api_key": "sk-1234567890abcdef",
        "nested": {"password": "do-not-store"},
    }
    assert _redact_cfg(cfg) == {
        "optimizer": "openai",
        "openai_api_key": REDACTED,
        "nested": {"password": REDACTED},
    }

    _save_history(str(tmp_path), [{"note": "Bearer abcdefghijklmnop", "token": "abc123456789"}])
    _save_runtime_state(str(tmp_path), {"api_key": "sk-1234567890abcdef"})

    history = json.loads((tmp_path / "history.json").read_text())
    state = json.loads((tmp_path / "runtime_state.json").read_text())
    assert history == [{"note": REDACTED, "token": REDACTED}]
    assert state == {"api_key": REDACTED}

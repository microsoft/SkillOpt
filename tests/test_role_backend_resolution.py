"""``--backend`` must survive the role backends the base config sets."""

from __future__ import annotations

import pytest

from scripts.eval_only import _set_role_if_default
from skillopt.engine.trainer import _resolve_role_backends

# What configs/_base_/default.yaml ships.
_BASE_CONFIG = ("openai_chat", "openai_chat")


@pytest.mark.parametrize("current", [None, "", "openai_chat"])
def test_eval_only_backend_label_replaces_inherited_role_defaults(current) -> None:
    cfg = {"target_backend": current}
    _set_role_if_default(
        cfg,
        "target_backend",
        "copilot_exec",
        explicitly_overridden=False,
    )
    assert cfg["target_backend"] == "copilot_exec"


def test_eval_only_backend_label_preserves_custom_yaml_role() -> None:
    cfg = {"optimizer_backend": "minimax_chat"}
    _set_role_if_default(
        cfg,
        "optimizer_backend",
        "copilot_chat",
        explicitly_overridden=False,
    )
    assert cfg["optimizer_backend"] == "minimax_chat"


def test_eval_only_backend_label_preserves_explicit_cli_role_override() -> None:
    cfg = {"optimizer_backend": "openai_chat"}
    _set_role_if_default(
        cfg,
        "optimizer_backend",
        "copilot_chat",
        explicitly_overridden=True,
    )
    assert cfg["optimizer_backend"] == "openai_chat"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("cursor", ("openai_chat", "cursor_exec")),
        ("cursor_exec", ("openai_chat", "cursor_exec")),
        ("claude", ("claude_chat", "claude_chat")),
        ("claude_chat", ("claude_chat", "claude_chat")),
        ("claude_code_exec", ("claude_code_exec", "claude_code_exec")),
        ("codex", ("codex_exec", "codex_exec")),
        ("codex_exec", ("codex_exec", "codex_exec")),
        ("qwen", ("openai_chat", "qwen_chat")),
        ("qwen_chat", ("openai_chat", "qwen_chat")),
    ],
)
def test_backend_flag_wins_over_base_config_defaults(backend, expected) -> None:
    # Regression: the base config sets both roles to openai_chat, which used to
    # skip resolution entirely and silently run --backend <x> on openai_chat.
    assert _resolve_role_backends(backend, *_BASE_CONFIG) == expected


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("anthropic", ("claude_chat", "claude_chat")),
        ("openai", ("codex_exec", "codex_exec")),
        ("cursor_agent", ("openai_chat", "cursor_exec")),
        ("copilot_cli", ("copilot_chat", "copilot_chat")),
        ("github_copilot", ("copilot_chat", "copilot_chat")),
        ("minimax", ("openai_chat", "minimax_chat")),
        ("compat", ("openai_compatible", "openai_compatible")),
        ("openai-compatible", ("openai_compatible", "openai_compatible")),
    ],
)
def test_backend_aliases_are_normalized_before_role_resolution(alias, expected) -> None:
    assert _resolve_role_backends(alias, *_BASE_CONFIG) == expected


def test_azure_openai_stays_on_openai_chat() -> None:
    assert _resolve_role_backends("azure_openai", *_BASE_CONFIG) == _BASE_CONFIG


def test_unset_roles_are_resolved() -> None:
    assert _resolve_role_backends("cursor", "", "") == ("openai_chat", "cursor_exec")
    assert _resolve_role_backends("cursor", None, None) == ("openai_chat", "cursor_exec")


def test_explicit_non_default_roles_are_preserved() -> None:
    # An operator who names a role backend outranks the high-level label.
    assert _resolve_role_backends("cursor", "qwen_chat", "minimax_chat") == (
        "qwen_chat",
        "minimax_chat",
    )


@pytest.mark.parametrize(
    ("backend", "expected_target"),
    [
        ("claude", "claude_chat"),
        ("codex", "codex_exec"),
        ("claude_code_exec", "claude_code_exec"),
        ("cursor", "cursor_exec"),
        ("copilot", "copilot_chat"),
        ("copilot_exec", "copilot_exec"),
        ("qwen", "qwen_chat"),
    ],
)
def test_explicit_optimizer_is_preserved_while_default_target_is_resolved(
    backend, expected_target
) -> None:
    assert _resolve_role_backends(backend, "minimax_chat", "openai_chat") == (
        "minimax_chat",
        expected_target,
    )


@pytest.mark.parametrize(
    ("backend", "expected_optimizer"),
    [
        ("claude", "claude_chat"),
        ("codex", "codex_exec"),
        ("claude_code_exec", "claude_code_exec"),
        ("cursor", "openai_chat"),
        ("copilot", "copilot_chat"),
        ("copilot_exec", "openai_chat"),
        ("qwen", "openai_chat"),
    ],
)
def test_default_optimizer_is_resolved_while_explicit_target_is_preserved(
    backend, expected_optimizer
) -> None:
    assert _resolve_role_backends(backend, "openai_chat", "minimax_chat") == (
        expected_optimizer,
        "minimax_chat",
    )


def test_explicit_target_is_preserved_when_optimizer_is_default() -> None:
    assert _resolve_role_backends("cursor", "openai_chat", "minimax_chat") == (
        "openai_chat",
        "minimax_chat",
    )


def test_copilot_maps_both_roles_to_the_cli_authenticated_backend() -> None:
    # No separate provider API key is needed because the CLI carries sign-in.
    assert _resolve_role_backends("copilot", *_BASE_CONFIG) == ("copilot_chat", "copilot_chat")
    assert _resolve_role_backends("copilot_chat", *_BASE_CONFIG) == (
        "copilot_chat",
        "copilot_chat",
    )


def test_copilot_exec_keeps_a_chat_optimizer() -> None:
    assert _resolve_role_backends("copilot_exec", *_BASE_CONFIG) == (
        "openai_chat",
        "copilot_exec",
    )


def test_claude_maps_both_roles() -> None:
    assert _resolve_role_backends("claude", None, None) == ("claude_chat", "claude_chat")
    # And -- the regression -- when the base config pins both roles to the
    # truthy openai_chat default, --backend claude must still win.
    assert _resolve_role_backends("claude", *_BASE_CONFIG) == ("claude_chat", "claude_chat")
    assert _resolve_role_backends("claude_chat", *_BASE_CONFIG) == ("claude_chat", "claude_chat")

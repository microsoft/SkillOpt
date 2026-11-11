"""Claude backends default to the Claude CLI's `sonnet` alias, from one source."""
from __future__ import annotations

import importlib.util

import pytest

from skillopt.model import claude_backend, claude_code_backend
from skillopt.model.common import default_model_for_backend


@pytest.mark.parametrize("backend", ["claude", "claude_chat", "claude_code_exec", "anthropic"])
def test_claude_backends_default_to_the_cli_sonnet_alias(backend: str) -> None:
    # `claude --help`: "Provide an alias for the latest model (e.g. 'fable', 'opus',
    # or 'sonnet') or a model's full name." A dated ID such as claude-sonnet-4-6
    # keeps every unconfigured run on that model after newer ones ship.
    assert default_model_for_backend(backend) == "sonnet"


def _fresh_copy(module):
    spec = importlib.util.spec_from_file_location(f"_fresh_{module.__name__}", module.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    return fresh


@pytest.mark.parametrize(
    ("module", "backend"),
    [(claude_backend, "claude"), (claude_code_backend, "claude_code_exec")],
)
def test_module_deployments_fall_back_to_the_backend_default(monkeypatch, module, backend) -> None:
    monkeypatch.delenv("OPTIMIZER_DEPLOYMENT", raising=False)
    monkeypatch.delenv("TARGET_DEPLOYMENT", raising=False)

    fresh = _fresh_copy(module)

    assert fresh.OPTIMIZER_DEPLOYMENT == default_model_for_backend(backend)
    assert fresh.TARGET_DEPLOYMENT == default_model_for_backend(backend)

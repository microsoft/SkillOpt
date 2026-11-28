"""config.json and summary.json must not hold provider API keys.

_redact_cfg only knew azure_api_key, api_key and openai_api_key, but the config
stores credentials as azure_openai_api_key, qwen_chat_api_key, minimax_api_key,
openai_compatible_api_key and their optimizer_/target_ variants, so those were
written to the run folder in plaintext.
"""

from __future__ import annotations

import json
import os
import sys
from unittest import mock

import pytest

import scripts.train as train_script
import skillopt.engine.trainer as trainer_mod
from skillopt.envs.base import EnvAdapter
from skillopt.model import backend_config

SECRETS = {
    "azure_openai_api_key": "AZ-SECRET-1111111111111111",
    "optimizer_azure_openai_api_key": "AZ-OPT-SECRET-2222222222",
    "target_azure_openai_api_key": "AZ-TGT-SECRET-3333333333",
    "qwen_chat_api_key": "QWEN-SECRET-444444444444",
    "target_qwen_chat_api_key": "QWEN-TGT-SECRET-55555555",
    "minimax_api_key": "MINIMAX-SECRET-66666666",
    "openai_compatible_api_key": "OAI-COMPAT-SECRET-777777",
    "optimizer_openai_compatible_api_key": "OAI-OPT-SECRET-88888888",
}


@pytest.fixture
def backends_not_configured(monkeypatch):
    # The trainer hands the keys to each backend's configure function, which
    # keeps them in module state and os.environ. Keep the fake keys out of both,
    # and put back the backends the trainer selects.
    for name in (
        "configure_azure_openai",
        "configure_qwen_chat",
        "configure_minimax_chat",
        "configure_openai_compatible",
    ):
        monkeypatch.setattr(trainer_mod, name, lambda *args, **kwargs: None)
    optimizer_backend = backend_config.get_optimizer_backend()
    target_backend = backend_config.get_target_backend()
    with mock.patch.dict(os.environ):
        yield
        backend_config.set_optimizer_backend(optimizer_backend)
        backend_config.set_target_backend(target_backend)


def _train(tmp_path, extra_cfg: dict) -> str:
    skill = tmp_path / "skill.md"
    skill.write_text("# Skill\n", encoding="utf-8")
    out_root = tmp_path / "out"
    cfg = {
        "out_root": str(out_root),
        "backend": "openai_chat",
        "optimizer_backend": "openai_chat",
        "target_backend": "openai_chat",
        "skill_init": str(skill),
        "num_epochs": 0,
        "train_size": 1,
        "batch_size": 1,
        "accumulation": 1,
        "merge_batch_size": 2,
        "edit_budget": 2,
        "seed": 42,
        "sel_env_num": 1,
        "eval_test": False,
        **extra_cfg,
    }
    adapter = mock.create_autospec(EnvAdapter, instance=True)
    adapter.requires_ray.return_value = False
    adapter.get_dataloader.return_value = None
    adapter.get_task_types.return_value = []
    adapter.build_eval_env.return_value = ["1"]
    adapter.rollout.return_value = [{"id": "1", "hard": 1, "soft": 1.0, "task_type": "qa"}]
    trainer_mod.ReflACTTrainer(cfg, adapter).train()
    return str(out_root)


def _files_containing(out_root: str, secret: str) -> list[str]:
    found = []
    for dirpath, _dirs, files in os.walk(out_root):
        for name in files:
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8", errors="replace") as f:
                if secret in f.read():
                    found.append(os.path.relpath(path, out_root))
    return found


def test_run_folder_holds_no_provider_api_key(tmp_path, backends_not_configured):
    out_root = _train(tmp_path, SECRETS)

    for key, secret in SECRETS.items():
        assert _files_containing(out_root, secret) == [], key
    with open(os.path.join(out_root, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    assert config["azure_openai_api_key"] == "AZ-S...1111"


def test_legacy_azure_api_key_alias_is_redacted():
    secret = "LEGACY-ONLY-SECRET-99999999"
    argv = ["skillopt-train", "--config", "configs/searchqa/default.yaml", "--azure_api_key", secret]
    with mock.patch.object(sys, "argv", argv), pytest.warns(DeprecationWarning, match="--azure_api_key"):
        cfg = train_script.load_config(train_script.parse_args())
    # load_config copies the legacy key into azure_openai_api_key.
    assert cfg["azure_openai_api_key"] == secret

    redacted = trainer_mod._redact_cfg(cfg)

    assert secret not in json.dumps(redacted)


def test_credential_suffixes_are_masked_but_token_budgets_are_not():
    secrets = {
        "auth_token": "TOKEN-VALUE-12345678",
        "client_secret": "CLIENT-VALUE-12345678",
        "db_password": "PASSWORD-VALUE-1234",
    }
    budgets = {
        "max_completion_tokens": 4096,
        "minimax_max_tokens": 1024,
        "claude_code_exec_max_thinking_tokens": 8000,
    }

    redacted = trainer_mod._redact_cfg({**secrets, **budgets})

    for value in secrets.values():
        assert value not in json.dumps(redacted)
    for key, value in budgets.items():
        assert redacted[key] == value


def test_keys_in_a_nested_section_are_masked():
    # A section flattening keeps as-is, e.g. from --cfg-options gateway.api_key=...
    redacted = trainer_mod._redact_cfg({"gateway": {"api_key": "GATEWAY-VALUE-12345678", "timeout": 30}})

    assert "GATEWAY-VALUE-12345678" not in json.dumps(redacted)
    assert redacted["gateway"]["timeout"] == 30

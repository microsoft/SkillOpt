"""Tests for MiniMax backend configuration and CLI pipeline."""

from __future__ import annotations

import tempfile
from unittest import mock

import pytest
import yaml

import scripts.eval_only as eval_only_script
import scripts.train as train_script
from skillopt.config import flatten_config, load_config


def test_flatten_config_maps_minimax_keys() -> None:
    structured_cfg = {
        "model": {
            "backend": "minimax_chat",
            "minimax_region": "global_en",
            "minimax_base_url": "https://api.minimax.io/v1",
            "minimax_api_key": "test-key",
            "minimax_model": "MiniMax-M3",
            "minimax_temperature": 0.7,
            "minimax_timeout_seconds": 150.0,
            "minimax_max_tokens": 4096,
            "minimax_enable_thinking": True,
        },
        "train": {"num_epochs": 1},
        "env": {"name": "searchqa"},
    }

    flat = flatten_config(structured_cfg)

    assert flat["model_backend"] == "minimax_chat"
    assert flat["minimax_region"] == "global_en"
    assert flat["minimax_base_url"] == "https://api.minimax.io/v1"
    assert flat["minimax_api_key"] == "test-key"
    assert flat["minimax_model"] == "MiniMax-M3"
    assert flat["minimax_temperature"] == 0.7
    assert flat["minimax_timeout_seconds"] == 150.0
    assert flat["minimax_max_tokens"] == 4096
    assert flat["minimax_enable_thinking"] is True


def test_load_config_with_cfg_options_overrides() -> None:
    raw = {
        "model": {
            "backend": "minimax_chat",
            "minimax_model": "MiniMax-M3",
        },
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        path = f.name

    overrides = [
        "model.minimax_timeout_seconds=45.0",
        "model.minimax_temperature=0.2",
    ]
    cfg = load_config(path, overrides=overrides)
    flat = flatten_config(cfg)

    assert flat["minimax_timeout_seconds"] == 45.0
    assert flat["minimax_temperature"] == 0.2
    assert flat["minimax_model"] == "MiniMax-M3"


def test_train_script_cli_to_structured_mapping() -> None:
    cli_args = [
        "--config", "configs/base.yaml",
        "--minimax_region", "global_en",
        "--minimax_base_url", "https://api.minimax.io/v1",
        "--minimax_model", "MiniMax-M3",
        "--minimax_temperature", "0.5",
        "--minimax_timeout_seconds", "120.0",
        "--minimax_max_tokens", "2048",
    ]

    with mock.patch("sys.argv", ["train.py"] + cli_args):
        args = train_script.parse_args()

    assert args.minimax_region == "global_en"
    assert args.minimax_base_url == "https://api.minimax.io/v1"
    assert args.minimax_model == "MiniMax-M3"
    assert args.minimax_temperature == 0.5
    assert args.minimax_timeout_seconds == 120.0
    assert args.minimax_max_tokens == 2048

    for flag in [
        "minimax_region",
        "minimax_base_url",
        "minimax_api_key",
        "minimax_model",
        "minimax_temperature",
        "minimax_timeout_seconds",
        "minimax_max_tokens",
        "minimax_enable_thinking",
    ]:
        assert flag in train_script._LEGACY_TO_STRUCTURED
        assert train_script._LEGACY_TO_STRUCTURED[flag] == f"model.{flag}"


def test_eval_only_parse_args_and_map() -> None:
    raw = {
        "model": {
            "backend": "minimax_chat",
        },
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        config_path = f.name

    cli_args = [
        "--config", config_path,
        "--skill", "skills/test.md",
        "--backend", "minimax_chat",
        "--minimax_timeout_seconds", "180.0",
    ]

    with mock.patch("sys.argv", ["eval_only.py"] + cli_args):
        args = eval_only_script.parse_args()
        cfg = eval_only_script.load_config(args)

    assert args.backend == "minimax_chat"
    assert args.minimax_timeout_seconds == 180.0
    assert cfg["minimax_timeout_seconds"] == 180.0


def test_eval_only_main_forwards_minimax_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {
        "model": {
            "backend": "minimax_chat",
            "minimax_model": "MiniMax-M3",
            "minimax_timeout_seconds": 180.0,
        },
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f_yaml:
        yaml.dump(raw, f_yaml)
        config_path = f_yaml.name

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f_skill:
        f_skill.write("# Test Skill\n")
        skill_path = f_skill.name

    configured_kwargs: dict = {}

    def fake_configure(**kwargs):
        configured_kwargs.update(kwargs)

    monkeypatch.setattr(eval_only_script, "configure_minimax_chat", fake_configure)

    class EarlyExit(Exception):
        pass

    def stop_after_config(*args, **kwargs):
        raise EarlyExit()

    monkeypatch.setattr(eval_only_script, "set_reasoning_effort", stop_after_config)

    cli_args = ["eval_only.py", "--config", config_path, "--skill", skill_path]
    with mock.patch("sys.argv", cli_args):
        with pytest.raises(EarlyExit):
            eval_only_script.main()

    assert configured_kwargs.get("timeout_seconds") == 180.0


def test_trainer_initialization_configures_minimax(monkeypatch: pytest.MonkeyPatch) -> None:
    import skillopt.engine.trainer as trainer_mod
    from skillopt.envs.base import EnvAdapter

    configured_kwargs: dict = {}

    def fake_configure(**kwargs):
        configured_kwargs.update(kwargs)

    monkeypatch.setattr(trainer_mod, "configure_minimax_chat", fake_configure)

    class EarlyExit(Exception):
        pass

    def stop_after_model_config(*args, **kwargs):
        raise EarlyExit()

    monkeypatch.setattr(trainer_mod, "_configure_trace_to_optimizer_gates", stop_after_model_config)

    cfg = {
        "model_backend": "minimax_chat",
        "optimizer_backend": "minimax_chat",
        "target_backend": "minimax_chat",
        "minimax_timeout_seconds": 210.0,
        "skill_init": "skills/empty.md",
        "num_epochs": 1,
        "train_size": 1,
        "batch_size": 1,
        "accumulation": 1,
        "merge_batch_size": 2,
        "edit_budget": 2,
        "seed": 42,
        "out_root": "/tmp/out",
    }

    mock_adapter = mock.create_autospec(EnvAdapter, instance=True)
    mock_adapter.requires_ray.return_value = False
    mock_adapter.get_dataloader.return_value = None

    trainer = trainer_mod.ReflACTTrainer(cfg, mock_adapter)
    with pytest.raises(EarlyExit):
        trainer.train()

    assert configured_kwargs.get("timeout_seconds") == 210.0

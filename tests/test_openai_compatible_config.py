"""Tests for OpenAI-compatible backend configuration pipeline."""

from __future__ import annotations

import os
import tempfile
from unittest import mock

import pytest
import yaml

import scripts.eval_only as eval_only_script
import scripts.train as train_script
from skillopt.config import flatten_config, load_config


def test_flatten_config_maps_openai_compatible_keys() -> None:
    structured_cfg = {
        "model": {
            "backend": "openai_compatible",
            "openai_compatible_base_url": "https://api.deepseek.com/v1",
            "openai_compatible_api_key": "sk-deepseek-test",
            "openai_compatible_model": "deepseek-chat",
            "openai_compatible_temperature": 0.3,
            "openai_compatible_timeout_seconds": 180.0,
            "openai_compatible_max_tokens": 4096,
            "optimizer_openai_compatible_base_url": "https://api.together.xyz/v1",
            "optimizer_openai_compatible_api_key": "sk-together-test",
            "optimizer_openai_compatible_model": "together-model",
            "optimizer_openai_compatible_temperature": 0.1,
            "optimizer_openai_compatible_timeout_seconds": 240.0,
            "optimizer_openai_compatible_max_tokens": 8192,
            "target_openai_compatible_base_url": "https://api.groq.com/openai/v1",
            "target_openai_compatible_api_key": "sk-groq-test",
            "target_openai_compatible_model": "groq-model",
            "target_openai_compatible_temperature": 0.7,
            "target_openai_compatible_timeout_seconds": 90.0,
            "target_openai_compatible_max_tokens": 2048,
        },
        "train": {"num_epochs": 1},
        "env": {"name": "searchqa"},
    }

    flat = flatten_config(structured_cfg)

    assert flat["model_backend"] == "openai_compatible"
    assert flat["openai_compatible_base_url"] == "https://api.deepseek.com/v1"
    assert flat["openai_compatible_api_key"] == "sk-deepseek-test"
    assert flat["openai_compatible_model"] == "deepseek-chat"
    assert flat["openai_compatible_temperature"] == 0.3
    assert flat["openai_compatible_timeout_seconds"] == 180.0
    assert flat["openai_compatible_max_tokens"] == 4096

    assert flat["optimizer_openai_compatible_base_url"] == "https://api.together.xyz/v1"
    assert flat["optimizer_openai_compatible_api_key"] == "sk-together-test"
    assert flat["optimizer_openai_compatible_model"] == "together-model"
    assert flat["optimizer_openai_compatible_temperature"] == 0.1
    assert flat["optimizer_openai_compatible_timeout_seconds"] == 240.0
    assert flat["optimizer_openai_compatible_max_tokens"] == 8192

    assert flat["target_openai_compatible_base_url"] == "https://api.groq.com/openai/v1"
    assert flat["target_openai_compatible_api_key"] == "sk-groq-test"
    assert flat["target_openai_compatible_model"] == "groq-model"
    assert flat["target_openai_compatible_temperature"] == 0.7
    assert flat["target_openai_compatible_timeout_seconds"] == 90.0
    assert flat["target_openai_compatible_max_tokens"] == 2048


def test_load_config_with_cfg_options_overrides() -> None:
    raw = {
        "model": {
            "backend": "openai_compatible",
            "openai_compatible_base_url": "http://localhost:11434/v1",
            "openai_compatible_model": "llama3",
        },
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        path = f.name

    overrides = [
        "model.openai_compatible_base_url=http://custom-host:8000/v1",
        "model.openai_compatible_temperature=0.8",
    ]
    cfg = load_config(path, overrides=overrides)
    flat = flatten_config(cfg)

    assert flat["openai_compatible_base_url"] == "http://custom-host:8000/v1"
    assert flat["openai_compatible_temperature"] == 0.8
    assert flat["openai_compatible_model"] == "llama3"


def test_train_script_cli_to_structured_mapping() -> None:
    cli_args = [
        "--config", "configs/base.yaml",
        "--openai_compatible_base_url", "https://api.openai-compat.com/v1",
        "--openai_compatible_model", "custom-model",
        "--openai_compatible_temperature", "0.4",
        "--openai_compatible_timeout_seconds", "150",
        "--openai_compatible_max_tokens", "3000",
        "--optimizer_openai_compatible_base_url", "https://api.opt.com/v1",
        "--target_openai_compatible_base_url", "https://api.target.com/v1",
    ]

    with mock.patch("sys.argv", ["train.py"] + cli_args):
        args = train_script.parse_args()

    assert args.openai_compatible_base_url == "https://api.openai-compat.com/v1"
    assert args.openai_compatible_model == "custom-model"
    assert args.openai_compatible_temperature == 0.4
    assert args.openai_compatible_timeout_seconds == 150.0
    assert args.openai_compatible_max_tokens == 3000
    assert args.optimizer_openai_compatible_base_url == "https://api.opt.com/v1"
    assert args.target_openai_compatible_base_url == "https://api.target.com/v1"

    for flag in [
        "openai_compatible_base_url",
        "openai_compatible_api_key",
        "openai_compatible_model",
        "openai_compatible_temperature",
        "openai_compatible_timeout_seconds",
        "openai_compatible_max_tokens",
        "optimizer_openai_compatible_base_url",
        "optimizer_openai_compatible_api_key",
        "optimizer_openai_compatible_model",
        "optimizer_openai_compatible_temperature",
        "optimizer_openai_compatible_timeout_seconds",
        "optimizer_openai_compatible_max_tokens",
        "target_openai_compatible_base_url",
        "target_openai_compatible_api_key",
        "target_openai_compatible_model",
        "target_openai_compatible_temperature",
        "target_openai_compatible_timeout_seconds",
        "target_openai_compatible_max_tokens",
    ]:
        assert flag in train_script._LEGACY_TO_STRUCTURED
        assert train_script._LEGACY_TO_STRUCTURED[flag] == f"model.{flag}"


def test_eval_only_parse_args_and_map() -> None:
    raw = {
        "model": {
            "backend": "openai_compatible",
        },
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        config_path = f.name

    cli_args = [
        "--config", config_path,
        "--skill", "skills/test.md",
        "--backend", "openai_compatible",
        "--openai_compatible_base_url", "https://eval.example/v1",
        "--openai_compatible_model", "eval-model",
        "--target_openai_compatible_temperature", "0.6",
    ]

    with mock.patch("sys.argv", ["eval_only.py"] + cli_args):
        args = eval_only_script.parse_args()
        cfg = eval_only_script.load_config(args)

    assert args.backend == "openai_compatible"
    assert args.openai_compatible_base_url == "https://eval.example/v1"
    assert args.openai_compatible_model == "eval-model"
    assert args.target_openai_compatible_temperature == 0.6
    assert cfg["openai_compatible_base_url"] == "https://eval.example/v1"
    assert cfg["openai_compatible_model"] == "eval-model"
    assert cfg["target_openai_compatible_temperature"] == 0.6


def test_train_script_credential_warnings_for_openai_compatible() -> None:
    raw = {
        "model": {"backend": "openai_compatible"},
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        config_path = f.name

    cli_args = [
        "--config", config_path,
        "--openai_compatible_api_key", "sk-secret-test",
    ]

    with mock.patch("sys.argv", ["train.py"] + cli_args):
        with pytest.deprecated_call(match="OPENAI_COMPATIBLE_API_KEY"):
            train_script.load_config(train_script.parse_args())


def test_trainer_initialization_configures_openai_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    import skillopt.engine.trainer as trainer_mod
    from skillopt.envs.base import EnvAdapter

    configured_kwargs: dict = {}

    def fake_configure(**kwargs):
        configured_kwargs.update(kwargs)

    monkeypatch.setattr(trainer_mod, "configure_openai_compatible", fake_configure)

    class EarlyExit(Exception):
        pass

    def stop_after_model_config(*args, **kwargs):
        raise EarlyExit()

    monkeypatch.setattr(trainer_mod, "_configure_trace_to_optimizer_gates", stop_after_model_config)

    cfg = {
        "model_backend": "openai_compatible",
        "optimizer_backend": "openai_compatible",
        "target_backend": "openai_compatible",
        "optimizer_model": "opt-model",
        "target_model": "target-model",
        "openai_compatible_base_url": "https://api.test.com/v1",
        "openai_compatible_api_key": "test-key",
        "openai_compatible_model": "test-model",
        "openai_compatible_temperature": 0.5,
        "openai_compatible_timeout_seconds": 120.0,
        "openai_compatible_max_tokens": 4096,
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

    assert configured_kwargs["base_url"] == "https://api.test.com/v1"
    assert configured_kwargs["api_key"] == "test-key"
    assert configured_kwargs["model"] == "test-model"
    assert configured_kwargs["temperature"] == 0.5
    assert configured_kwargs["timeout_seconds"] == 120.0
    assert configured_kwargs["max_tokens"] == 4096


def test_train_script_backend_choices_accepts_openai_compatible() -> None:
    raw = {
        "model": {"backend": "azure_openai", "optimizer": "gpt-5.5", "target": "gpt-5.5"},
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        config_path = f.name

    cli_args = [
        "--config", config_path,
        "--backend", "openai_compatible",
    ]
    with mock.patch("sys.argv", ["train.py"] + cli_args):
        args = train_script.parse_args()
        flat = train_script.load_config(args)

    assert args.backend == "openai_compatible"
    assert flat["optimizer_backend"] == "openai_compatible"
    assert flat["target_backend"] == "openai_compatible"
    assert flat["optimizer_model"] == "gpt-4o-mini"
    assert flat["target_model"] == "gpt-4o-mini"


def test_train_script_openai_compatible_model_precedence() -> None:
    raw = {
        "model": {"backend": "azure_openai", "optimizer": "gpt-5.5", "target": "gpt-5.5"},
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        config_path = f.name

    # 1. Shared model override
    with mock.patch("sys.argv", [
        "train.py",
        "--config", config_path,
        "--backend", "openai_compatible",
        "--openai_compatible_model", "deepseek-chat",
    ]):
        flat = train_script.load_config(train_script.parse_args())
        assert flat["optimizer_model"] == "deepseek-chat"
        assert flat["target_model"] == "deepseek-chat"

    # 2. Per-role compatible model overrides
    with mock.patch("sys.argv", [
        "train.py",
        "--config", config_path,
        "--backend", "openai_compatible",
        "--openai_compatible_model", "fallback-shared",
        "--optimizer_openai_compatible_model", "deepseek-coder",
        "--target_openai_compatible_model", "deepseek-v3",
    ]):
        flat = train_script.load_config(train_script.parse_args())
        assert flat["optimizer_model"] == "deepseek-coder"
        assert flat["target_model"] == "deepseek-v3"

    # 3. Explicit per-role model overrides take highest precedence
    with mock.patch("sys.argv", [
        "train.py",
        "--config", config_path,
        "--backend", "openai_compatible",
        "--openai_compatible_model", "fallback-shared",
        "--optimizer_model", "explicit-opt-model",
        "--target_model", "explicit-tgt-model",
    ]):
        flat = train_script.load_config(train_script.parse_args())
        assert flat["optimizer_model"] == "explicit-opt-model"
        assert flat["target_model"] == "explicit-tgt-model"


@pytest.mark.parametrize("backend_flag", ["qwen", "qwen_chat"])
def test_eval_only_qwen_role_and_model_resolution(backend_flag: str) -> None:
    raw = {
        "model": {"backend": "azure_openai", "optimizer": "gpt-5.5", "target": "gpt-5.5"},
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        config_path = f.name

    cli_args = [
        "--config", config_path,
        "--skill", "skills/test.md",
        "--backend", backend_flag,
    ]
    with mock.patch("sys.argv", ["eval_only.py"] + cli_args):
        args = eval_only_script.parse_args()
        cfg = eval_only_script.load_config(args)

    assert cfg["optimizer_backend"] == "openai_chat"
    assert cfg["target_backend"] == "qwen_chat"
    assert cfg["optimizer_model"] == "gpt-5.5"  # openai_chat keeps base sentinel or default
    assert cfg["target_model"] == "Qwen/Qwen3.5-4B"  # normalized from gpt-5.5 sentinel


def test_eval_only_openai_compatible_model_precedence() -> None:
    raw = {
        "model": {"backend": "azure_openai", "optimizer": "gpt-5.5", "target": "gpt-5.5"},
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        config_path = f.name

    # 1. Default fallback
    with mock.patch("sys.argv", [
        "eval_only.py",
        "--config", config_path,
        "--skill", "skills/test.md",
        "--backend", "openai_compatible",
    ]):
        cfg = eval_only_script.load_config(eval_only_script.parse_args())
        assert cfg["optimizer_model"] == "gpt-4o-mini"
        assert cfg["target_model"] == "gpt-4o-mini"

    # 2. Shared model override
    with mock.patch("sys.argv", [
        "eval_only.py",
        "--config", config_path,
        "--skill", "skills/test.md",
        "--backend", "openai_compatible",
        "--openai_compatible_model", "deepseek-chat",
    ]):
        cfg = eval_only_script.load_config(eval_only_script.parse_args())
        assert cfg["optimizer_model"] == "deepseek-chat"
        assert cfg["target_model"] == "deepseek-chat"

    # 3. Per-role compatible model overrides
    with mock.patch("sys.argv", [
        "eval_only.py",
        "--config", config_path,
        "--skill", "skills/test.md",
        "--backend", "openai_compatible",
        "--optimizer_openai_compatible_model", "opt-compat-model",
        "--target_openai_compatible_model", "tgt-compat-model",
    ]):
        cfg = eval_only_script.load_config(eval_only_script.parse_args())
        assert cfg["optimizer_model"] == "opt-compat-model"
        assert cfg["target_model"] == "tgt-compat-model"

    # 4. Explicit per-role overrides take precedence
    with mock.patch("sys.argv", [
        "eval_only.py",
        "--config", config_path,
        "--skill", "skills/test.md",
        "--backend", "openai_compatible",
        "--openai_compatible_model", "fallback-model",
        "--optimizer_model", "explicit-opt",
        "--target_model", "explicit-tgt",
    ]):
        cfg = eval_only_script.load_config(eval_only_script.parse_args())
        assert cfg["optimizer_model"] == "explicit-opt"
        assert cfg["target_model"] == "explicit-tgt"


def test_eval_only_forwards_optimizer_qwen_chat_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {
        "model": {"backend": "qwen_chat"},
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        config_path = f.name

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("# Dummy Skill\n")
        skill_path = f.name

    cli_args = [
        "--config", config_path,
        "--skill", skill_path,
        "--optimizer_qwen_chat_base_url", "http://opt-qwen:8000/v1",
        "--optimizer_qwen_chat_api_key", "opt-qwen-key",
        "--optimizer_qwen_chat_temperature", "0.2",
        "--optimizer_qwen_chat_timeout_seconds", "90",
        "--optimizer_qwen_chat_max_tokens", "2048",
        "--optimizer_qwen_chat_enable_thinking", "true",
        "--optimizer_qwen_chat_thinking_mode", "deep",
        "--target_qwen_chat_base_url", "http://tgt-qwen:8000/v1",
    ]

    qwen_kwargs: dict = {}

    def fake_configure_qwen(**kwargs):
        qwen_kwargs.update(kwargs)

    monkeypatch.setattr(eval_only_script, "configure_qwen_chat", fake_configure_qwen)

    class StopExecution(Exception):
        pass

    def stop_at_adapter(*args, **kwargs):
        raise StopExecution()

    monkeypatch.setattr(eval_only_script, "get_adapter", stop_at_adapter)

    with mock.patch("sys.argv", ["eval_only.py"] + cli_args):
        with pytest.raises(StopExecution):
            eval_only_script.main()

    assert qwen_kwargs["optimizer_base_url"] == "http://opt-qwen:8000/v1"
    assert qwen_kwargs["optimizer_api_key"] == "opt-qwen-key"
    assert qwen_kwargs["optimizer_temperature"] == 0.2
    assert qwen_kwargs["optimizer_timeout_seconds"] == 90.0
    assert qwen_kwargs["optimizer_max_tokens"] == 2048
    assert qwen_kwargs["optimizer_enable_thinking"] is True
    assert qwen_kwargs["optimizer_thinking_mode"] == "deep"
    assert qwen_kwargs["target_base_url"] == "http://tgt-qwen:8000/v1"



def test_trainer_preserves_explicit_role_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import skillopt.engine.trainer as trainer_mod
    import skillopt.model.openai_compatible_backend as openai_compat
    from skillopt.envs.base import EnvAdapter

    class EarlyExit(Exception):
        pass

    def stop_after_model_config(*args, **kwargs):
        raise EarlyExit()

    monkeypatch.setattr(trainer_mod, "_configure_trace_to_optimizer_gates", stop_after_model_config)

    cfg = {
        "model_backend": "openai_compatible",
        "optimizer_backend": "openai_compatible",
        "target_backend": "openai_compatible",
        "optimizer_model": "explicit-optimizer-model",
        "target_model": "explicit-target-model",
        "openai_compatible_base_url": "https://api.test.com/v1",
        "openai_compatible_model": "fallback-shared",
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

    assert openai_compat.OPTIMIZER_CONFIG.deployment == "explicit-optimizer-model"
    assert openai_compat.TARGET_CONFIG.deployment == "explicit-target-model"


def test_eval_only_preserves_explicit_role_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import skillopt.model.openai_compatible_backend as openai_compat

    cfg = {
        "model_backend": "openai_compatible",
        "optimizer_backend": "openai_compatible",
        "target_backend": "openai_compatible",
        "optimizer_model": "eval-optimizer-model",
        "target_model": "eval-target-model",
        "openai_compatible_base_url": "https://api.test.com/v1",
        "openai_compatible_model": "fallback-shared",
        "out_root": "/tmp/out",
    }

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("# Dummy Skill\n")
        skill_path = f.name

    class EarlyExit(Exception):
        pass

    def stop_after_config(*args, **kwargs):
        raise EarlyExit()

    monkeypatch.setattr(eval_only_script, "set_reasoning_effort", stop_after_config)

    fake_args = mock.Mock()
    fake_args.skill = skill_path

    with mock.patch("scripts.eval_only.load_config", return_value=cfg):
        with mock.patch("scripts.eval_only.parse_args", return_value=fake_args):
            with pytest.raises(EarlyExit):
                eval_only_script.main()

    assert openai_compat.OPTIMIZER_CONFIG.deployment == "eval-optimizer-model"
    assert openai_compat.TARGET_CONFIG.deployment == "eval-target-model"


def test_eval_only_minimal_yaml_resolves_runtime_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import skillopt.model.openai_compatible_backend as openai_compat

    raw = {
        "model": {
            "backend": "openai_compatible",
            "openai_compatible_model": "synthetic-model",
            "openai_compatible_base_url": "http://audit.invalid/v1",
        },
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f_yaml:
        yaml.dump(raw, f_yaml)
        config_path = f_yaml.name

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f_skill:
        f_skill.write("# Test Skill\n")
        skill_path = f_skill.name

    class EarlyExit(Exception):
        pass

    def stop_after_config(*args, **kwargs):
        raise EarlyExit()

    monkeypatch.setattr(eval_only_script, "set_reasoning_effort", stop_after_config)

    cli_args = ["eval_only.py", "--config", config_path, "--skill", skill_path]
    with mock.patch("sys.argv", cli_args):
        with pytest.raises(EarlyExit):
            eval_only_script.main()

    assert openai_compat.OPTIMIZER_CONFIG.deployment == "synthetic-model"
    assert openai_compat.TARGET_CONFIG.deployment == "synthetic-model"
    assert os.environ.get("OPTIMIZER_DEPLOYMENT") == "synthetic-model"
    assert os.environ.get("TARGET_DEPLOYMENT") == "synthetic-model"
    assert openai_compat.OPTIMIZER_CONFIG.base_url == "http://audit.invalid/v1"
    assert openai_compat.TARGET_CONFIG.base_url == "http://audit.invalid/v1"


def test_eval_only_minimal_yaml_with_role_override_and_backend_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import skillopt.model.openai_compatible_backend as openai_compat

    # Case 1: Role-specific compatible override
    raw_role = {
        "model": {
            "backend": "openai_compatible",
            "openai_compatible_model": "synthetic-shared",
            "optimizer_openai_compatible_model": "synthetic-opt",
            "openai_compatible_base_url": "http://audit.invalid/v1",
        },
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f_yaml:
        yaml.dump(raw_role, f_yaml)
        config_path = f_yaml.name

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f_skill:
        f_skill.write("# Test Skill\n")
        skill_path = f_skill.name

    class EarlyExit(Exception):
        pass

    def stop_after_config(*args, **kwargs):
        raise EarlyExit()

    monkeypatch.setattr(eval_only_script, "set_reasoning_effort", stop_after_config)

    with mock.patch("sys.argv", ["eval_only.py", "--config", config_path, "--skill", skill_path]):
        with pytest.raises(EarlyExit):
            eval_only_script.main()

    assert openai_compat.OPTIMIZER_CONFIG.deployment == "synthetic-opt"
    assert openai_compat.TARGET_CONFIG.deployment == "synthetic-shared"

    # Case 2: Minimal YAML with no model specified falls back to backend default (gpt-4o-mini)
    raw_default = {
        "model": {
            "backend": "openai_compatible",
            "openai_compatible_base_url": "http://audit.invalid/v1",
        },
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f_yaml_def:
        yaml.dump(raw_default, f_yaml_def)
        config_path_def = f_yaml_def.name

    with mock.patch("sys.argv", ["eval_only.py", "--config", config_path_def, "--skill", skill_path]):
        with pytest.raises(EarlyExit):
            eval_only_script.main()

    assert openai_compat.OPTIMIZER_CONFIG.deployment == "gpt-4o-mini"
    assert openai_compat.TARGET_CONFIG.deployment == "gpt-4o-mini"

    # Case 3: Explicit per-role CLI overrides take precedence over minimal YAML
    with mock.patch(
        "sys.argv",
        [
            "eval_only.py",
            "--config", config_path,
            "--skill", skill_path,
            "--optimizer_model", "cli-optimizer",
            "--target_model", "cli-target",
        ],
    ):
        with pytest.raises(EarlyExit):
            eval_only_script.main()

    assert openai_compat.OPTIMIZER_CONFIG.deployment == "cli-optimizer"
    assert openai_compat.TARGET_CONFIG.deployment == "cli-target"


def test_train_script_minimal_yaml_resolves_role_models() -> None:
    raw = {
        "model": {
            "backend": "openai_compatible",
            "openai_compatible_model": "synthetic-train-model",
            "openai_compatible_base_url": "http://audit.invalid/v1",
        },
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        config_path = f.name

    with mock.patch("sys.argv", ["train.py", "--config", config_path]):
        args = train_script.parse_args()
        cfg = train_script.load_config(args)

    assert cfg["optimizer_model"] == "synthetic-train-model"
    assert cfg["target_model"] == "synthetic-train-model"

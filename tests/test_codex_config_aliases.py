import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import skillopt.model.backend_config as backend_config
from skillopt.config import flatten_config, is_structured, load_config
from skillopt.model.backend_config import (
    configure_codex_exec,
    configure_codex_exec_from_config,
    validate_exec_sandbox,
)

_CODEX_ENV_KEYS = (
    "CODEX_EXEC_PATH",
    "CODEX_CLI_BIN",
    "CODEX_EXEC_SANDBOX",
    "CODEX_SANDBOX_MODE",
    "CODEX_EXEC_PROFILE",
    "CODEX_EXEC_FULL_AUTO",
    "CODEX_EXEC_REASONING_EFFORT",
    "CODEX_EXEC_USE_SDK",
    "CODEX_EXEC_NETWORK_ACCESS",
    "CODEX_EXEC_WEB_SEARCH",
    "CODEX_EXEC_APPROVAL_POLICY",
)
_CODEX_CONFIG_GLOBALS = (
    "CODEX_EXEC_PATH",
    "CODEX_EXEC_SANDBOX",
    "CODEX_EXEC_PROFILE",
    "CODEX_EXEC_FULL_AUTO",
    "CODEX_EXEC_REASONING_EFFORT",
    "CODEX_EXEC_USE_SDK",
    "CODEX_EXEC_NETWORK_ACCESS",
    "CODEX_EXEC_WEB_SEARCH",
    "CODEX_EXEC_APPROVAL_POLICY",
    "EXEC_EMPTY_RESPONSE_RETRIES",
)
_DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "_base_" / "default.yaml"


def _command_config_overrides(command: list[str]) -> list[str]:
    return [
        command[index + 1]
        for index, argument in enumerate(command)
        if argument == "-c"
    ]


@pytest.fixture(autouse=True)
def _restore_codex_runtime_state():
    """Configuration tests must not alter later tests in the same process."""
    env_before = {key: os.environ[key] for key in _CODEX_ENV_KEYS if key in os.environ}
    globals_before = {
        key: getattr(backend_config, key)
        for key in _CODEX_CONFIG_GLOBALS
    }
    yield
    for key in _CODEX_ENV_KEYS:
        if key in env_before:
            os.environ[key] = env_before[key]
        else:
            os.environ.pop(key, None)
    for key, value in globals_before.items():
        setattr(backend_config, key, value)


def test_codex_config_aliases_flatten_config():
    structured_cfg = {
        "model": {
            "sandbox": "danger-full-access",
            "codex_cli_bin": "/custom/path/codex",
        }
    }
    flat = flatten_config(structured_cfg)
    assert flat["codex_exec_sandbox"] == "danger-full-access"
    assert flat["codex_exec_path"] == "/custom/path/codex"


def test_canonical_structured_codex_config_wins_over_aliases():
    structured_cfg = {
        "model": {
            "codex_bin": "/legacy/bin",
            "codex_cli_bin": "/legacy/cli",
            "codex_path": "/legacy/path",
            "codex_exec_path": "/canonical/codex",
            "codex_sandbox": "workspace-write",
            "sandbox": "danger-full-access",
            "codex_exec_sandbox": "read-only",
        }
    }

    flat = flatten_config(structured_cfg)

    assert flat["codex_exec_path"] == "/canonical/codex"
    assert flat["codex_exec_sandbox"] == "read-only"


def test_child_aliases_override_base_canonical_values(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "model:\n"
        "  codex_exec_path: /base/codex\n"
        "  codex_exec_sandbox: read-only\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "_base_: base.yaml\n"
        "model:\n"
        "  codex_cli_bin: /child/codex\n"
        "  sandbox: danger-full-access\n",
        encoding="utf-8",
    )

    cfg = load_config(str(child))

    assert cfg["model"]["codex_exec_path"] == "/child/codex"
    assert cfg["model"]["codex_exec_sandbox"] == "danger-full-access"
    assert "codex_cli_bin" not in cfg["model"]
    assert "sandbox" not in cfg["model"]


def test_dotted_alias_overrides_replace_inherited_canonical_values(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "model:\n"
        "  codex_exec_path: /base/codex\n"
        "  codex_exec_sandbox: read-only\n",
        encoding="utf-8",
    )

    cfg = load_config(
        str(base),
        overrides=[
            "model.codex_path=/override/codex",
            "model.sandbox=workspace-write",
        ],
    )

    assert flatten_config(cfg)["codex_exec_path"] == "/override/codex"
    assert flatten_config(cfg)["codex_exec_sandbox"] == "workspace-write"


def test_dotted_codex_override_keeps_legacy_flat_config_flat(tmp_path):
    config = tmp_path / "flat.yaml"
    config.write_text(
        "batch_size: 7\n"
        "optimizer_model: keep-me\n"
        "codex_exec_sandbox: read-only\n",
        encoding="utf-8",
    )

    cfg = load_config(
        str(config),
        overrides=["model.codex_path=/override/codex"],
    )

    assert not is_structured(cfg)
    assert cfg == {
        "batch_size": 7,
        "optimizer_model": "keep-me",
        "codex_exec_sandbox": "read-only",
        "codex_exec_path": "/override/codex",
    }


@pytest.mark.parametrize("child_format", ["flat", "structured"])
def test_mixed_format_inheritance_preserves_child_precedence(
    tmp_path,
    child_format,
):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"

    if child_format == "flat":
        base.write_text(
            "model:\n"
            "  codex_exec_path: /base/codex\n"
            "train:\n"
            "  batch_size: 3\n",
            encoding="utf-8",
        )
        child.write_text(
            "_base_: base.yaml\n"
            "codex_path: /child/codex\n"
            "batch_size: 9\n"
            "custom_runtime_value: preserved\n",
            encoding="utf-8",
        )
    else:
        base.write_text(
            "codex_exec_path: /base/codex\n"
            "batch_size: 3\n"
            "custom_runtime_value: preserved\n",
            encoding="utf-8",
        )
        child.write_text(
            "_base_: base.yaml\n"
            "model:\n"
            "  codex_path: /child/codex\n"
            "train:\n"
            "  batch_size: 9\n",
            encoding="utf-8",
        )

    flat = flatten_config(load_config(str(child)))

    assert flat["codex_exec_path"] == "/child/codex"
    assert flat["batch_size"] == 9
    assert flat["custom_runtime_value"] == "preserved"


@pytest.mark.parametrize(
    ("overrides", "expected_path", "expected_sandbox"),
    [
        (
            ["codex_path=/override/codex", "sandbox=workspace-write"],
            "/override/codex",
            "workspace-write",
        ),
        (
            [
                "codex_exec_path=/canonical/codex",
                "codex_exec_sandbox=danger-full-access",
            ],
            "/canonical/codex",
            "danger-full-access",
        ),
    ],
)
def test_flat_codex_overrides_apply_to_structured_config(
    tmp_path,
    overrides,
    expected_path,
    expected_sandbox,
):
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n"
        "  codex_exec_path: /base/codex\n"
        "  codex_exec_sandbox: read-only\n",
        encoding="utf-8",
    )

    cfg = load_config(str(config), overrides=overrides)
    flat = flatten_config(cfg)

    assert flat["codex_exec_path"] == expected_path
    assert flat["codex_exec_sandbox"] == expected_sandbox
    assert "codex_exec_path" not in cfg
    assert "codex_exec_sandbox" not in cfg


def test_all_flat_canonical_codex_overrides_apply_to_structured_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n"
        "  backend: codex\n",
        encoding="utf-8",
    )
    expected = {
        "codex_exec_path": "/override/codex",
        "codex_exec_sandbox": "read-only",
        "codex_exec_profile": "override-profile",
        "codex_exec_full_auto": True,
        "codex_exec_reasoning_effort": "high",
        "codex_exec_use_sdk": "cli",
        "codex_exec_network_access": True,
        "codex_exec_web_search": False,
        "codex_exec_approval_policy": "on-request",
    }

    cfg = load_config(
        str(config),
        overrides=[f"{key}={str(value).lower()}" for key, value in expected.items()],
    )
    flat = flatten_config(cfg)

    for key, value in expected.items():
        assert flat[key] == value
        assert key not in cfg


def test_legacy_flat_child_alias_overrides_base_canonical_value(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("codex_exec_path: /base/codex\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text(
        "_base_: base.yaml\n"
        "codex_bin: /child/codex\n",
        encoding="utf-8",
    )

    cfg = load_config(str(child))

    assert cfg["codex_exec_path"] == "/child/codex"
    assert "codex_bin" not in cfg


def test_default_config_preserves_preloaded_codex_environment():
    env = os.environ.copy()
    for key in _CODEX_ENV_KEYS:
        env.pop(key, None)
    env.update(
        {
            "CODEX_CLI_BIN": "/env/codex",
            "CODEX_SANDBOX_MODE": "danger-full-access",
            "CODEX_EXEC_PROFILE": "env-profile",
            "CODEX_EXEC_REASONING_EFFORT": "high",
            "CODEX_EXEC_USE_SDK": "cli",
            "CODEX_EXEC_NETWORK_ACCESS": "true",
            "CODEX_EXEC_WEB_SEARCH": "true",
            "CODEX_EXEC_APPROVAL_POLICY": "on-request",
        }
    )
    code = f"""
import json
from skillopt.config import flatten_config, load_config
from skillopt.model.backend_config import (
    configure_codex_exec_from_config,
    get_codex_exec_config,
)
configure_codex_exec_from_config(flatten_config(load_config({str(_DEFAULT_CONFIG)!r})))
print(json.dumps(get_codex_exec_config()))
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    loaded = json.loads(proc.stdout)

    assert loaded["path"] == "/env/codex"
    assert loaded["sandbox"] == "danger-full-access"
    assert loaded["profile"] == "env-profile"
    assert loaded["reasoning_effort"] == "high"
    assert loaded["use_sdk"] == "cli"
    assert loaded["network_access"] is True
    assert loaded["web_search"] is True
    assert loaded["approval_policy"] == "on-request"


def test_codex_backend_env_aliases_are_isolated_in_subprocess():
    env = os.environ.copy()
    for key in (
        "CODEX_EXEC_PATH",
        "CODEX_PATH",
        "CODEX_CLI_BIN",
        "CODEX_EXEC_SANDBOX",
        "CODEX_SANDBOX_MODE",
        "CODEX_SANDBOX",
    ):
        env.pop(key, None)
    env["CODEX_CLI_BIN"] = "/env/path/codex"
    env["CODEX_SANDBOX_MODE"] = "danger-full-access"
    code = """
import json
from skillopt.model import backend_config, codex_backend
print(json.dumps({
    "config_path": backend_config.CODEX_EXEC_PATH,
    "config_sandbox": backend_config.CODEX_EXEC_SANDBOX,
    "backend_path": codex_backend.CODEX_BIN,
    "backend_sandbox": codex_backend.CODEX_SANDBOX_MODE,
}))
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    loaded = json.loads(proc.stdout)

    assert loaded == {
        "config_path": "/env/path/codex",
        "config_sandbox": "danger-full-access",
        "backend_path": "/env/path/codex",
        "backend_sandbox": "danger-full-access",
    }


def test_configure_codex_exec_sets_aliases(monkeypatch):
    monkeypatch.setattr(backend_config, "CODEX_EXEC_PATH", "")
    monkeypatch.setattr(backend_config, "CODEX_EXEC_SANDBOX", "")

    configure_codex_exec(path="/configured/codex", sandbox="danger-full-access")

    assert os.environ["CODEX_EXEC_PATH"] == "/configured/codex"
    assert os.environ["CODEX_CLI_BIN"] == "/configured/codex"
    assert os.environ["CODEX_EXEC_SANDBOX"] == "danger-full-access"
    assert os.environ["CODEX_SANDBOX_MODE"] == "danger-full-access"


def test_validate_exec_sandbox():
    assert validate_exec_sandbox("read-only") == "read-only"
    assert validate_exec_sandbox("workspace-write") == "workspace-write"
    assert validate_exec_sandbox("danger-full-access") == "danger-full-access"

    with pytest.raises(ValueError, match="Invalid codex_exec sandbox"):
        validate_exec_sandbox("invalid-mode")

    with pytest.raises(ValueError, match="Invalid codex_exec sandbox"):
        configure_codex_exec(sandbox="unrestricted-bad-mode")


@pytest.mark.parametrize(
    ("cfg", "expected_path", "expected_sandbox"),
    [
        (
            {
                "codex_exec_path": "/path/exec",
                "codex_path": "/path/codex",
                "codex_cli_bin": "/path/cli",
                "codex_bin": "/path/bin",
                "codex_exec_sandbox": "read-only",
                "sandbox": "danger-full-access",
                "codex_sandbox": "workspace-write",
            },
            "/path/exec",
            "read-only",
        ),
        (
            {
                "codex_cli_bin": "/path/cli",
                "sandbox": "danger-full-access",
            },
            "/path/cli",
            "danger-full-access",
        ),
    ],
)
def test_shared_config_entry_point_alias_precedence(cfg, expected_path, expected_sandbox):
    with mock.patch.object(backend_config, "configure_codex_exec") as configure:
        configure_codex_exec_from_config(cfg)

    assert configure.call_args.kwargs["path"] == expected_path
    assert configure.call_args.kwargs["sandbox"] == expected_sandbox


class _EntryPointReached(Exception):
    pass


def test_trainer_uses_shared_codex_config_entry_point(monkeypatch, tmp_path):
    import skillopt.engine.trainer as trainer_module

    cfg = {
        "out_root": str(tmp_path),
        "model_backend": "azure_openai",
        "optimizer_backend": "openai_chat",
        "target_backend": "openai_chat",
        "optimizer_model": "optimizer",
        "target_model": "target",
        "codex_exec_sandbox": "read-only",
    }
    adapter = mock.Mock()
    adapter.get_dataloader.return_value = None
    for name in (
        "configure_azure_openai",
        "set_optimizer_backend",
        "set_target_backend",
        "set_optimizer_deployment",
        "set_target_deployment",
    ):
        monkeypatch.setattr(trainer_module, name, mock.Mock())

    def stop_at_codex_config(actual_cfg):
        assert actual_cfg is cfg
        raise _EntryPointReached

    monkeypatch.setattr(
        trainer_module,
        "configure_codex_exec_from_config",
        stop_at_codex_config,
    )

    with pytest.raises(_EntryPointReached):
        trainer_module.ReflACTTrainer(cfg, adapter).train()


def test_trainer_configures_target_backend_before_adapter_setup(
    monkeypatch,
    tmp_path,
):
    import skillopt.engine.trainer as trainer_module

    cfg = {
        "out_root": str(tmp_path),
        "model_backend": "azure_openai",
        "optimizer_backend": "openai_chat",
        "target_backend": "codex_exec",
    }
    for name in ("OPTIMIZER_BACKEND", "TARGET_BACKEND"):
        monkeypatch.setattr(backend_config, name, getattr(backend_config, name))
        # Record even an originally absent environment key so teardown removes
        # the value written by the real backend setter.
        monkeypatch.setenv(name, os.environ.get(name, ""))

    adapter = mock.Mock()

    def assert_backend_is_ready(actual_cfg):
        assert actual_cfg is cfg
        assert backend_config.get_target_backend() == "codex_exec"
        raise _EntryPointReached

    adapter.setup.side_effect = assert_backend_is_ready

    with pytest.raises(_EntryPointReached):
        trainer_module.ReflACTTrainer(cfg, adapter).train()


def test_eval_only_uses_shared_codex_config_entry_point(monkeypatch, tmp_path):
    import scripts.eval_only as eval_script

    skill_path = tmp_path / "skill.md"
    skill_path.write_text("# Test skill\n", encoding="utf-8")
    cfg = {
        "out_root": str(tmp_path / "output"),
        "model_backend": "azure_openai",
        "optimizer_backend": "openai_chat",
        "target_backend": "openai_chat",
        "optimizer_model": "optimizer",
        "target_model": "target",
        "codex_exec_sandbox": "read-only",
    }
    args = SimpleNamespace(
        config="unused.yaml",
        skill=str(skill_path),
        split=None,
        cfg_options=[],
        backend=None,
    )
    monkeypatch.setattr(eval_script, "parse_args", lambda: args)
    monkeypatch.setattr("skillopt.config.load_config", lambda *args, **kwargs: dict(cfg))
    for name in (
        "configure_azure_openai",
        "set_optimizer_backend",
        "set_target_backend",
        "set_optimizer_deployment",
        "set_target_deployment",
    ):
        monkeypatch.setattr(eval_script, name, mock.Mock())

    def stop_at_codex_config(actual_cfg):
        assert actual_cfg["codex_exec_sandbox"] == "read-only"
        raise _EntryPointReached

    monkeypatch.setattr(
        eval_script,
        "configure_codex_exec_from_config",
        stop_at_codex_config,
    )

    with pytest.raises(_EntryPointReached):
        eval_script.main()


@pytest.mark.parametrize("full_auto_setting", [True, False])
def test_configure_full_auto_is_deprecated_and_does_not_mutate_state(
    monkeypatch,
    full_auto_setting,
):
    original_setting = not full_auto_setting
    monkeypatch.setattr(backend_config, "CODEX_EXEC_FULL_AUTO", original_setting)
    monkeypatch.setenv("CODEX_EXEC_FULL_AUTO", "unchanged")

    with pytest.warns(FutureWarning, match="deprecated and ignored"):
        configure_codex_exec(full_auto=full_auto_setting)

    assert backend_config.CODEX_EXEC_FULL_AUTO is original_setting
    assert os.environ["CODEX_EXEC_FULL_AUTO"] == "unchanged"


def test_sequential_configs_restore_process_startup_baseline():
    baseline = backend_config.get_codex_exec_config()
    environment_baseline = {
        key: os.environ[key]
        for key in backend_config._CODEX_EXEC_MUTATED_ENV_KEYS
        if key in os.environ
    }
    configured = {
        "codex_exec_path": "/first-run/codex",
        "codex_exec_sandbox": "danger-full-access",
        "codex_exec_profile": "first-run",
        "codex_exec_reasoning_effort": "high",
        "codex_exec_use_sdk": "cli",
        "codex_exec_network_access": True,
        "codex_exec_web_search": True,
        "codex_exec_approval_policy": "on-request",
    }

    configure_codex_exec_from_config(configured)
    assert backend_config.get_codex_exec_config()["sandbox"] == "danger-full-access"

    configure_codex_exec_from_config({})
    restored = backend_config.get_codex_exec_config()

    for key in (
        "path",
        "sandbox",
        "profile",
        "reasoning_effort",
        "use_sdk",
        "network_access",
        "web_search",
        "approval_policy",
    ):
        assert restored[key] == baseline[key]
    for key in backend_config._CODEX_EXEC_MUTATED_ENV_KEYS:
        if key in environment_baseline:
            assert os.environ[key] == environment_baseline[key]
        else:
            assert key not in os.environ


@pytest.mark.parametrize("disabled_value", ["false", "0", "off", "no"])
def test_codex_permission_strings_are_parsed_as_false(disabled_value):
    configure_codex_exec(
        network_access=disabled_value,
        web_search=disabled_value,
    )

    configured = backend_config.get_codex_exec_config()
    assert configured["network_access"] is False
    assert configured["web_search"] is False


@pytest.mark.parametrize(
    "setting_name",
    ["network_access", "web_search"],
)
def test_invalid_codex_permission_strings_are_rejected(setting_name):
    with pytest.raises(ValueError, match=f"codex_exec_{setting_name}"):
        configure_codex_exec(**{setting_name: "sometimes"})


def test_full_auto_environment_variable_warns_in_subprocess():
    env = os.environ.copy()
    env["CODEX_EXEC_FULL_AUTO"] = "true"

    proc = subprocess.run(
        [
            sys.executable,
            "-W",
            "always::FutureWarning",
            "-c",
            (
                "from skillopt.model.backend_config import get_codex_exec_config; "
                "print(get_codex_exec_config()['full_auto'])"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "CODEX_EXEC_FULL_AUTO is deprecated and ignored" in proc.stderr
    assert proc.stdout.strip() == "False"


@pytest.mark.parametrize("full_auto_setting", [True, False])
@mock.patch("skillopt.model.codex_harness.subprocess.run")
def test_codex_harness_full_auto_is_ignored(
    mock_run,
    monkeypatch,
    tmp_path,
    full_auto_setting,
):
    from skillopt.model.codex_harness import run_codex_exec

    configure_codex_exec(
        sandbox="danger-full-access",
        use_sdk="cli",
        approval_policy="never",
    )
    monkeypatch.setattr(backend_config, "EXEC_EMPTY_RESPONSE_RETRIES", 0)

    mock_proc = mock.MagicMock(returncode=0, stdout="", stderr="")
    mock_run.return_value = mock_proc

    with (
        mock.patch("skillopt.model.codex_harness._persist_codex_artifacts"),
        mock.patch("skillopt.model.azure_openai.tracker.record", create=True),
        pytest.warns(FutureWarning, match="deprecated and ignored"),
    ):
        run_codex_exec(
            work_dir=str(tmp_path),
            prompt="test",
            model="test-model",
            timeout=10,
            full_auto=full_auto_setting,
        )

    mock_run.assert_called_once()
    command = mock_run.call_args.args[0]

    assert command[command.index("--sandbox") + 1] == "danger-full-access"
    assert "--approval-policy" not in command
    assert "--full-auto" not in command
    config_overrides = _command_config_overrides(command)
    assert 'approval_policy="never"' in config_overrides


@pytest.mark.parametrize(
    ("network_access", "web_search", "expected_network", "expected_search"),
    [
        (False, False, "false", "disabled"),
        (False, True, "false", "live"),
        (True, False, "true", "disabled"),
        (True, True, "true", "live"),
    ],
)
@mock.patch("skillopt.model.codex_harness.subprocess.run")
def test_codex_harness_cli_forwards_network_and_web_search(
    mock_run,
    monkeypatch,
    tmp_path,
    network_access,
    web_search,
    expected_network,
    expected_search,
):
    from skillopt.model.codex_harness import run_codex_exec

    configure_codex_exec(
        use_sdk="cli",
        network_access=network_access,
        web_search=web_search,
    )
    monkeypatch.setattr(backend_config, "EXEC_EMPTY_RESPONSE_RETRIES", 0)
    mock_run.return_value = mock.MagicMock(returncode=0, stdout="", stderr="")

    with (
        mock.patch("skillopt.model.codex_harness._persist_codex_artifacts"),
        mock.patch("skillopt.model.azure_openai.tracker.record", create=True),
    ):
        run_codex_exec(
            work_dir=str(tmp_path),
            prompt="test",
            model="test-model",
            timeout=10,
        )

    command = mock_run.call_args.args[0]
    overrides = _command_config_overrides(command)
    assert (
        f"sandbox_workspace_write.network_access={expected_network}" in overrides
    )
    assert f'web_search="{expected_search}"' in overrides


@pytest.mark.parametrize(
    ("network_access", "web_search", "expected_network", "expected_search"),
    [
        (False, False, "false", "disabled"),
        (False, True, "false", "live"),
        (True, False, "true", "disabled"),
        (True, True, "true", "live"),
    ],
)
def test_codex_optimizer_cli_forwards_network_and_web_search(
    monkeypatch,
    network_access,
    web_search,
    expected_network,
    expected_search,
):
    from skillopt.model import codex_backend

    configure_codex_exec(
        network_access=network_access,
        web_search=web_search,
    )
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        output_path = command[command.index("--output-last-message") + 1]
        Path(output_path).write_text("ok", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(codex_backend.subprocess, "run", fake_run)

    response, _usage = codex_backend._run_codex_exec(
        model="test-model",
        prompt="test",
        attachments=[],
        output_schema=None,
        timeout=10,
    )

    assert response == "ok"
    overrides = _command_config_overrides(commands[0])
    assert (
        f"sandbox_workspace_write.network_access={expected_network}" in overrides
    )
    assert f'web_search="{expected_search}"' in overrides

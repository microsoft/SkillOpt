"""OpenCode CLI as an optimizer backend (``opencode_exec``).

The fixtures under ``tests/fixtures/opencode_exec`` are real ``opencode run
--format json`` output captured from OpenCode v2.2.5. The tests swap the CLI for
a fake that records how it was launched, so no live provider is called.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

import scripts.train as train_script
from skillopt.model import backend_config, opencode_backend
from skillopt.model.common import default_model_for_backend, normalize_backend_name

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "opencode_exec"

_FAKE = """#!{python}
import json, os, sys, time
if sys.argv[1:] == ["--version"]:
    print(os.environ.get("FAKE_OPENCODE_VERSION", "2.2.5"))
    sys.exit(0)
record = {{
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "cwd_entries": sorted(os.listdir(".")),
    "stdin": sys.stdin.read(),
    "pwd": os.environ.get("PWD"),
    "env": {{k: v for k, v in os.environ.items() if k.startswith(("OPENCODE", "XDG_CONFIG_HOME", "NO_COLOR"))}},
}}
cfg_home = os.environ.get("XDG_CONFIG_HOME", "")
cfg_path = os.path.join(cfg_home, "opencode", "opencode.json")
record["config"] = json.load(open(cfg_path)) if cfg_home and os.path.exists(cfg_path) else None
with open(os.environ["FAKE_OPENCODE_RECORD"], "a") as f:
    f.write(json.dumps(record) + "\\n")
if os.environ.get("FAKE_OPENCODE_SLEEP"):
    time.sleep(float(os.environ["FAKE_OPENCODE_SLEEP"]))
sys.stdout.write(open(os.environ["FAKE_OPENCODE_STDOUT"]).read())
sys.stderr.write(os.environ.get("FAKE_OPENCODE_STDERR", ""))
sys.exit(int(os.environ.get("FAKE_OPENCODE_EXIT", "0")))
"""


@pytest.fixture
def fake_opencode(tmp_path, monkeypatch):
    script = tmp_path / "opencode"
    script.write_text(_FAKE.format(python=sys.executable), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    record = tmp_path / "record.jsonl"
    monkeypatch.setenv("FAKE_OPENCODE_RECORD", str(record))
    monkeypatch.setenv("FAKE_OPENCODE_STDOUT", str(_FIXTURES / "run_text.jsonl"))
    monkeypatch.delenv("FAKE_OPENCODE_EXIT", raising=False)
    monkeypatch.delenv("FAKE_OPENCODE_SLEEP", raising=False)
    monkeypatch.setenv("OPENCODE_CONFIG", "/elsewhere/opencode.json")
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", "/elsewhere")
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '{"mcp":{}}')
    backend_config.configure_opencode_exec(path=str(script))
    opencode_backend.set_optimizer_deployment("openai/gpt-6.1-sol")
    opencode_backend.reset_token_tracker()
    monkeypatch.setattr(opencode_backend.time, "sleep", lambda _s: None)

    def calls() -> list[dict]:
        if not record.exists():
            return []
        return [json.loads(line) for line in record.read_text().splitlines()]

    yield calls
    backend_config.configure_opencode_exec(path="opencode")


def test_optimizer_call_returns_final_text_and_usage(fake_opencode) -> None:
    text, usage = opencode_backend.chat_optimizer(system="be terse", user="ping", retries=1)

    assert text == "PONG"
    assert usage == {"prompt_tokens": 20697, "completion_tokens": 6, "total_tokens": 20703}
    summary = opencode_backend.get_token_summary()
    assert summary["optimizer"]["calls"] == 1
    assert summary["_total"]["total_tokens"] == 20703


def test_call_is_isolated_from_user_config_tools_and_project(fake_opencode) -> None:
    opencode_backend.chat_optimizer(system="SYSTEM RULES", user="USER TEXT", retries=1)

    (call,) = fake_opencode()
    argv = call["argv"]
    assert argv[0] == "run"
    assert "--standalone" in argv
    assert argv[argv.index("--format") + 1] == "json"
    assert argv[argv.index("-m") + 1] == "openai/gpt-6.1-sol"
    agent = argv[argv.index("--agent") + 1]

    # The prompt travels on stdin, the system prompt as the agent prompt.
    assert call["stdin"] == "USER TEXT"
    agent_cfg = call["config"]["agent"][agent]
    assert agent_cfg["prompt"] == "SYSTEM RULES"
    # The global list is the only gate; the agent entry carries no rules.
    assert call["config"]["permissions"] == [{"action": "*", "resource": "*", "effect": "deny"}]
    for key in ("OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE_DISABLE_CLAUDE_CODE", "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"):
        assert call["env"][key] == "1"
    assert "mcp" not in call["config"]

    # User config never reaches the child; it runs in an empty directory.
    assert call["env"]["XDG_CONFIG_HOME"] != os.environ.get("XDG_CONFIG_HOME")
    for key in ("OPENCODE_CONFIG", "OPENCODE_CONFIG_DIR", "OPENCODE_CONFIG_CONTENT"):
        assert key not in call["env"]
    assert call["cwd_entries"] == []
    # OpenCode resolves its project directory from PWD, not from the process cwd.
    assert os.path.realpath(call["pwd"]) == os.path.realpath(call["cwd"])
    assert not Path(call["cwd"]).exists()


def test_error_event_is_reported_with_provider_message(fake_opencode, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_OPENCODE_STDOUT", str(_FIXTURES / "run_error_model.jsonl"))
    monkeypatch.setenv("FAKE_OPENCODE_EXIT", "1")

    with pytest.raises(RuntimeError, match="not supported when using Codex with a ChatGPT account"):
        opencode_backend.chat_optimizer(system="s", user="u", retries=2)
    assert len(fake_opencode()) == 2
    assert opencode_backend.get_token_summary()["_total"]["calls"] == 0


def test_nonzero_exit_without_events_reports_stderr_tail(fake_opencode, monkeypatch, tmp_path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("FAKE_OPENCODE_STDOUT", str(empty))
    monkeypatch.setenv("FAKE_OPENCODE_EXIT", "3")
    monkeypatch.setenv("FAKE_OPENCODE_STDERR", "boom: server failed to start")

    with pytest.raises(RuntimeError, match="exited 3.*server failed to start"):
        opencode_backend.chat_optimizer(system="s", user="u", retries=1)


def test_garbage_output_is_an_error_not_text(fake_opencode, monkeypatch, tmp_path) -> None:
    junk = tmp_path / "junk.jsonl"
    junk.write_text("not json at all\n", encoding="utf-8")
    monkeypatch.setenv("FAKE_OPENCODE_STDOUT", str(junk))

    with pytest.raises(RuntimeError, match="malformed JSONL"):
        opencode_backend.chat_optimizer(system="s", user="u", retries=1)


def test_timeout_does_not_hang(fake_opencode, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_OPENCODE_SLEEP", "5")

    with pytest.raises(RuntimeError, match="timed out after 1s"):
        opencode_backend.chat_optimizer(system="s", user="u", retries=1, timeout=1)


def test_tool_calls_are_rejected_explicitly(fake_opencode) -> None:
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    with pytest.raises(NotImplementedError, match="tool"):
        opencode_backend.chat_optimizer_messages(
            [{"role": "user", "content": "x"}], tools=tools
        )
    assert fake_opencode() == []


def test_message_list_folds_history_into_the_prompt(fake_opencode) -> None:
    text, _usage = opencode_backend.chat_optimizer_messages(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ],
        retries=1,
    )

    assert text == "PONG"
    (call,) = fake_opencode()
    agent = call["argv"][call["argv"].index("--agent") + 1]
    assert call["config"]["agent"][agent]["prompt"] == "SYS"
    assert call["stdin"].index("first") < call["stdin"].index("reply") < call["stdin"].index("second")


def test_dispatcher_routes_optimizer_calls(fake_opencode, monkeypatch) -> None:
    import skillopt.model as model

    monkeypatch.setattr(backend_config, "OPTIMIZER_BACKEND", backend_config.OPTIMIZER_BACKEND)
    monkeypatch.setenv("OPTIMIZER_BACKEND", backend_config.OPTIMIZER_BACKEND)
    model.set_optimizer_backend("opencode")

    assert backend_config.get_optimizer_backend() == "opencode_exec"
    assert backend_config.is_optimizer_chat_backend()
    text, _usage = model.chat_optimizer(system="s", user="u", retries=1)
    assert text == "PONG"
    assert model.get_token_summary()["optimizer"]["calls"] >= 1


def test_names_defaults_and_roles(monkeypatch) -> None:
    # Restore the role globals this test sets.
    monkeypatch.setattr(backend_config, "TARGET_BACKEND", backend_config.TARGET_BACKEND)
    monkeypatch.setenv("TARGET_BACKEND", backend_config.TARGET_BACKEND)
    for alias in ("opencode", "opencode_exec", "opencode_chat", "opencode-cli"):
        assert normalize_backend_name(alias) == "opencode_exec"
    assert default_model_for_backend("opencode_exec").startswith("openai/")
    # Both roles: the target runs through codex_harness.run_opencode_exec.
    backend_config.set_target_backend("opencode")
    assert backend_config.get_target_backend() == "opencode_exec"
    assert backend_config.is_target_exec_backend()
    assert not backend_config.is_target_chat_backend()


def _train_cfg(monkeypatch, *extra_argv: str) -> dict:
    monkeypatch.setattr(
        sys,
        "argv",
        ["skillopt-train", "--config", str(_ROOT / "configs" / "searchqa" / "default.yaml"), *extra_argv],
    )
    return train_script.load_config(train_script.parse_args())


def test_train_optimizer_opt_in_normalizes_model(monkeypatch) -> None:
    cfg = _train_cfg(monkeypatch, "--optimizer_backend", "opencode_exec")

    assert cfg["optimizer_backend"] == "opencode_exec"
    assert cfg["optimizer_model"] == default_model_for_backend("opencode_exec")
    assert cfg["target_backend"] == "openai_chat"


def test_train_keeps_explicit_optimizer_model_and_path(monkeypatch) -> None:
    cfg = _train_cfg(
        monkeypatch,
        "--optimizer_backend",
        "opencode",
        "--optimizer_model",
        "openai/gpt-6.1-sol",
        "--opencode_exec_path",
        "/opt/bin/opencode",
    )

    assert cfg["optimizer_backend"] == "opencode_exec"
    assert cfg["optimizer_model"] == "openai/gpt-6.1-sol"
    assert cfg["opencode_exec_path"] == "/opt/bin/opencode"


def test_train_alias_gets_the_namespaced_default_model(monkeypatch) -> None:
    cfg = _train_cfg(monkeypatch, "--optimizer_backend", "opencode")

    assert cfg["optimizer_backend"] == "opencode_exec"
    assert cfg["optimizer_model"] == default_model_for_backend("opencode_exec")


# ── new-backend.md, step 3: routing, precedence, setters, tokens, errors ─────


def test_dispatcher_routes_message_list_calls(fake_opencode, monkeypatch) -> None:
    import skillopt.model as model

    monkeypatch.setattr(backend_config, "OPTIMIZER_BACKEND", backend_config.OPTIMIZER_BACKEND)
    monkeypatch.setenv("OPTIMIZER_BACKEND", backend_config.OPTIMIZER_BACKEND)
    model.set_optimizer_backend("opencode_exec")

    text, _usage = model.chat_optimizer_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], retries=1)

    assert text == "PONG"
    (call,) = fake_opencode()
    assert call["stdin"] == "U"


def test_dispatcher_setters_reach_the_backend(fake_opencode, monkeypatch) -> None:
    import skillopt.model as model

    monkeypatch.setattr(backend_config, "OPTIMIZER_BACKEND", backend_config.OPTIMIZER_BACKEND)
    monkeypatch.setenv("OPTIMIZER_BACKEND", backend_config.OPTIMIZER_BACKEND)
    model.set_optimizer_backend("opencode_exec")
    model.set_optimizer_deployment("openai/gpt-6-luna")
    model.set_target_deployment("openai/gpt-6.1-sol")
    model.set_reasoning_effort("high")

    model.chat_optimizer(system="s", user="u", retries=1)

    (call,) = fake_opencode()
    assert call["argv"][call["argv"].index("-m") + 1] == "openai/gpt-6-luna"
    assert opencode_backend.TARGET_DEPLOYMENT == "openai/gpt-6.1-sol"
    # Reasoning is chosen per model variant in OpenCode; the setting is kept, not sent.
    assert opencode_backend.REASONING_EFFORT == "high"
    assert "high" not in " ".join(call["argv"])
    model.set_reasoning_effort(None)


def test_shared_token_summary_counts_opencode_once(fake_opencode, monkeypatch) -> None:
    import skillopt.model as model

    monkeypatch.setattr(backend_config, "OPTIMIZER_BACKEND", backend_config.OPTIMIZER_BACKEND)
    monkeypatch.setenv("OPTIMIZER_BACKEND", backend_config.OPTIMIZER_BACKEND)
    model.set_optimizer_backend("opencode_exec")
    model.reset_token_tracker()

    model.chat_optimizer(system="s", user="u", retries=1, stage="analyst")

    summary = model.get_token_summary()
    assert summary["analyst"] == {"calls": 1, "prompt_tokens": 20697, "completion_tokens": 6, "total_tokens": 20703}
    assert summary["_total"]["calls"] == 1 and summary["_total"]["total_tokens"] == 20703
    model.reset_token_tracker()
    assert model.get_token_summary()["_total"]["calls"] == 0


def test_config_path_wins_over_env_and_blank_keeps_env(monkeypatch) -> None:
    from skillopt.engine.trainer import ReflACTTrainer  # noqa: F401  (trainer applies the same rule)

    monkeypatch.setattr(backend_config, "OPENCODE_EXEC_PATH", "/env/opencode")
    monkeypatch.setattr(backend_config, "OPENCODE_EXEC_ALLOW_SHELL", True)
    monkeypatch.setenv("OPENCODE_EXEC_PATH", "/env/opencode")

    backend_config.configure_opencode_exec(path=None, allow_shell=None)
    assert backend_config.get_opencode_exec_config()["path"] == "/env/opencode"
    assert backend_config.get_opencode_exec_config()["allow_shell"] is True

    backend_config.configure_opencode_exec(path="/cfg/opencode", allow_shell="false")
    assert backend_config.get_opencode_exec_config()["path"] == "/cfg/opencode"
    assert backend_config.get_opencode_exec_config()["allow_shell"] is False

    with pytest.raises(ValueError, match="opencode_exec_allow_shell"):
        backend_config.configure_opencode_exec(allow_shell="maybe")


def test_base_config_declares_the_opencode_keys_blank(monkeypatch) -> None:
    import yaml

    base = yaml.safe_load((_ROOT / "configs" / "_base_" / "default.yaml").read_text())["model"]
    assert base["opencode_exec_path"] == ""
    assert base["opencode_exec_allow_shell"] is None


def test_missing_cli_is_an_actionable_error(tmp_path) -> None:
    backend_config.configure_opencode_exec(path=str(tmp_path / "no-such-opencode"))
    try:
        with pytest.raises(RuntimeError, match="OpenCode could not be started"):
            opencode_backend.chat_optimizer(system="s", user="u", retries=1)
    finally:
        backend_config.configure_opencode_exec(path="opencode")


def test_opencode_1_is_refused_with_the_supported_range(fake_opencode, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_OPENCODE_VERSION", "1.14.0")

    with pytest.raises(RuntimeError, match=r"OpenCode 2\.x.*1\.14\.0"):
        opencode_backend.chat_optimizer(system="s", user="u", retries=3)
    assert fake_opencode() == []


def test_default_cli_path_is_resolved_at_import_like_the_other_exec_backends(tmp_path) -> None:
    # Windows npm installs are .cmd shims that CreateProcess cannot find by bare name (#250).
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "my-opencode"
    shim.write_text("#!/bin/sh\n")
    shim.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", "OPENCODE_EXEC_PATH": "my-opencode"}
    out = subprocess.run(
        [sys.executable, "-c", "from skillopt.model import backend_config as b; print(b.OPENCODE_EXEC_PATH)"],
        capture_output=True, text=True, env=env, cwd=_ROOT, check=True,
    )
    assert out.stdout.strip() == str(shim)

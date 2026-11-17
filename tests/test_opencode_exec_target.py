"""OpenCode CLI as an exec target (``opencode_exec``).

Fixtures are real OpenCode 2.2.5 output (paths rewritten to ``/tmp/skillopt-probe``).
A fake CLI records how it was launched, so no model is called.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

import scripts.train as train_script
from skillopt.model import backend_config, codex_harness

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "opencode_exec"

_FAKE = """#!{python}
import json, os, sys
if sys.argv[1:] == ["--version"]:
    print(os.environ.get("FAKE_OPENCODE_VERSION", "2.2.5"))
    sys.exit(0)
record = {{
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "stdin": sys.stdin.read(),
    "pwd": os.environ.get("PWD"),
    "env": {{k: v for k, v in os.environ.items() if k.startswith(("OPENCODE", "XDG_CONFIG_HOME"))}},
}}
cfg = os.path.join(os.environ.get("XDG_CONFIG_HOME", ""), "opencode", "opencode.json")
record["config"] = json.load(open(cfg)) if os.path.exists(cfg) else None
with open(os.environ["FAKE_OPENCODE_RECORD"], "a") as f:
    f.write(json.dumps(record) + "\\n")
outputs = os.environ["FAKE_OPENCODE_STDOUT"].split(os.pathsep)
n = sum(1 for _ in open(os.environ["FAKE_OPENCODE_RECORD"]))
sys.stdout.write(open(outputs[min(n, len(outputs)) - 1]).read())
sys.exit(int(os.environ.get("FAKE_OPENCODE_EXIT", "0")))
"""

DENY_ALL = {"action": "*", "resource": "*", "effect": "deny"}


def _allowed(config: dict) -> set[tuple[str, str]]:
    rules = config["permissions"]
    assert rules[0] == DENY_ALL
    return {(r["action"], r["resource"]) for r in rules[1:] if r["effect"] == "allow"}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "opencode"
    script.write_text(_FAKE.format(python=sys.executable), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    record = tmp_path / "record.jsonl"
    monkeypatch.setenv("FAKE_OPENCODE_RECORD", str(record))
    monkeypatch.setenv("FAKE_OPENCODE_STDOUT", str(_FIXTURES / "target_skill_read.jsonl"))
    monkeypatch.delenv("FAKE_OPENCODE_EXIT", raising=False)
    backend_config.configure_opencode_exec(path=str(script), allow_shell=False)
    yield lambda: [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
    backend_config.configure_opencode_exec(path="opencode", allow_shell=False)


def _work(tmp_path: Path) -> Path:
    pred = tmp_path / "predictions" / "item-1"
    work = pred / "workspace"
    (work / ".agents" / "skills" / "skillopt-target").mkdir(parents=True)
    (work / "task.md").write_text("task")
    return work


def test_read_only_run_returns_answer_and_trace(fake, tmp_path) -> None:
    work = _work(tmp_path)

    response, raw = codex_harness.run_opencode_exec(
        work_dir=str(work), prompt="Read `task.md` and answer.", model="openai/gpt-6.1-sol", timeout=60
    )

    assert response == "<answer>The capital of France is Paris. Data value: 4242. BANANA42</answer>"
    assert '"tool_use"' in raw
    (call,) = fake()
    argv = call["argv"]
    assert argv[:2] == ["run", "--standalone"]
    assert argv[argv.index("--format") + 1] == "json"
    assert argv[argv.index("-m") + 1] == "openai/gpt-6.1-sol"
    assert os.path.realpath(call["cwd"]) == os.path.realpath(work)
    # OpenCode resolves its project directory from PWD, not from the process cwd.
    assert os.path.realpath(call["pwd"]) == os.path.realpath(work)
    assert _allowed(call["config"]) == {("read", "*"), ("grep", "*"), ("glob", "*"), ("list", "*")}
    assert "mcp" not in call["config"]
    assert "Do not modify files." in call["stdin"]
    assert "Read `task.md` and answer." in call["stdin"]
    for key in ("OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE_DISABLE_CLAUDE_CODE", "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"):
        assert call["env"][key] == "1"
    assert call["env"]["XDG_CONFIG_HOME"] != os.environ.get("XDG_CONFIG_HOME")

    pred = work.parent
    assert (pred / "opencode_raw.txt").read_text().strip()
    steps = (pred / "opencode_trace_steps.txt").read_text()
    assert "read" in steps and "task.md" in steps and "completed" in steps
    assert "tool calls: 3" in (pred / "opencode_trace_summary.txt").read_text()


def test_file_edits_allow_edit_but_not_shell_unless_configured(fake, tmp_path) -> None:
    work = _work(tmp_path)
    codex_harness.run_opencode_exec(work_dir=str(work), prompt="p", model="m", timeout=60, allow_file_edits=True)
    backend_config.configure_opencode_exec(allow_shell=True)
    codex_harness.run_opencode_exec(work_dir=str(work), prompt="p", model="m", timeout=60, allow_file_edits=True)
    codex_harness.run_opencode_exec(work_dir=str(work), prompt="p", model="m", timeout=60)

    edit, edit_shell, read_only = fake()
    assert ("edit", "*") in _allowed(edit["config"]) and ("shell", "*") not in _allowed(edit["config"])
    assert "You may modify files" in edit["stdin"]
    assert {("edit", "*"), ("shell", "*")} <= _allowed(edit_shell["config"])
    # Shell is a file-edit-mode permission only.
    assert not {("edit", "*"), ("shell", "*")} & _allowed(read_only["config"])


def test_data_dirs_and_image_dirs_are_the_only_outside_paths(fake, tmp_path) -> None:
    work = _work(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    image = tmp_path / "imgs" / "a.png"
    image.parent.mkdir()
    image.write_bytes(b"png")

    codex_harness.run_opencode_exec(
        work_dir=str(work), prompt="p", model="m", timeout=60, data_dirs=[str(data)], images=[str(image)]
    )

    (call,) = fake()
    outside = {res for action, res in _allowed(call["config"]) if action == "external_directory"}
    assert outside == {os.path.realpath(data) + "/*", os.path.realpath(image.parent) + "/*"}
    assert call["argv"][call["argv"].index("--file") + 1] == os.path.realpath(image)


def test_denied_data_dir_is_refused_before_launch(fake, tmp_path) -> None:
    work = _work(tmp_path)
    # Benchmark answer splits must never reach the agent.
    secret = tmp_path / "officeqa_split"
    secret.mkdir()

    with pytest.raises(ValueError, match="denied data directory"):
        codex_harness.run_opencode_exec(work_dir=str(work), prompt="p", model="m", timeout=60, data_dirs=[str(secret)])
    assert fake() == []


def test_empty_answer_is_retried_then_succeeds(fake, tmp_path, monkeypatch) -> None:
    empty = tmp_path / "empty.jsonl"
    lines = (_FIXTURES / "target_skill_read.jsonl").read_text().splitlines()
    empty.write_text("\n".join(line for line in lines if json.loads(line)["type"] != "text") + "\n")
    monkeypatch.setenv(
        "FAKE_OPENCODE_STDOUT", os.pathsep.join([str(empty), str(_FIXTURES / "target_skill_read.jsonl")])
    )
    monkeypatch.setattr(backend_config, "EXEC_EMPTY_RESPONSE_RETRIES", 1)

    response, _raw = codex_harness.run_opencode_exec(work_dir=str(_work(tmp_path)), prompt="p", model="m", timeout=60)

    assert "BANANA42" in response
    first, second = fake()
    assert "Previous execution returned an empty final response" in second["stdin"]


def test_error_event_raises_with_provider_message(fake, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_OPENCODE_STDOUT", str(_FIXTURES / "run_error_model.jsonl"))
    monkeypatch.setenv("FAKE_OPENCODE_EXIT", "1")

    with pytest.raises(RuntimeError, match="not supported when using Codex with a ChatGPT account"):
        codex_harness.run_opencode_exec(work_dir=str(_work(tmp_path)), prompt="p", model="m", timeout=60)


def test_target_dispatch_and_both_roles(fake, tmp_path, monkeypatch) -> None:
    import skillopt.model as model

    for name in ("OPTIMIZER_BACKEND", "TARGET_BACKEND"):
        monkeypatch.setattr(backend_config, name, getattr(backend_config, name))
        monkeypatch.setenv(name, getattr(backend_config, name))

    model.set_backend("opencode")

    assert (backend_config.get_optimizer_backend(), backend_config.get_target_backend()) == ("opencode_exec", "opencode_exec")
    assert backend_config.is_target_exec_backend()
    assert model.get_backend_name() == "opencode_exec"
    response, _raw = codex_harness.run_target_exec(work_dir=str(_work(tmp_path)), prompt="p", model="m", timeout=60)
    assert "BANANA42" in response


def _train_cfg(monkeypatch, *argv: str) -> dict:
    monkeypatch.setattr(
        sys, "argv", ["skillopt-train", "--config", str(_ROOT / "configs" / "searchqa" / "default.yaml"), *argv]
    )
    return train_script.load_config(train_script.parse_args())


def test_train_backend_opencode_sets_both_roles_and_models(monkeypatch) -> None:
    cfg = _train_cfg(monkeypatch, "--backend", "opencode")

    assert cfg["optimizer_backend"] == "opencode_exec"
    assert cfg["target_backend"] == "opencode_exec"
    assert cfg["optimizer_model"].startswith("openai/")
    assert cfg["target_model"].startswith("openai/")


def test_train_target_only_keeps_the_optimizer(monkeypatch) -> None:
    cfg = _train_cfg(
        monkeypatch, "--target_backend", "opencode_exec", "--target_model", "openai/gpt-6.1-sol", "--opencode_exec_allow_shell", "true"
    )

    assert cfg["optimizer_backend"] == "openai_chat"
    assert cfg["target_backend"] == "opencode_exec"
    assert cfg["target_model"] == "openai/gpt-6.1-sol"
    assert cfg["opencode_exec_allow_shell"] is True


def test_eval_only_backend_opencode_sets_both_roles(monkeypatch, tmp_path) -> None:
    import scripts.eval_only as eval_script

    skill = tmp_path / "skill.md"
    skill.write_text("# skill\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillopt-eval",
            "--config", str(_ROOT / "configs" / "searchqa" / "default.yaml"),
            "--skill", str(skill),
            "--backend", "opencode",
            "--opencode_exec_allow_shell", "false",
        ],
    )
    cfg = eval_script.load_config(eval_script.parse_args())

    assert (cfg["optimizer_backend"], cfg["target_backend"]) == ("opencode_exec", "opencode_exec")
    assert cfg["target_model"].startswith("openai/")
    assert cfg["opencode_exec_allow_shell"] is False


def test_target_refuses_opencode_1(fake, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_OPENCODE_VERSION", "opencode 1.2.27")

    with pytest.raises(RuntimeError, match=r"OpenCode 2\.x"):
        codex_harness.run_opencode_exec(work_dir=str(_work(tmp_path)), prompt="p", model="m", timeout=60)
    assert fake() == []


def test_target_missing_cli_is_an_actionable_error(tmp_path) -> None:
    backend_config.configure_opencode_exec(path=str(tmp_path / "no-such-opencode"))
    try:
        with pytest.raises(RuntimeError, match="OpenCode could not be (executed|started)"):
            codex_harness.run_opencode_exec(work_dir=str(_work(tmp_path)), prompt="p", model="m", timeout=60)
    finally:
        backend_config.configure_opencode_exec(path="opencode")

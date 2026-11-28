"""Offline tests for the AppWorld environment (no AppWorld server, no model calls)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillopt.envs.appworld import make_split
from skillopt.envs.appworld.adapter import AppWorldAdapter
from skillopt.envs.appworld.dataloader import AppWorldDataLoader
from skillopt.envs.appworld.episode import REACT_PROMPT_PATH, initial_messages, load_app_descriptions
from skillopt.envs.appworld.grading import grade_rollout
from skillopt.envs.appworld.rollout import (
    _conversation_for_reflection,
    _fail_reason,
    build_skill_suffix,
    run_batch,
)

SPLITS = Path(make_split.__file__).resolve().parent / "splits"


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "appworld"
    lists = {
        "train": ["aaa1111_1", "aaa1111_2"],
        "dev": ["bbb2222_1"],
        "test_normal": ["ccc3333_1", "ccc3333_2"],
        "test_challenge": ["ddd4444_1"],
    }
    (root / "data" / "datasets").mkdir(parents=True)
    for name, ids in lists.items():
        (root / "data" / "datasets" / f"{name}.txt").write_text("\n".join(ids) + "\n")
        for task_id in ids:
            task = root / "data" / "tasks" / task_id
            task.mkdir(parents=True)
            (task / "specs.json").write_text(json.dumps({"instruction": f"do {task_id}"}))
    return root


def _episode(**kw) -> SimpleNamespace:
    base = dict(tgc=False, stop_reason="no_tool_call", task_completed=True,
                verifier_components={"failed_requirements": ["assert answers match."]},
                reward=0.5, verifier_status="ok", messages=[{}, {}], dispatches=[{}, {}])
    base.update(kw)
    return SimpleNamespace(**base)


# ── split construction and loading ──────────────────────────────────────────

def test_make_split_full_layout_is_deterministic(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    counts = make_split.build(root, tmp_path / "a", "full")
    make_split.build(root, tmp_path / "b", "full")
    assert counts == {"train": 2, "val": 1, "test": 3}
    for split in ("train", "val", "test"):
        a = (tmp_path / "a" / split / "items.json").read_text()
        assert a == (tmp_path / "b" / split / "items.json").read_text()
    test = json.loads((tmp_path / "a" / "test" / "items.json").read_text())
    assert [t["source_split"] for t in test] == ["test_normal", "test_normal", "test_challenge"]
    assert test[0] == {"id": "ccc3333_1", "task_id": "ccc3333_1", "instruction": "do ccc3333_1",
                       "source_split": "test_normal", "scenario_id": "ccc3333", "task_type": "appworld"}


def test_make_split_rejects_a_task_in_two_splits(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    (root / "data" / "datasets" / "dev.txt").write_text("aaa1111_1\n")
    with pytest.raises(SystemExit, match="in both"):
        make_split.build(root, tmp_path / "out", "full")


def test_make_split_rejects_unknown_test_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_split.build(_fake_root(tmp_path), tmp_path / "out", "rest")


def test_scen1_lists_hold_one_task_per_scenario() -> None:
    for name, size in (("test_normal_scen1", 56), ("test_challenge_scen1", 139)):
        ids = (SPLITS / f"{name}.txt").read_text().split()
        assert len(ids) == size == len(set(ids))
        scenarios = [i.rsplit("_", 1)[0] for i in ids]
        assert len(set(scenarios)) == size


def test_dataloader_reads_items_and_sets_task_description(tmp_path: Path) -> None:
    make_split.build(_fake_root(tmp_path), tmp_path / "split", "full")
    items = AppWorldDataLoader(split_dir=str(tmp_path / "split")).load_split_items(
        str(tmp_path / "split" / "train"))
    assert [i["task_description"] for i in items] == ["do aaa1111_1", "do aaa1111_2"]


def test_dataloader_names_make_split_when_items_are_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make_split"):
        AppWorldDataLoader(split_dir=str(tmp_path)).load_split_items(str(tmp_path / "train"))


# ── registration ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module", ["scripts.train", "scripts.eval_only"])
def test_appworld_is_registered(module: str) -> None:
    mod = __import__(module, fromlist=["_register_builtins"])
    mod._register_builtins()
    assert mod._ENV_REGISTRY["appworld"] is AppWorldAdapter


# ── failure text and reflection conversation ────────────────────────────────

def test_checks_style_leads_with_the_failed_checks() -> None:
    text = _fail_reason(_episode(), "checks")
    assert text.startswith("Failed AppWorld checks: assert answers match.")
    assert "stop_reason" not in text
    assert "ended normally" in text


@pytest.mark.parametrize("stop, words", [("budget_exhausted", "tool-call budget"),
                                         ("truncated", "token limit"),
                                         ("context_overflow", "model context")])
def test_checks_style_names_a_stop_reason_that_explains_the_failure(stop: str, words: str) -> None:
    assert words in _fail_reason(_episode(stop_reason=stop, task_completed=False), "checks")


def test_checks_style_reports_a_missing_complete_task() -> None:
    text = _fail_reason(_episode(task_completed=False), "checks")
    assert "before apis.supervisor.complete_task succeeded" in text


def test_legacy_style_keeps_the_original_text() -> None:
    assert _fail_reason(_episode(task_completed=False), "legacy") == (
        "Episode ended with stop_reason=no_tool_call. "
        "apis.supervisor.complete_task was never called successfully. "
        "Failed AppWorld checks: assert answers match.")


def test_passed_task_has_no_failure_text() -> None:
    for style in ("checks", "legacy"):
        assert _fail_reason(_episode(tgc=True), style) == ""


def test_adapter_rejects_an_unknown_fail_reason_style() -> None:
    with pytest.raises(ValueError):
        AppWorldAdapter(fail_reason_style="verbose")


def test_conversation_pairs_tool_calls_with_their_results() -> None:
    call = {"id": "c1", "function": {"name": "execute_code", "arguments": json.dumps({"code": "print(1)"})}}
    episode = _episode(messages=[
        {"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
        {"role": "assistant", "content": "thinking", "tool_calls": [call]},
        {"role": "tool", "content": "1"},
        {"role": "assistant", "content": "Task complete."},
    ])
    result = {"stop_reason": "no_tool_call", "hard": 0, "soft": 0.5, "n_turns": 1,
              "fail_reason": _fail_reason(episode, "checks")}
    conv = _conversation_for_reflection({"instruction": "do it"}, episode, result, "checks")
    assert conv[0] == {"role": "user", "content": "Task: do it"}
    assert conv[1] == {"role": "assistant", "content": "thinking"}
    assert conv[2] == {"type": "tool_call", "cmd": "print(1)", "obs": "1"}
    assert conv[3] == {"role": "assistant", "content": "Task complete."}
    assert conv[-1]["role"] == "system" and conv[-1]["content"].startswith("Outcome: task_goal_completion=False")
    assert "stop_reason=" not in conv[-1]["content"]


# ── skill injection and rollout plumbing ────────────────────────────────────

def test_skill_suffix_is_empty_for_an_empty_skill() -> None:
    assert build_skill_suffix("") == build_skill_suffix("  \n") == ""
    assert build_skill_suffix("Rule A").endswith("Rule A\n")
    assert "## Skill Knowledge" in build_skill_suffix("Rule A")


def test_run_batch_rejects_an_unknown_fail_reason_style(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_batch(items=[], skill_content="", out_root=str(tmp_path), appworld_root="", appworld_venv="",
                  workers=1, port_base=0, port_span=1, max_steps=1, max_completion_tokens=1,
                  episode_token_wall=0, request_timeout=1, call_timeout=None, fail_reason_style="x")


def test_rollout_reuses_a_finished_batch_of_the_same_tasks(tmp_path: Path) -> None:
    rows = [{"id": "t1", "hard": 1}, {"id": "t2", "hard": 0}]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    adapter = AppWorldAdapter(appworld_venv="/nonexistent")
    got = adapter.rollout([{"id": "t2"}, {"id": "t1"}], "skill", str(tmp_path))
    assert got == rows


# ── grading and prompt ──────────────────────────────────────────────────────

def _audit(passes, failures, success):
    return {"appworld": {"tracker": {"passes": passes, "failures": failures, "success": success,
                                     "num_tests": len(passes) + len(failures)}}}


def test_grading_ignores_tests_a_do_nothing_agent_passes() -> None:
    free = {"requirement": "assert no model changes", "label": "no_op_pass"}
    earned = {"requirement": "assert song liked", "label": "no_op_fail"}
    missed = {"requirement": "assert answers match", "label": "no_op_fail"}
    out = grade_rollout(task_id="abc_1", audit_data=_audit([free, earned], [missed], False))
    assert (out.score, out.passed, out.status) == (0.5, False, "scored")
    assert out.components["failed_requirements"] == ["assert answers match"]
    nothing = grade_rollout(task_id="abc_1", audit_data=_audit([free], [earned, missed], False))
    assert nothing.score == 0.0
    assert grade_rollout(task_id="abc_1", audit_data=_audit([free, earned, missed], [], True)).passed is True


def test_grading_without_a_tracker_is_an_infrastructure_failure() -> None:
    out = grade_rollout(task_id="abc_1", audit_data=None)
    assert (out.score, out.passed, out.status) == (None, None, "grader_failure")


def test_initial_messages_fill_the_official_react_prompt() -> None:
    info = {"instruction": "Like my top song.", "datetime": "2023-05-18T12:00:00",
            "supervisor": {"first_name": "Ann", "last_name": "Lee", "email": "ann@x.com", "phone_number": "555"}}
    system, user = initial_messages(info, REACT_PROMPT_PATH.read_text(), budget=50, app_descriptions="[]")
    assert system["role"] == "system" and user["role"] == "user"
    assert user["content"].startswith("Like my top song.\n\n")
    assert "{{" not in user["content"] and "Ann" in user["content"]
    assert user["content"].endswith("You have at most 50 tool calls.")


def test_a_placeholder_with_no_value_raises() -> None:
    info = {"instruction": "x", "datetime": "d",
            "supervisor": {"first_name": "A", "last_name": "B", "email": "e", "phone_number": "p"}}
    with pytest.raises(ValueError, match="new_field"):
        initial_messages(info, "Hi {{ new_field }}", 50, app_descriptions="[]")


def test_initial_messages_fill_the_app_descriptions() -> None:
    info = {"instruction": "x", "datetime": "d",
            "supervisor": {"first_name": "A", "last_name": "B", "email": "e", "phone_number": "p"}}
    apps = json.dumps([{"name": "spotify", "description": "A music app."}], indent=1)
    _, user = initial_messages(info, REACT_PROMPT_PATH.read_text(), 50, app_descriptions=apps)
    assert "{{" not in user["content"]
    assert 'Output:\n```\n[\n {\n  "name": "spotify"' in user["content"]


def _fake_venv(tmp_path: Path, stdout: str, rc: int = 0) -> Path:
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text(f"#!/bin/sh\ncat <<'EOF'\n{stdout}\nEOF\nexit {rc}\n")
    (bin_dir / "python").chmod(0o755)
    return tmp_path / "venv"


def test_load_app_descriptions_reads_the_appworld_venv(tmp_path: Path) -> None:
    text = json.dumps([{"name": "venmo", "description": "Pay people."}], indent=1)
    assert load_app_descriptions(_fake_venv(tmp_path, text)) == text


@pytest.mark.parametrize("stdout, rc", [("[]", 0), ("oops", 1)])
def test_load_app_descriptions_fails_loudly(tmp_path: Path, stdout: str, rc: int) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        load_app_descriptions(_fake_venv(tmp_path, stdout, rc))


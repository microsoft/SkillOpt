"""`skillopt-eval --split all`, the default, must work for every adapter.

eval_only joined the train, valid_seen and valid_unseen envs with `+`, which
only works when build_eval_env returns a list. ALFWorld returns an
ALFWorldBatchRun, so the default command crashed with a TypeError before any
rollout.
"""

from __future__ import annotations

import json
import os
import sys
from unittest import mock

import pytest

import scripts.eval_only as eval_cli
from skillopt.envs.alfworld.adapter import ALFWorldAdapter


@pytest.fixture
def isolated_model_state(monkeypatch):
    # main() configures every model backend, which changes module state and
    # os.environ shared with other tests. The rollout is stubbed, so skip it.
    for name in (
        "configure_azure_openai",
        "configure_claude_code_exec",
        "configure_codex_exec_from_config",
        "configure_copilot_chat",
        "configure_copilot_exec",
        "configure_cursor_exec",
        "configure_minimax_chat",
        "configure_openai_compatible",
        "configure_qwen_chat",
        "set_optimizer_backend",
        "set_optimizer_deployment",
        "set_reasoning_effort",
        "set_target_backend",
        "set_target_deployment",
    ):
        monkeypatch.setattr(eval_cli, name, mock.Mock())
    with mock.patch.dict(os.environ):
        yield


def _run_eval(tmp_path, config: str, extra_args=()) -> dict:
    skill = tmp_path / "skill.md"
    skill.write_text("# Skill\n", encoding="utf-8")
    out_root = tmp_path / "out"
    argv = [
        "skillopt-eval",
        "--config",
        config,
        "--skill",
        str(skill),
        "--out_root",
        str(out_root),
        *extra_args,
    ]  # no --split: the default is "all"
    with mock.patch.object(sys, "argv", argv):
        eval_cli.main()
    return json.loads((out_root / "eval_summary.json").read_text(encoding="utf-8"))


def test_split_all_rolls_out_each_alfworld_split(tmp_path, isolated_model_state):
    split_dir = tmp_path / "split"
    for split, sub in (("train", "train"), ("val", "valid_seen"), ("test", "valid_unseen")):
        (split_dir / split).mkdir(parents=True)
        (split_dir / split / "items.json").write_text(
            json.dumps([{"id": f"{split}_0", "gamefile": f"/data/json_2.1.1/{sub}/task_{split}/game.tw-pddl"}])
        )
    calls = []

    def fake_rollout(self, env_manager, skill_content, out_dir, **kwargs):
        ids = [item["id"] for item in env_manager]
        calls.append((os.path.relpath(out_dir, tmp_path / "out"), ids))
        return [{"id": i, "hard": 1, "soft": 1.0} for i in ids]

    with mock.patch.object(ALFWorldAdapter, "rollout", fake_rollout):
        summary = _run_eval(tmp_path, "configs/alfworld/default.yaml", ["--split_dir", str(split_dir)])

    # ALFWorld's rollout resumes from <out_dir>/results.jsonl, so each split
    # needs its own folder or the later splits would get the train results.
    assert calls == [
        ("train", ["train_0"]),
        ("valid_seen", ["val_0"]),
        ("valid_unseen", ["test_0"]),
    ]
    assert summary["split"] == "all"
    assert summary["n_items"] == 3


def test_split_all_still_uses_one_rollout_for_list_envs(tmp_path, isolated_model_state):
    rollouts = []

    class ListAdapter:
        def setup(self, cfg):
            pass

        def build_eval_env(self, env_num, split, seed, **kwargs):
            return [{"id": f"{split}_0"}]

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            rollouts.append((out_dir, [item["id"] for item in env_manager]))
            return [{"id": item["id"], "hard": 1, "soft": 1.0} for item in env_manager]

    with mock.patch.object(eval_cli, "get_adapter", lambda cfg: ListAdapter()):
        summary = _run_eval(tmp_path, "configs/searchqa/default.yaml")

    assert rollouts == [(str(tmp_path / "out"), ["train_0", "valid_seen_0", "valid_unseen_0"])]
    assert summary["n_items"] == 3

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skillopt.envs.alfworld import rollout
from skillopt.envs.alfworld.dataloader import ALFWorldDataLoader
from skillopt.envs.task_output import task_output_segment
from skillopt.gradient.reflect import fmt_minibatch_trajectories
from skillopt.optimizer.slow_update import _read_trajectory


class _DummyEnvManager:
    def __init__(self, n_envs: int = 1, won: bool = True):
        self.n_envs = n_envs
        self.won = won
        self.reset_called = False
        self.step_called = False

    def reset(self, _kwargs):
        self.reset_called = True
        return {
            "text": ["You are in a kitchen."] * self.n_envs,
            "anchor": ["Your task is to: find a mug"] * self.n_envs,
        }, [{}] * self.n_envs

    def step(self, actions):
        self.step_called = True
        return (
            {"anchor": ["You picked up the mug."] * self.n_envs},
            [1.0] * self.n_envs,
            [True] * self.n_envs,
            [{"won": self.won}] * self.n_envs,
        )


def test_alfworld_rollout_maps_unsafe_task_ids_and_confines_write(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rollout,
        "chat_target",
        lambda **_kwargs: ("<think>look around</think><action>take mug</action>", None),
    )

    env_mgr = _DummyEnvManager(n_envs=1, won=True)
    out_root = tmp_path / "out"

    results = rollout.run_alfworld_batch(
        env_mgr,
        skill_content="",
        out_root=str(out_root),
        result_ids=["../../escape"],
    )

    assert len(results) == 1
    assert results[0]["id"] == "../../escape"
    assert not (tmp_path / "escape").exists()

    segment = task_output_segment("../../escape", map_unsafe=True)
    assert segment.startswith("_mapped-")
    conv_path = out_root / "predictions" / segment / "conversation.json"
    assert conv_path.is_file()


def test_alfworld_rollout_writes_confined_conversation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rollout,
        "chat_target",
        lambda **_kwargs: ("<think>look around</think><action>take mug</action>", None),
    )

    env_mgr = _DummyEnvManager(n_envs=1, won=True)
    out_root = tmp_path / "out"

    results = rollout.run_alfworld_batch(
        env_mgr,
        skill_content="Be observant.",
        out_root=str(out_root),
        result_ids=["task_001"],
    )

    assert len(results) == 1
    assert results[0]["id"] == "task_001"
    assert results[0]["hard"] == 1
    assert results[0]["soft"] == 1.0

    conv_path = out_root / "predictions" / "task_001" / "conversation.json"
    assert conv_path.is_file()

    with open(conv_path, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["action"] == "take mug"
    assert data[0]["reasoning"] == "look around"
    assert data[0]["reward"] == 1.0
    assert data[0]["done"] is True


def test_alfworld_rollout_writes_confined_conversation_default_id(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rollout,
        "chat_target",
        lambda **_kwargs: ("<think>look around</think><action>take mug</action>", None),
    )

    env_mgr = _DummyEnvManager(n_envs=1, won=True)
    out_root = tmp_path / "out"

    results = rollout.run_alfworld_batch(
        env_mgr,
        skill_content="",
        out_root=str(out_root),
        result_ids=None,
    )

    assert len(results) == 1
    assert results[0]["id"] == "env_000"

    conv_path = out_root / "predictions" / "env_000" / "conversation.json"
    assert conv_path.is_file()


def test_alfworld_rollout_rejects_symlink_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rollout,
        "chat_target",
        lambda **_kwargs: ("<action>look</action>", None),
    )

    predictions = tmp_path / "out" / "predictions"
    predictions.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()

    try:
        os.symlink(outside, predictions / "task_001", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are not available on this host")

    with pytest.raises(ValueError, match="escapes predictions"):
        rollout.run_alfworld_batch(
            _DummyEnvManager(n_envs=1, won=True),
            skill_content="",
            out_root=str(tmp_path / "out"),
            result_ids=["task_001"],
        )


def test_alfworld_rollout_without_out_root_does_not_write(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rollout,
        "chat_target",
        lambda **_kwargs: ("<think>think</think><action>take mug</action>", None),
    )

    env_mgr = _DummyEnvManager(n_envs=1, won=True)
    results = rollout.run_alfworld_batch(
        env_mgr,
        skill_content="",
        out_root="",
        result_ids=["task_001"],
    )

    assert len(results) == 1
    assert results[0]["id"] == "task_001"
    assert not (tmp_path / "predictions").exists()


def test_alfworld_rollout_with_shipped_split_ids_is_readable(monkeypatch, tmp_path):
    items_path = Path(__file__).resolve().parent.parent / "data" / "alfworld_path_split" / "train" / "items.json"
    with open(items_path, encoding="utf-8") as f:
        shipped_items = json.load(f)[:2]

    meta = ALFWorldDataLoader._metadata_for_items(shipped_items, split="train", phase="train")
    result_ids = meta["result_ids"]
    assert all(":" in rid for rid in result_ids)

    monkeypatch.setattr(
        rollout,
        "chat_target",
        lambda **_kwargs: ("<think>search</think><action>take clock</action>", None),
    )

    env_mgr = _DummyEnvManager(n_envs=len(shipped_items), won=True)
    out_root = tmp_path / "out"

    results = rollout.run_alfworld_batch(
        env_mgr,
        skill_content="Be observant.",
        out_root=str(out_root),
        result_ids=result_ids,
    )

    assert len(results) == len(shipped_items)
    for rid in result_ids:
        traj = _read_trajectory(str(out_root), rid)
        assert "take clock" in traj
        assert "(trajectory not available)" not in traj

    formatted = fmt_minibatch_trajectories(results, str(out_root / "predictions"))
    assert "take clock" in formatted

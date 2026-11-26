"""optimizer.longitudinal_traj_char_budget and optimizer.longitudinal_parse_retries."""
from __future__ import annotations

import sys

import pytest

import scripts.train as train_script
from skillopt.optimizer import meta_skill, slow_update
from skillopt.optimizer.slow_update import _clip_middle, _fair_caps, format_comparison_text


def _pair(task_id: str, category: str, traj_len: int) -> dict:
    side = {"hard": 0, "soft": 0.0, "predicted_answer": "x"}
    return {"id": task_id, "task": f"task {task_id}", "category": category,
            "prev": dict(side), "curr": dict(side),
            "prev_trajectory": "p" * traj_len, "curr_trajectory": "c" * traj_len}


# ── _fair_caps / _clip_middle ───────────────────────────────────────────────

def test_fair_caps_keeps_everything_under_budget() -> None:
    assert _fair_caps([10, 20, 30], 100) == 30


def test_fair_caps_clips_only_the_longest() -> None:
    cap = _fair_caps([10, 20, 1000], 100)
    assert cap == 70
    assert sum(min(n, cap) for n in [10, 20, 1000]) <= 100


def test_fair_caps_splits_evenly_when_all_are_long() -> None:
    cap = _fair_caps([500, 500, 500], 300)
    assert cap == 100


def test_fair_caps_empty() -> None:
    assert _fair_caps([], 100) == 0


def test_clip_middle_keeps_head_and_tail_within_limit() -> None:
    text = "H" * 500 + "M" * 1000 + "T" * 500
    out = _clip_middle(text, 300)
    assert len(out) <= 300
    assert out.startswith("H") and out.endswith("T")
    assert "chars omitted" in out


def test_clip_middle_leaves_short_text_and_zero_limit_alone() -> None:
    assert _clip_middle("short", 100) == "short"
    assert _clip_middle("x" * 1000, 0) == "x" * 1000


# ── format_comparison_text ──────────────────────────────────────────────────

def test_budget_zero_matches_no_budget_argument() -> None:
    pairs = [_pair("a", "regressed", 5000), _pair("b", "persistent_fail", 7000)]
    assert format_comparison_text(pairs) == format_comparison_text(pairs, traj_char_budget=0)
    assert "chars omitted" not in format_comparison_text(pairs)


def test_budget_above_total_changes_nothing() -> None:
    pairs = [_pair("a", "regressed", 100)]
    assert format_comparison_text(pairs, traj_char_budget=10_000) == format_comparison_text(pairs)


def test_budget_clips_shown_trajectories_to_fit() -> None:
    pairs = [_pair("a", "regressed", 50_000), _pair("b", "improved", 50_000),
             _pair("c", "stable_success", 50_000)]  # stable successes show no trajectory
    full = format_comparison_text(pairs)
    clipped = format_comparison_text(pairs, traj_char_budget=20_000)
    assert len(full) - len(clipped) >= 200_000 - 20_000 - 1_000
    assert clipped.count("chars omitted") == 4


# ── parse retries ───────────────────────────────────────────────────────────

class _FakeChat:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, **_kw):
        self.calls += 1
        return self.answers.pop(0), {}


@pytest.mark.parametrize("retries, answers, expect_calls, expect_ok", [
    (0, ["not json", '{"slow_update_content": "G"}'], 1, False),
    (2, ["not json", '{"slow_update_content": "G"}'], 2, True),
    (2, ["no", "no", "no"], 3, False),
])
def test_slow_update_parse_retries(monkeypatch, retries, answers, expect_calls, expect_ok) -> None:
    fake = _FakeChat(answers)
    monkeypatch.setattr(slow_update, "chat_optimizer", fake)
    out = slow_update.run_slow_update("skill", [], [], [], comparison_pairs=[],
                                      system_prompt="sys", parse_retries=retries)
    assert fake.calls == expect_calls
    assert (out is not None) is expect_ok


def test_meta_skill_parse_retries(monkeypatch) -> None:
    fake = _FakeChat(["", '{"meta_skill_content": "M"}'])
    monkeypatch.setattr(meta_skill, "chat_optimizer", fake)
    out = meta_skill.run_meta_skill("a", "b", [], system_prompt="sys", parse_retries=1)
    assert fake.calls == 2 and out["meta_skill_content"] == "M"


def test_optimizer_call_failure_is_not_retried(monkeypatch) -> None:
    calls = []

    def boom(**_kw):
        calls.append(1)
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(slow_update, "chat_optimizer", boom)
    out = slow_update.run_slow_update("skill", [], [], [], comparison_pairs=[],
                                      system_prompt="sys", parse_retries=3)
    assert out is None and len(calls) == 1


# ── config / CLI plumbing ───────────────────────────────────────────────────

def test_defaults_and_cli_flags_reach_the_trainer_config(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "configs/_base_/default.yaml"])
    cfg = train_script.load_config(train_script.parse_args())
    assert cfg["longitudinal_traj_char_budget"] == 0 and cfg["longitudinal_parse_retries"] == 0
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "configs/_base_/default.yaml",
                                      "--longitudinal_traj_char_budget", "330000",
                                      "--longitudinal_parse_retries", "2"])
    cfg = train_script.load_config(train_script.parse_args())
    assert cfg["longitudinal_traj_char_budget"] == 330000 and cfg["longitudinal_parse_retries"] == 2

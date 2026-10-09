"""The rubric judge's 0..1 score contract is enforced before scores reach the gate.

The judge prompt asks for a 0..1 score, but ``CliBackend.judge`` accepted any
float, including ``NaN``/``Infinity`` (which ``json.loads`` parses). On the
default mixed gate metric a single out-of-range soft score could carry a
candidate whose held-out accuracy actually regressed.
"""
from __future__ import annotations

import pytest

from skillopt_sleep.backend import CliBackend
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.types import TaskRecord


class _ScriptedJudgeBackend(CliBackend):
    """Only the provider call is scripted; parsing and gating are shipped code."""

    name = "scripted"

    def __init__(self, replies=None):
        super().__init__()
        self.replies = dict(replies or {})

    def _call(self, prompt, *, max_tokens=1024):
        if prompt.startswith("Score how well"):
            candidate = "CAND" in prompt.split("# Response", 1)[1]
            for rubric, (baseline_reply, candidate_reply) in self.replies.items():
                if rubric in prompt:
                    return candidate_reply if candidate else baseline_reply
            return '{"score": 0.0}'
        if prompt.startswith("You are SkillOpt's optimizer"):
            return '[{"op":"add","content":"RULE terse"}]'
        return "CAND answer" if "RULE terse" in prompt else "BASE answer"


def _rubric_task(task_id, split, rubric):
    return TaskRecord(
        id=task_id,
        project="/project",
        intent=f"do {task_id}",
        reference_kind="rubric",
        reference=rubric,
        split=split,
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"score": 0.85, "reason": "good"}', (1.0, 0.85, "good")),
        ('{"score": 0.5}', (0.0, 0.5, "")),
        ('{"score": 1}', (1.0, 1.0, "")),
        ('{"score": 0}', (0.0, 0.0, "")),
        ('{"score": "0.9"}', (1.0, 0.9, "")),
        ('{"score": 2.0}', (0.0, 0.0, "judge-score-out-of-range: 2.0")),
        ('{"score": 8}', (0.0, 0.0, "judge-score-out-of-range: 8")),
        ('{"score": -0.1}', (0.0, 0.0, "judge-score-out-of-range: -0.1")),
        ('{"score": Infinity}', (0.0, 0.0, "judge-score-out-of-range: inf")),
        ('{"score": NaN}', (0.0, 0.0, "judge-score-out-of-range: nan")),
        ('{"score": true}', (0.0, 0.0, "judge-parse-failed")),
        ('{"score": "high"}', (0.0, 0.0, "judge-parse-failed")),
        ("no json here", (0.0, 0.0, "judge-parse-failed")),
    ],
)
def test_judge_accepts_only_finite_scores_in_the_unit_interval(reply, expected) -> None:
    backend = _ScriptedJudgeBackend({"RUBRIC": (reply, reply)})
    task = _rubric_task("t", "val", "RUBRIC")
    assert backend.judge(task, "BASE answer") == expected


def test_out_of_range_judge_score_cannot_carry_a_regressed_candidate() -> None:
    """v1 regresses 1.0 -> 0.0; v2's judge answers 2.0 for the candidate.

    Before the contract was enforced, the mixed metric read 0.625 -> 0.750 and
    the gate accepted a candidate with flat hard accuracy and one regression.
    """
    backend = _ScriptedJudgeBackend({
        "RUBRIC-V1": ('{"score": 1.0}', '{"score": 0.0}'),
        "RUBRIC-V2": ('{"score": 0.5}', '{"score": 2.0}'),
    })
    tasks = [
        _rubric_task("t1", "train", "RUBRIC-T"),
        _rubric_task("v1", "val", "RUBRIC-V1"),
        _rubric_task("v2", "val", "RUBRIC-V2"),
    ]
    result = consolidate(backend, tasks, "skill", "", gate_metric="mixed", evolve_memory=False)
    assert result.accepted is False
    assert result.gate_action.startswith("reject")
    skill_trial = next(t for t in result.gate_trials if t["target"] == "skill")
    assert all(d["candidate_score"] <= 1.0 for d in skill_trial["task_deltas"])

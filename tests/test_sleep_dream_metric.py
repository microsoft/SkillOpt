"""Gate-aligned scoring for SkillOpt-Sleep contrastive dream rollouts."""
from __future__ import annotations

from unittest import mock

from skillopt_sleep.backend import Backend, MockBackend
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.rollout import RolloutSet, contrastive_reflect
from skillopt_sleep.types import ReplayResult, TaskRecord


class RecordingBackend(Backend):
    name = "recording"

    def __init__(self) -> None:
        super().__init__()
        self.prompts = []

    def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        self.prompts.append(prompt)
        return '[{"op":"add","content":"prefer the stronger attempt"}]'


def _soft_spread() -> RolloutSet:
    task = TaskRecord(
        id="t1", project="/p", intent="produce the best answer",
        source_sessions=["s1", "s2"],
    )
    return RolloutSet(
        task=task,
        attempts=[
            ReplayResult(id="t1", hard=1.0, soft=0.2, response="weak partial answer"),
            ReplayResult(id="t1", hard=1.0, soft=0.9, response="strong complete answer"),
        ],
    )


def test_hard_metric_preserves_no_contrast_for_equal_hard_scores():
    backend = RecordingBackend()

    edits = contrastive_reflect(backend, [_soft_spread()], "skill", "")

    assert edits == []
    assert backend.prompts == []


def test_soft_metric_learns_from_partial_credit_spread():
    backend = RecordingBackend()

    edits = contrastive_reflect(
        backend,
        [_soft_spread()],
        "skill",
        "",
        gate_metric="soft",
    )

    assert len(edits) == 1
    assert "strong complete answer" in backend.prompts[0]
    assert "weak partial answer" in backend.prompts[0]
    assert "soft score 0.900" in backend.prompts[0]
    assert "soft score 0.200" in backend.prompts[0]
    assert edits[0].source_task_ids == ["t1"]
    assert edits[0].source_session_ids == ["s1", "s2"]


def test_contrastive_edit_provenance_uses_only_informative_tasks():
    backend = RecordingBackend()
    informative = _soft_spread()
    uninformative_task = TaskRecord(
        id="t2", project="/p", intent="not included in the prompt",
        source_sessions=["s3"],
    )
    uninformative = RolloutSet(
        task=uninformative_task,
        attempts=[
            ReplayResult(id="t2", hard=1.0, soft=0.5),
            ReplayResult(id="t2", hard=1.0, soft=0.5),
        ],
    )

    edits = contrastive_reflect(
        backend, [informative, uninformative], "skill", "", gate_metric="soft"
    )

    assert len(edits) == 1
    assert edits[0].source_task_ids == ["t1"]
    assert edits[0].source_session_ids == ["s1", "s2"]
    assert "not included in the prompt" not in backend.prompts[0]


def test_mixed_metric_uses_the_configured_weight():
    backend = RecordingBackend()

    edits = contrastive_reflect(
        backend,
        [_soft_spread()],
        "skill",
        "",
        gate_metric="mixed",
        gate_mixed_weight=0.25,
    )

    assert len(edits) == 1
    assert "mixed score 0.975" in backend.prompts[0]
    assert "mixed score 0.800" in backend.prompts[0]


def test_consolidate_passes_gate_objective_to_dream_reflection():
    tasks = [
        TaskRecord(id="train", project="/p", intent="train", split="train"),
        TaskRecord(id="val", project="/p", intent="validate", split="val"),
    ]
    rollout = RolloutSet(
        task=tasks[0],
        attempts=[ReplayResult(id="train", hard=0.0, soft=0.1)],
    )

    with mock.patch("skillopt_sleep.rollout.multi_rollout", return_value=rollout), mock.patch(
        "skillopt_sleep.rollout.contrastive_reflect", return_value=[]
    ) as reflect:
        consolidate(
            MockBackend(),
            tasks,
            "# skill\n",
            "",
            rollouts_k=2,
            gate_metric="soft",
            gate_mixed_weight=0.7,
            evolve_memory=False,
        )

    assert reflect.call_args.kwargs["gate_metric"] == "soft"
    assert reflect.call_args.kwargs["gate_mixed_weight"] == 0.7


def test_single_shot_edits_record_task_and_session_provenance():
    from skillopt_sleep.types import EditRecord

    train = TaskRecord(
        id="train-1", project="/p", intent="train", split="train",
        source_sessions=["session-a", "session-shared"],
    )
    val = TaskRecord(
        id="val-1", project="/p", intent="validate", split="val",
        source_sessions=["session-heldout"],
    )

    class EditBackend(MockBackend):
        def reflect(self, failures, successes, skill, memory, **kwargs):
            return [EditRecord(target="skill", op="add", content="new rule")]

    def replay(_backend, tasks, _skill, _memory):
        return [(task, ReplayResult(id=task.id, hard=0.0)) for task in tasks]

    with mock.patch("skillopt_sleep.consolidate.replay_batch", side_effect=replay):
        result = consolidate(
            EditBackend(), [train, val], "# skill\n", "",
            evolve_memory=False,
        )

    edits = result.applied_edits + result.rejected_edits
    assert len(edits) == 1
    assert edits[0].source_task_ids == ["train-1"]
    assert edits[0].source_session_ids == ["session-a", "session-shared"]

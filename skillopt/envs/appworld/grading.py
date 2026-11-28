"""Turn AppWorld's own evaluation of one episode into a score.

AppWorld grades deterministically: ``/evaluate`` diffs the task's start and end databases and runs
the task's own unit tests. No rubric and no LLM judge is involved.

Two numbers come out of it:

* ``passed`` is AppWorld's official task goal completion (TGC): every test passes. Reported
  success rates use this.
* ``score`` is the fraction of the task's tests passed, counting only the tests a do-nothing
  agent would fail. AppWorld labels each test by how an empty agent (one that calls nothing and
  changes nothing) fares on it:

  - ``no_op_fail``: the empty agent fails it, so passing it means the agent did the work;
  - ``no_op_pass``: the empty agent passes it anyway. These are almost all side-effect guards
    ("assert no model changes"), which hold trivially when nothing was done.

  Counting the second kind would give partial credit for doing nothing, so a do-nothing episode
  scores 0.0 here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GradeOutcome:
    """Result of grading one episode. ``score``/``passed`` are None when it could not be graded."""

    score: float | None
    passed: bool | None
    status: str
    components: dict[str, Any]
    error: str | None = None


def _tracker_of(audit_data: dict[str, Any] | None) -> dict[str, Any]:
    appworld = (audit_data or {}).get("appworld") or {}
    tracker = appworld.get("tracker")
    if not isinstance(tracker, dict):
        raise ValueError("audit_data['appworld']['tracker'] is missing; run /evaluate before grading")
    return tracker


def grade_rollout(*, task_id: str, audit_data: dict[str, Any] | None) -> GradeOutcome:
    """Score one episode from the ``/evaluate`` tracker the episode collected.

    ``audit_data["appworld"]`` holds ``tracker`` (the ``/evaluate`` output), ``task_completed`` and
    ``interactions``.
    """
    try:
        tracker = _tracker_of(audit_data)
    except Exception as exc:  # noqa: BLE001 - an absent tracker is an infrastructure fault
        return GradeOutcome(
            score=None,
            passed=None,
            status="grader_failure",
            components={"benchmark": "appworld", "task_id": task_id},
            error=f"{type(exc).__name__}: {exc}",
        )

    passes = tracker.get("passes") or []
    failures = tracker.get("failures") or []
    num_tests = tracker.get("num_tests")
    if not isinstance(num_tests, int) or num_tests <= 0:
        num_tests = len(passes) + len(failures)

    def _is_free(entry: Any) -> bool:
        return isinstance(entry, dict) and entry.get("label") == "no_op_pass"

    earned_passes = [entry for entry in passes if not _is_free(entry)]
    free_passes = [entry for entry in passes if _is_free(entry)]
    free_failures = [entry for entry in failures if _is_free(entry)]
    # Denominator: only the tests an empty agent would fail.
    scored_tests = num_tests - len(free_passes) - len(free_failures)

    if scored_tests <= 0:
        # Nothing here separates doing the task from doing nothing. That is a property of the
        # task, not an agent failure, so it is not scored as 0.
        return GradeOutcome(
            score=None,
            passed=None,
            status="grader_failure",
            components={
                "benchmark": "appworld",
                "task_id": task_id,
                "num_tests": num_tests,
                "no_op_pass_tests": len(free_passes) + len(free_failures),
            },
            error="every AppWorld test for this task passes for a do-nothing agent",
        )

    pass_count = len(earned_passes)
    score = round(pass_count / scored_tests, 6)
    official_success = bool(tracker.get("success"))
    appworld_block = (audit_data or {}).get("appworld") or {}

    return GradeOutcome(
        score=score,
        passed=official_success,
        status="scored",
        components={
            "benchmark": "appworld",
            "task_id": task_id,
            "grader": "appworld.evaluator (deterministic unit tests)",
            "reward_definition": "earned_passes / tests_a_do_nothing_agent_would_fail",
            "pass_count": pass_count,
            "fail_count": len(failures),
            "num_tests": num_tests,
            "scored_tests": scored_tests,
            "no_op_pass_tests": len(free_passes) + len(free_failures),
            "raw_pass_ratio": round(len(passes) / num_tests, 6) if num_tests else None,
            "task_goal_completion": official_success,
            "difficulty": tracker.get("difficulty"),
            "scenario_id": str(task_id).rsplit("_", 1)[0],
            "task_completed_signal": appworld_block.get("task_completed"),
            "interactions": appworld_block.get("interactions"),
            "failed_requirements": [
                str(failure.get("requirement")) for failure in failures if isinstance(failure, dict)
            ][:20],
            "no_op_labels": sorted(
                {
                    str(entry.get("label"))
                    for entry in list(passes) + list(failures)
                    if isinstance(entry, dict) and entry.get("label")
                }
            ),
        },
        error=None,
    )


__all__ = ["GradeOutcome", "grade_rollout"]

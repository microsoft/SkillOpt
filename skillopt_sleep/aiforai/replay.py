"""Mock replay and gate utilities for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from skillopt_sleep.aiforai.types import AiforaiReplayResult, AiforaiTaskRecord


@dataclass(slots=True)
class AiforaiScoreSummary:
    aggregate_hard: float
    aggregate_soft: float
    by_source: dict[str, float] = field(default_factory=dict)
    by_family: dict[str, float] = field(default_factory=dict)
    results: list[AiforaiReplayResult] = field(default_factory=list)


@dataclass(slots=True)
class AiforaiGateDecision:
    accepted: bool
    action: str
    reason: str


def evaluate_tasks(
    tasks: list[AiforaiTaskRecord],
    skill: str,
) -> AiforaiScoreSummary:
    results: list[AiforaiReplayResult] = []
    for task in tasks:
        hard, soft, missing = _score_task(task, skill)
        results.append(
            AiforaiReplayResult(
                task_id=task.id,
                source_agent=task.source_agent,
                task_family=task.task_family,
                hard=hard,
                soft=soft,
                response=skill,
                fail_reason=f"missing: {', '.join(missing)}" if missing else "",
            )
        )
    return _summarize(results)


def propose_mock_rules(
    tasks: list[AiforaiTaskRecord],
    skill: str,
) -> list[str]:
    lower_skill = skill.lower()
    proposed: list[str] = []
    seen: set[str] = set()

    for task in tasks:
        missing = _missing_required_checks(task, lower_skill)
        if not missing:
            continue
        rule = f"For {task.task_family} tasks, explicitly include: {', '.join(missing)}."
        key = rule.lower()
        if key in seen:
            continue
        seen.add(key)
        proposed.append(rule)

    return proposed


def gate_candidate(
    baseline: AiforaiScoreSummary,
    candidate: AiforaiScoreSummary,
) -> AiforaiGateDecision:
    if candidate.aggregate_hard > baseline.aggregate_hard:
        return AiforaiGateDecision(
            accepted=True,
            action="accept",
            reason="candidate aggregate hard score improved",
        )
    return AiforaiGateDecision(
        accepted=False,
        action="reject",
        reason="candidate did not improve aggregate hard score",
    )


def _score_task(task: AiforaiTaskRecord, skill: str) -> tuple[float, float, list[str]]:
    required = _required_contains_checks(task)
    if not required:
        return 0.0, 0.0, ["no local checks"]

    lower_skill = skill.lower()
    missing = [item for item in required if item.lower() not in lower_skill]
    present = len(required) - len(missing)
    soft = present / len(required)
    hard = 1.0 if not missing else 0.0
    return hard, soft, missing


def _required_contains_checks(task: AiforaiTaskRecord) -> list[str]:
    checks = task.judge.get("checks", []) if task.judge else []
    required: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("op") != "contains":
            continue
        arg = str(check.get("arg", "")).strip()
        if arg:
            required.append(arg)
    return required


def _missing_required_checks(task: AiforaiTaskRecord, lower_skill: str) -> list[str]:
    return [item for item in _required_contains_checks(task) if item.lower() not in lower_skill]


def _summarize(results: list[AiforaiReplayResult]) -> AiforaiScoreSummary:
    if not results:
        return AiforaiScoreSummary(aggregate_hard=0.0, aggregate_soft=0.0)

    aggregate_hard = sum(result.hard for result in results) / len(results)
    aggregate_soft = sum(result.soft for result in results) / len(results)
    return AiforaiScoreSummary(
        aggregate_hard=aggregate_hard,
        aggregate_soft=aggregate_soft,
        by_source=_group_mean(results, "source_agent"),
        by_family=_group_mean(results, "task_family"),
        results=results,
    )


def _group_mean(results: list[AiforaiReplayResult], attr: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for result in results:
        buckets[str(getattr(result, attr))].append(result.hard)
    return {
        key: sum(values) / len(values)
        for key, values in buckets.items()
    }

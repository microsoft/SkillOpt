"""Audit orchestration for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.harvesters import (
    ClaudeHarvester,
    CodeWhaleHarvester,
    CodexHarvester,
    Harvester,
)
from skillopt_sleep.aiforai.mine import mine_tasks, split_tasks
from skillopt_sleep.aiforai.report import make_staging_dir, write_audit_report
from skillopt_sleep.aiforai.types import AiforaiRunResult, AiforaiSessionDigest


def default_harvesters(sources: Iterable[str]) -> list[Harvester]:
    known = {
        "codex": CodexHarvester,
        "claude": ClaudeHarvester,
        "codewhale": CodeWhaleHarvester,
    }
    harvesters: list[Harvester] = []
    for source in sources:
        harvester_cls = known.get(source)
        if harvester_cls is None:
            continue
        harvesters.append(harvester_cls())
    return harvesters


def run_audit(
    cfg: AiforaiConfig,
    harvesters: Iterable[Harvester] | None = None,
) -> AiforaiRunResult:
    active_harvesters = list(harvesters) if harvesters is not None else default_harvesters(cfg.sources)
    sessions: list[AiforaiSessionDigest] = []
    notes: list[str] = []
    sessions_by_source = {source: 0 for source in cfg.sources}

    for harvester in active_harvesters:
        source = str(getattr(harvester, "source_agent", "unknown"))
        sessions_by_source.setdefault(source, 0)
        try:
            harvested = harvester.harvest(cfg)
        except Exception as exc:
            notes.append(f"{source}: harvester failed: {exc}")
            continue
        sessions.extend(harvested)
        sessions_by_source[source] += len(harvested)

    tasks, uncheckable = mine_tasks(
        sessions,
        max_tasks_per_source=cfg.max_tasks_per_source,
    )
    split_tasks(
        tasks,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        seed=cfg.seed,
    )

    task_counts = Counter(task.source_agent for task in tasks)
    tasks_by_source = {source: task_counts.get(source, 0) for source in sessions_by_source}
    staging_dir = make_staging_dir(cfg.target_skill_repo, "audit")
    result = AiforaiRunResult(
        mode="audit",
        staging_dir=staging_dir,
        sessions_by_source=sessions_by_source,
        tasks_by_source=tasks_by_source,
        checkable_tasks=len(tasks),
        uncheckable_candidates=len(uncheckable),
        accepted=False,
        notes=notes,
    )
    write_audit_report(staging_dir, sessions, tasks, uncheckable, result)
    return result

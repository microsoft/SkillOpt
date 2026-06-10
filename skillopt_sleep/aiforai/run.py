"""Audit and staged mock-run orchestration for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.harvesters import (
    ClaudeHarvester,
    CodeWhaleHarvester,
    CodexHarvester,
    Harvester,
)
from skillopt_sleep.aiforai.mine import mine_tasks, split_tasks
from skillopt_sleep.aiforai.regression_suite import curated_regression_tasks
from skillopt_sleep.aiforai.report import (
    make_staging_dir,
    write_audit_report,
    write_jsonl,
    write_run_report,
)
from skillopt_sleep.aiforai.replay import evaluate_tasks, gate_candidate, propose_mock_rules
from skillopt_sleep.aiforai.skill_adapter import (
    apply_learned_rules,
    current_learned_rules,
    read_skill,
    run_aiforai_validators,
    write_skill,
)
from skillopt_sleep.aiforai.types import AiforaiRunResult, AiforaiSessionDigest, AiforaiTaskRecord


SUPPORTED_SOURCES = ("codex", "claude", "codewhale")


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
    sessions, notes, sessions_by_source = _harvest_sessions(
        cfg,
        harvesters=harvesters,
        require_nonempty=True,
    )

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

    tasks_by_source = _task_counts(tasks, sessions_by_source)
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


def run_mock_gate(
    cfg: AiforaiConfig,
    *,
    sessions: Iterable[AiforaiSessionDigest] | None = None,
    run_validators: bool = True,
) -> AiforaiRunResult:
    notes: list[str] = []
    if sessions is None:
        selected_sessions, harvest_notes, sessions_by_source = _harvest_sessions(
            cfg,
            require_nonempty=False,
        )
        notes.extend(harvest_notes)
    else:
        selected_sessions = list(sessions)
        sessions_by_source = _source_counts(selected_sessions, cfg.sources)

    tasks, uncheckable = mine_tasks(
        selected_sessions,
        max_tasks_per_source=cfg.max_tasks_per_source,
    )
    split_tasks(
        tasks,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        seed=cfg.seed,
    )

    live_skill = read_skill(cfg.skill_path)
    real_eval_tasks = [task for task in tasks if task.split == "val"] or list(tasks)
    curated_tasks = curated_regression_tasks()
    eval_tasks = list(real_eval_tasks) + curated_tasks

    has_selected_sessions = bool(selected_sessions)
    has_real_tasks = bool(tasks)
    if has_selected_sessions and has_real_tasks:
        learned_rules = current_learned_rules(live_skill)
        proposed_rules = propose_mock_rules(eval_tasks, live_skill)
        candidate_skill = apply_learned_rules(live_skill, learned_rules + proposed_rules)
    else:
        candidate_skill = live_skill
        if not has_selected_sessions:
            notes.append(
                "curated regression tasks are supplemental only; no harvested sessions were available."
            )
        if not has_real_tasks:
            notes.append(
                "curated regression tasks are supplemental only; no mined real tasks were available."
            )

    baseline = evaluate_tasks(eval_tasks, live_skill)
    candidate = evaluate_tasks(eval_tasks, candidate_skill)
    gate = gate_candidate(baseline, candidate)
    validation = (
        _validate_candidate_skill(cfg, candidate_skill)
        if run_validators
        else {"ok": True, "commands": [], "skipped": True}
    )
    accepted = has_selected_sessions and has_real_tasks and gate.accepted and bool(
        validation.get("ok")
    )

    notes.append(gate.reason)
    if run_validators and not validation.get("ok"):
        notes.append("validators failed")

    staging_dir = make_staging_dir(cfg.target_skill_repo, "run")
    result = AiforaiRunResult(
        mode="run",
        staging_dir=staging_dir,
        sessions_by_source=sessions_by_source,
        tasks_by_source=_task_counts(tasks, sessions_by_source),
        checkable_tasks=len(tasks),
        uncheckable_candidates=len(uncheckable),
        accepted=accepted,
        baseline_score=baseline.aggregate_hard,
        candidate_score=candidate.aggregate_hard,
        notes=notes,
    )

    with open(os.path.join(staging_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "live_skill_path": cfg.skill_path,
                "accepted": accepted,
                "has_skill": True,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    write_jsonl(
        os.path.join(staging_dir, "task_manifest.jsonl"),
        [task.to_dict() for task in tasks],
    )
    write_jsonl(
        os.path.join(staging_dir, "uncheckable_candidates.jsonl"),
        uncheckable,
    )
    write_run_report(
        staging_dir,
        result,
        live_skill,
        candidate_skill,
        [row.to_dict() for row in baseline.results],
        [row.to_dict() for row in candidate.results],
        validation,
        coverage={
            "sessions_by_source": result.sessions_by_source,
            "tasks_by_source": result.tasks_by_source,
            "real_task_count": len(tasks),
            "curated_task_count": len(curated_tasks),
            "eval_task_count": len(eval_tasks),
        },
        live_skill_path=cfg.skill_path,
    )
    return result


def adopt_latest(cfg: AiforaiConfig) -> list[str]:
    root = Path(cfg.staging_root)
    if not root.is_dir():
        return []

    configured_live_path = cfg.skill_path
    allowed_live_path = Path(cfg.skill_path).resolve(strict=False)
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.is_symlink()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for staging in candidates:
        manifest_path = staging / "manifest.json"
        proposed_path = staging / "proposed_SKILL.md"
        if not manifest_path.is_file() or not proposed_path.is_file():
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if manifest.get("accepted") is not True:
            continue

        live_path = manifest.get("live_skill_path")
        if not isinstance(live_path, str) or not live_path.strip():
            continue
        resolved_live_path = Path(live_path).resolve(strict=False)
        if resolved_live_path != allowed_live_path:
            continue

        backup_dir = _prepare_backup_dir(staging)
        if backup_dir is None:
            continue

        backup_path = backup_dir / "SKILL.md"
        if os.path.exists(configured_live_path):
            shutil.copy2(configured_live_path, backup_path)

        write_skill(configured_live_path, proposed_path.read_text(encoding="utf-8"))
        return [configured_live_path]
    return []


def _prepare_backup_dir(staging: Path) -> Path | None:
    backup_dir = staging / "backup"
    if backup_dir.is_symlink():
        return None

    if backup_dir.exists():
        return backup_dir if backup_dir.is_dir() else None

    try:
        backup_dir.mkdir()
    except FileExistsError:
        if backup_dir.is_symlink():
            return None
        return backup_dir if backup_dir.is_dir() else None

    return backup_dir


def _harvest_sessions(
    cfg: AiforaiConfig,
    *,
    harvesters: Iterable[Harvester] | None = None,
    require_nonempty: bool,
) -> tuple[list[AiforaiSessionDigest], list[str], dict[str, int]]:
    active_harvesters = list(harvesters) if harvesters is not None else default_harvesters(cfg.sources)
    if not active_harvesters:
        requested_sources = ", ".join(cfg.sources) if cfg.sources else "<none>"
        raise ValueError(
            f"No supported AIForAI harvesters selected for sources: {requested_sources}"
        )

    sessions: list[AiforaiSessionDigest] = []
    notes: list[str] = []
    sessions_by_source = {source: 0 for source in cfg.sources}

    for harvester in active_harvesters:
        source = str(getattr(harvester, "source_agent", "unknown"))
        try:
            harvested = harvester.harvest(cfg)
        except Exception as exc:
            notes.append(f"{source}: harvester failed: {exc}")
            continue

        harvested_counts = Counter(str(session.source_agent) for session in harvested)
        if any(emitted_source != source for emitted_source in harvested_counts):
            source_summary = ", ".join(
                f"{emitted_source}={count}"
                for emitted_source, count in harvested_counts.items()
            )
            notes.append(
                f"{source}: harvested session source mismatch; using digest sources "
                f"({source_summary})"
            )
        sessions.extend(harvested)

    if not sessions and require_nonempty:
        requested_sources = ", ".join(cfg.sources) if cfg.sources else "<none>"
        message = f"No AIForAI sessions were harvested for selected sources: {requested_sources}"
        failure_notes = [note for note in notes if "harvester failed:" in note]
        if failure_notes:
            message = f"{message}. Failures: {'; '.join(failure_notes)}"
        raise ValueError(message)

    session_counts = Counter(str(session.source_agent) for session in sessions)
    for source, count in session_counts.items():
        sessions_by_source[source] = count

    return sessions, notes, sessions_by_source


def _source_counts(
    sessions: Iterable[AiforaiSessionDigest],
    configured_sources: Iterable[str],
) -> dict[str, int]:
    counts = {source: 0 for source in configured_sources}
    for source, count in Counter(str(session.source_agent) for session in sessions).items():
        counts[source] = count
    return counts


def _task_counts(
    tasks: Iterable[AiforaiTaskRecord],
    sessions_by_source: dict[str, int],
) -> dict[str, int]:
    counts = Counter(task.source_agent for task in tasks)
    return {source: counts.get(source, 0) for source in sessions_by_source}


def _validate_candidate_skill(cfg: AiforaiConfig, candidate_skill: str) -> dict[str, object]:
    try:
        with tempfile.TemporaryDirectory(prefix="aiforai-validate-") as tmp:
            repo_copy = os.path.join(tmp, "repo")
            shutil.copytree(
                cfg.target_skill_repo,
                repo_copy,
                ignore=shutil.ignore_patterns(".git", ".skillopt-sleep", "__pycache__"),
            )
            write_skill(os.path.join(repo_copy, cfg.skill_rel_path), candidate_skill)
            return run_aiforai_validators(repo_copy)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "commands": [],
            "error": str(exc),
        }

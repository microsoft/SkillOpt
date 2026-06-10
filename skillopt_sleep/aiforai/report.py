"""Audit report writers for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

from skillopt_sleep.aiforai.types import (
    AiforaiRunResult,
    AiforaiSessionDigest,
    AiforaiTaskRecord,
)


def make_staging_dir(target_skill_repo: str, prefix: str) -> str:
    staging_root = os.path.join(target_skill_repo, ".skillopt-sleep", "staging")
    os.makedirs(staging_root, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{stamp}-{prefix}"
    out_dir = os.path.join(staging_root, base_name)
    suffix = 1
    while os.path.exists(out_dir):
        out_dir = os.path.join(staging_root, f"{base_name}-{suffix}")
        suffix += 1
    os.makedirs(out_dir, exist_ok=False)
    return out_dir


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_audit_report(
    out_dir: str,
    sessions: list[AiforaiSessionDigest],
    tasks: list[AiforaiTaskRecord],
    uncheckable: list[dict[str, Any]],
    result: AiforaiRunResult,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    session_rows = [session.to_dict() for session in sessions]
    task_rows = [task.to_dict() for task in tasks]
    family_counts = Counter(task.task_family for task in tasks)
    split_counts = Counter(task.split for task in tasks)
    uncheckable_reason_counts = Counter(
        str(candidate.get("reason") or "unknown") for candidate in uncheckable
    )

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "result": result.to_dict(),
                "source_coverage": {
                    "sessions_by_source": result.sessions_by_source,
                    "tasks_by_source": result.tasks_by_source,
                },
                "task_families": dict(family_counts),
                "checkability": {
                    "checkable_tasks": result.checkable_tasks,
                    "uncheckable_candidates": result.uncheckable_candidates,
                    "uncheckable_reasons": dict(uncheckable_reason_counts),
                },
                "splits": dict(split_counts),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    write_jsonl(os.path.join(out_dir, "sessions.jsonl"), session_rows)
    write_jsonl(os.path.join(out_dir, "task_manifest.jsonl"), task_rows)
    write_jsonl(
        os.path.join(out_dir, "uncheckable_candidates.jsonl"),
        uncheckable,
    )

    lines = [
        "# AIForAI Audit Report",
        "",
        "Audit boundary: read-only.",
        "No live AIForAI skill files were modified.",
        "",
        "## Summary",
        f"- Mode: {result.mode}",
        f"- Staging dir: {result.staging_dir}",
        f"- Sessions harvested: {len(sessions)}",
        f"- Checkable tasks: {result.checkable_tasks}",
        f"- Uncheckable candidates: {result.uncheckable_candidates}",
        "",
        "## Source Coverage",
    ]
    if result.sessions_by_source:
        for source, count in result.sessions_by_source.items():
            lines.append(
                f"- {source}: {count} sessions, {result.tasks_by_source.get(source, 0)} checkable tasks"
            )
    else:
        lines.append("- No sessions harvested.")

    lines.extend(["", "## Task Families"])
    if family_counts:
        for family, count in sorted(family_counts.items()):
            lines.append(f"- {family}: {count}")
    else:
        lines.append("- No checkable task families found.")

    lines.extend(
        [
            "",
            "## Checkability",
            f"- Checkable tasks: {result.checkable_tasks}",
            f"- Uncheckable candidates: {result.uncheckable_candidates}",
        ]
    )
    if split_counts:
        lines.append(
            "- Split allocation: "
            + ", ".join(f"{split}={count}" for split, count in sorted(split_counts.items()))
        )
    if uncheckable_reason_counts:
        lines.append(
            "- Uncheckable reasons: "
            + ", ".join(
                f"{reason}={count}" for reason, count in sorted(uncheckable_reason_counts.items())
            )
        )

    lines.extend(["", "## Notes"])
    if result.notes:
        lines.extend(f"- {note}" for note in result.notes)
    else:
        lines.append("- None.")

    lines.append("")
    with open(os.path.join(out_dir, "audit_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_run_report(
    out_dir: str,
    result: AiforaiRunResult,
    proposed_skill: str,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "proposed_SKILL.md"), "w", encoding="utf-8") as handle:
        handle.write(proposed_skill)

    write_jsonl(os.path.join(out_dir, "baseline_results.jsonl"), baseline_rows)
    write_jsonl(os.path.join(out_dir, "candidate_results.jsonl"), candidate_rows)

    with open(os.path.join(out_dir, "validation.log"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(validation, ensure_ascii=False, indent=2))
        handle.write("\n")

    lines = [
        "# AIForAI SkillOpt-Sleep Run Report",
        "",
        "## Summary",
        f"- accepted: {result.accepted}",
        f"- baseline_score: {result.baseline_score:.4f}",
        f"- candidate_score: {result.candidate_score:.4f}",
        f"- checkable_tasks: {result.checkable_tasks}",
        f"- uncheckable_candidates: {result.uncheckable_candidates}",
        "",
        "## Boundary",
        "- This run staged a proposal only.",
        "- Live skill mutation requires explicit adopt.",
        "",
        "## Notes",
    ]
    if result.notes:
        lines.extend(f"- {note}" for note in result.notes)
    else:
        lines.append("- None.")
    lines.append("")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "result": result.to_dict(),
                "baseline_results": baseline_rows,
                "candidate_results": candidate_rows,
                "validation": validation,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

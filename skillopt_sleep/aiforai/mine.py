"""Mining and source-stratified splitting for AIForAI tasks."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from skillopt_sleep.aiforai.types import AiforaiSessionDigest, AiforaiTaskRecord


FAMILY_RULES: list[tuple[str, tuple[str, ...], list[dict]]] = [
    (
        "training_contract",
        ("training contract", "resume training", "start a training", "训练", "train"),
        [
            {"op": "contains", "arg": "training contract"},
            {"op": "contains", "arg": "evaluation contract"},
            {"op": "contains", "arg": "stop criteria"},
            {"op": "contains", "arg": "artifact paths"},
        ],
    ),
    (
        "data_acquisition",
        ("download", "dataset", "data acquisition", "数据", "下载"),
        [
            {"op": "contains", "arg": "scope"},
            {"op": "contains", "arg": "shared storage"},
            {"op": "contains", "arg": "do not download full datasets locally"},
        ],
    ),
    (
        "cluster_preflight",
        ("cluster", "volcano", "k8s", "kubectl", "muxi", "ascend"),
        [
            {"op": "contains", "arg": "image"},
            {"op": "contains", "arg": "dependencies"},
            {"op": "contains", "arg": "data access"},
            {"op": "contains", "arg": "artifact"},
        ],
    ),
    (
        "dirty_worktree_gate",
        ("dirty", "git status", "worktree", "uncommitted"),
        [
            {"op": "contains", "arg": "git status"},
            {"op": "contains", "arg": "dirty"},
            {"op": "contains", "arg": "formal"},
        ],
    ),
    (
        "claim_integrity",
        ("done", "complete", "完成", "ready", "status"),
        [
            {"op": "contains", "arg": "Delivered artifact"},
            {"op": "contains", "arg": "Verified evidence"},
            {"op": "contains", "arg": "Unverified boundary"},
            {"op": "contains", "arg": "Next deliverable"},
        ],
    ),
    (
        "rag_agent_diagnosis",
        ("rag", "retrieval", "agent failure", "tool trajectory", "trajectory"),
        [
            {"op": "contains", "arg": "retrieval"},
            {"op": "contains", "arg": "tool"},
            {"op": "contains", "arg": "trajectory"},
        ],
    ),
]


def mine_tasks(
    sessions: list[AiforaiSessionDigest],
    *,
    max_tasks_per_source: int = 40,
) -> tuple[list[AiforaiTaskRecord], list[dict]]:
    tasks: list[AiforaiTaskRecord] = []
    uncheckable: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    for session in sessions:
        if counts[session.source_agent] >= max_tasks_per_source:
            continue
        text = "\n".join(session.user_prompts + session.assistant_finals)
        family, checks = classify_family(text)
        if not family:
            uncheckable.append(
                {
                    "source_agent": session.source_agent,
                    "session_id": session.session_id,
                    "reason": "no_checkable_aiforai_family",
                    "preview": text[:300],
                }
            )
            continue
        intent = session.user_prompts[0] if session.user_prompts else text[:200]
        task = AiforaiTaskRecord(
            id=_task_id(session.source_agent, family, intent),
            source_agent=session.source_agent,
            source_sessions=[session.session_id],
            project=session.cwd,
            intent=intent[:800],
            context_excerpt=text[:1200],
            task_family=family,
            outcome=_outcome(session.feedback_signals),
            judge={"kind": "rule", "checks": checks},
        )
        tasks.append(task)
        counts[session.source_agent] += 1
    return dedup_tasks(tasks), uncheckable


def classify_family(text: str) -> tuple[str, list[dict]]:
    low = (text or "").lower()
    best_family = ""
    best_checks: list[dict] = []
    best_score = 0
    for family, needles, checks in FAMILY_RULES:
        score = sum(needle.lower() in low for needle in needles)
        if score > best_score:
            best_family = family
            best_checks = list(checks)
            best_score = score
    return best_family, best_checks


def split_tasks(
    tasks: list[AiforaiTaskRecord],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[AiforaiTaskRecord]:
    groups: dict[tuple[str, str], list[AiforaiTaskRecord]] = defaultdict(list)
    for task in tasks:
        groups[(task.source_agent, task.task_family)].append(task)
    for (_source, _family), group in groups.items():
        ordered = sorted(group, key=lambda task: _stable_hash(f"{seed}:{task.id}"))
        n = len(ordered)
        n_val = 1 if n >= 2 else 0
        n_val = max(n_val, int(round(n * val_fraction)))
        n_test = int(round(n * test_fraction))
        for idx, task in enumerate(ordered):
            if idx < n_val:
                task.split = "val"
            elif idx < n_val + n_test:
                task.split = "test"
            else:
                task.split = "train"
        if n >= 2 and not any(task.split == "train" for task in group):
            ordered[-1].split = "train"
    return tasks


def dedup_tasks(tasks: list[AiforaiTaskRecord]) -> list[AiforaiTaskRecord]:
    by_id: dict[str, AiforaiTaskRecord] = {}
    for task in tasks:
        existing = by_id.get(task.id)
        if existing is None:
            by_id[task.id] = task
            continue
        existing.source_sessions = list(
            dict.fromkeys(existing.source_sessions + task.source_sessions)
        )
    return list(by_id.values())


def _task_id(source: str, family: str, intent: str) -> str:
    digest = hashlib.sha256(
        f"{source}:{family}:{intent.lower()}".encode("utf-8")
    ).hexdigest()[:12]
    return f"aiforai_{digest}"


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _outcome(signals: list[str]) -> str:
    if any(signal.startswith("neg:") for signal in signals):
        return "fail"
    if any(signal.startswith("pos:") for signal in signals):
        return "success"
    return "unknown"

"""Mining and source-stratified splitting for AIForAI tasks."""

from __future__ import annotations

import hashlib
import re
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

LOW_SIGNAL_NEEDLES = {"done", "complete", "ready", "status"}


def mine_tasks(
    sessions: list[AiforaiSessionDigest],
    *,
    max_tasks_per_source: int = 40,
) -> tuple[list[AiforaiTaskRecord], list[dict]]:
    tasks: list[AiforaiTaskRecord] = []
    uncheckable: list[dict] = []
    for session in sessions:
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

    deduped = dedup_tasks(tasks)
    capped: list[AiforaiTaskRecord] = []
    counts: dict[str, int] = defaultdict(int)
    for task in deduped:
        if counts[task.source_agent] >= max_tasks_per_source:
            continue
        capped.append(task)
        counts[task.source_agent] += 1
    return capped, uncheckable


def classify_family(text: str) -> tuple[str, list[dict]]:
    low = (text or "").lower()
    matches: list[tuple[str, list[str], list[dict]]] = []
    for family, needles, checks in FAMILY_RULES:
        matched_needles = [
            needle for needle in needles if _contains_needle(low, needle.lower())
        ]
        if matched_needles:
            matches.append((family, matched_needles, list(checks)))

    if not matches:
        return "", []
    if len(matches) == 1:
        family, _needles, checks = matches[0]
        return family, checks

    matches.sort(key=lambda item: len(item[1]), reverse=True)
    top_family, top_needles, top_checks = matches[0]
    second_score = len(matches[1][1])
    signal_families = [
        family
        for family, needles, _checks in matches
        if any(needle not in LOW_SIGNAL_NEEDLES for needle in needles)
    ]
    if len(signal_families) > 1:
        return "", []
    if len(top_needles) == second_score:
        return "", []
    return top_family, top_checks


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
        n_val = min(n_val, max(0, n - 1))
        n_test = int(round(n * test_fraction))
        max_test = max(0, n - n_val - 1)
        if test_fraction > 0 and max_test > 0:
            n_test = max(1, n_test)
        n_test = min(n_test, max_test)
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
        existing.outcome = _merge_outcome(existing.outcome, task.outcome)
    return list(by_id.values())


def _task_id(source: str, family: str, intent: str) -> str:
    digest = hashlib.sha256(
        f"{source}:{family}:{intent.lower()}".encode("utf-8")
    ).hexdigest()[:12]
    return f"aiforai_{digest}"


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _contains_needle(low_text: str, low_needle: str) -> bool:
    if not low_needle:
        return False
    if low_needle.isascii():
        pattern = rf"(?<!\w){re.escape(low_needle)}(?!\w)"
        return re.search(pattern, low_text) is not None
    return low_needle in low_text


def _merge_outcome(left: str, right: str) -> str:
    if left == "mixed" or right == "mixed":
        return "mixed"
    if left == right:
        return left
    if left == "unknown":
        return right
    if right == "unknown":
        return left
    return "mixed"


def _outcome(signals: list[str]) -> str:
    if any(signal.startswith("neg:") for signal in signals):
        return "fail"
    if any(signal.startswith("pos:") for signal in signals):
        return "success"
    return "unknown"

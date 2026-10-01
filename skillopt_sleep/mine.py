"""SkillOpt-Sleep — Stage 2: mine.

Turn :class:`SessionDigest` objects into :class:`TaskRecord` training units.

Two miners:
  * heuristic_mine  — deterministic, no API. Detects retry chains (a prompt
    re-asked after negative feedback => the early attempt failed), extracts
    the user's recurring intents, and labels outcomes from feedback signals.
  * llm_mine        — optional; uses an optimizer backend to produce richer
    TaskRecords with checkable references. Falls back to heuristic on error.

The heuristic miner is what makes the whole cycle runnable offline and is the
basis of the deterministic experiment.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from collections import Counter
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Set, Tuple

from skillopt_sleep.backend import CursorBackendError
from skillopt_sleep.types import SessionDigest, TaskRecord

_LOGGER = logging.getLogger("skillopt_sleep")


def _tid(project: str, intent: str) -> str:
    h = hashlib.sha256((project + "::" + intent).encode("utf-8")).hexdigest()[:12]
    return "task_" + h


def _short(text: str, n: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + " …"


def session_skill_hint(digest: SessionDigest) -> str:
    """Return the session's unambiguous skill, or "" when there is none.

    A session that invoked exactly one skill names the skill its tasks
    exercised. Zero skills (unknown) and several skills (ambiguous) both
    return "", which leaves those tasks in the existing catch-all path.
    """
    skills = [s for s in (digest.skills_used or []) if s]
    unique = list(dict.fromkeys(skills))
    return unique[0] if len(unique) == 1 else ""


def _looks_negative(signals: List[str]) -> bool:
    return any(s.startswith("neg:") for s in signals)


def _looks_positive(signals: List[str]) -> bool:
    return any(s.startswith("pos:") for s in signals)


_TARGET_STOPWORDS = {
    "about", "after", "again", "agent", "agents", "all", "also", "always",
    "and", "any", "are", "before", "being", "but", "can", "codex",
    "current", "default", "docs", "does", "done", "each", "file", "files",
    "for", "from", "have", "into", "keep", "must", "not", "only", "path",
    "paths", "project", "read", "repo", "request", "requests", "rule",
    "rules", "same", "should", "skill", "skills", "source", "start",
    "task", "tasks", "that", "the", "their", "then", "this", "unless",
    "update", "user", "users", "when", "with", "work", "workflow",
}


def _target_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in re.findall(r"[\w][\w.-]*", (text or "").lower(), flags=re.UNICODE):
        parts = [raw] + re.split(r"[\W_]+", raw, flags=re.UNICODE)
        for part in parts:
            if len(part) < 3 or part.isdigit() or part in _TARGET_STOPWORDS:
                continue
            tokens.append(part)
    return tokens


def _expand_target_keywords(keywords: Set[str]) -> None:
    if "mcp" in keywords:
        keywords.update({
            "configure", "configuration", "connect", "connected", "enable",
            "enabled", "install", "installed", "server", "servers",
            "настрой", "настроить", "подключи", "подключить",
        })
    if {"conflict", "conflicts"} & keywords:
        keywords.update({
            "cherry", "conflict", "conflicts", "git", "merge", "rebase",
            "unmerged", "конфликт", "конфликты",
        })


def target_task_keywords(
    target_skill_text: str,
    target_skill_path: str = "",
    *,
    limit: int = 180,
) -> Tuple[Set[str], Set[str]]:
    """Return (strong, weak) keywords that describe a target skill."""
    path_text = (target_skill_path or "").replace(os.sep, " ")
    headings = "\n".join(re.findall(r"(?m)^#+\s+(.+)$", target_skill_text or ""))
    strong = set(_target_tokens(path_text + "\n" + headings))
    weak = set(strong)
    counts = Counter(_target_tokens(target_skill_text or ""))
    for token, _count in counts.most_common(limit):
        weak.add(token)
    _expand_target_keywords(strong)
    _expand_target_keywords(weak)
    return strong, weak


def _task_search_text(task: TaskRecord) -> str:
    return "\n".join([
        task.intent or "",
        task.context_excerpt or "",
        " ".join(task.tags or []),
    ])


def filter_tasks_for_target(
    tasks: List[TaskRecord],
    target_skill_text: str,
    target_skill_path: str = "",
) -> List[TaskRecord]:
    """Prefer tasks whose language overlaps the explicit target skill.

    If nothing matches, return the original list. This keeps a target run useful
    even when transcripts are too sparse or the skill is too generic.
    """
    strong, weak = target_task_keywords(target_skill_text, target_skill_path)
    if not tasks or not (strong or weak):
        return tasks

    ranked = []
    for idx, task in enumerate(tasks):
        tokens = set(_target_tokens(_task_search_text(task)))
        strong_hits = tokens & strong
        weak_hits = tokens & weak
        if not strong_hits and len(weak_hits) < 2:
            continue
        score = len(strong_hits) * 3 + len(weak_hits)
        ranked.append((score, idx, task))
    if not ranked:
        return tasks
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [task for _score, _idx, task in ranked]


def heuristic_mine(
    digests: List[SessionDigest],
    *,
    max_tasks: int = 40,
) -> List[TaskRecord]:
    """Deterministic miner — no API calls.

    Strategy:
      * Each session with >=1 real user prompt yields one TaskRecord whose
        intent is the FIRST substantive prompt (the original ask).
      * Outcome is inferred:
          - negative feedback present and no later positive  -> "fail"
          - positive feedback present                         -> "success"
          - re-asks (multiple user turns) without resolution  -> "mixed"
          - otherwise                                         -> "unknown"
      * attempted_solution = the last assistant final (what was produced).
      * reference_kind defaults to "none"; the consolidation step will use a
        rubric judge for these. (Exact refs are added by the experiment data
        or by the LLM miner when it can derive a checkable answer.)
    """
    tasks: List[TaskRecord] = []
    for d in digests:
        if not d.user_prompts:
            continue
        intent = d.user_prompts[0]
        if len(intent.strip()) < 8:
            continue
        if _looks_positive(d.feedback_signals) and not _looks_negative(d.feedback_signals):
            outcome = "success"
        elif _looks_negative(d.feedback_signals):
            outcome = "fail"
        elif d.n_user_turns >= 3:
            outcome = "mixed"
        else:
            outcome = "unknown"

        attempted = d.assistant_finals[-1] if d.assistant_finals else ""
        context = ""
        if len(d.user_prompts) > 1:
            # later prompts often carry the corrective detail / real constraints
            context = "Follow-up constraints from the same session:\n- " + "\n- ".join(
                _short(p, 200) for p in d.user_prompts[1:4]
            )
        tags = []
        if d.tools_used:
            tags.append("tools:" + "+".join(d.tools_used[:4]))
        if d.git_branch:
            tags.append("branch:" + d.git_branch)

        tasks.append(
            TaskRecord(
                id=_tid(d.project, intent),
                project=d.project,
                intent=_short(intent, 800),
                context_excerpt=_short(context, 600),
                attempted_solution=_short(attempted, 600),
                outcome=outcome,
                reference_kind="none",
                reference="",
                tags=tags,
                source_sessions=[d.session_id],
                skill_hint=session_skill_hint(d),
            )
        )
        if len(tasks) >= max_tasks:
            break
    return tasks


def dedup_tasks(tasks: List[TaskRecord]) -> List[TaskRecord]:
    """Merge tasks sharing an id (same project+intent across sessions)."""
    by_id: dict = {}
    hints_by_id: dict = {}
    for t in tasks:
        if t.skill_hint:
            hints_by_id.setdefault(t.id, set()).add(t.skill_hint)
        if t.id in by_id:
            ex = by_id[t.id]
            ex.source_sessions = list(dict.fromkeys(ex.source_sessions + t.source_sessions))
            # prefer a resolved outcome if either session resolved it
            order = {"success": 3, "fail": 2, "mixed": 1, "unknown": 0}
            if order.get(t.outcome, 0) > order.get(ex.outcome, 0):
                ex.outcome = t.outcome
        else:
            by_id[t.id] = t
    for task_id, task in by_id.items():
        hints = hints_by_id.get(task_id, set())
        task.skill_hint = next(iter(hints)) if len(hints) == 1 else ""
    return list(by_id.values())


def group_tasks_by_skill_hint(
    tasks: List[TaskRecord],
    managed_skill_name: str,
) -> Dict[str, List[TaskRecord]]:
    """Group mined tasks by their skill hint, in first-seen order.

    A task ID reaches a hinted group only when every observation of it agrees on
    the same non-empty hint. Missing hints, conflicting hints, and a mixture of
    hinted and unhinted observations all fall back to ``managed_skill_name`` —
    the existing catch-all skill — so ambiguous evidence never invents a group.
    Each task ID is emitted exactly once, with ``dedup_tasks`` merge semantics.
    """
    observed: dict = {}
    for t in tasks:
        observed.setdefault(t.id, set()).add((t.skill_hint or "").strip())

    # ``dedup_tasks`` merges records in place, so operate on shallow dataclass
    # copies (including the only list it mutates) rather than caller-owned tasks.
    copied = [replace(t, source_sessions=list(t.source_sessions)) for t in tasks]
    groups: Dict[str, List[TaskRecord]] = {}
    for task in dedup_tasks(copied):
        hints = observed[task.id]
        hint = next(iter(hints)) if len(hints) == 1 else ""
        # Keep the returned record aligned with the normalized evidence used for
        # routing. In particular, blank and partially observed hints stay empty.
        task.skill_hint = hint
        groups.setdefault(hint or managed_skill_name, []).append(task)
    return groups


def assign_splits(
    tasks: List[TaskRecord],
    *,
    val_fraction: float = 0.34,
    test_fraction: float = 0.0,
    holdout_fraction: float | None = None,  # legacy alias for val_fraction
    seed: int = 42,
) -> List[TaskRecord]:
    """Deterministically split tasks into train / val / test.

    Anti-overfitting contract (the user's design):
      * ``val`` and ``test`` are drawn ONLY from REAL mined tasks (origin=='real')
        and never overlap. val gates updates; test is the final held-out measure.
      * ``train`` may include DREAM-augmented tasks (origin=='dream'); those are
        NEVER placed in val/test.

    A stable hash of the task id keeps the same real task in the same split across
    nights (a fixed held-out gate, like SkillOpt's D_sel/D_test). For a small
    batch whose hash assignment cannot supply both train and val, the minimum
    number of test tasks is reassigned and a warning records the reduced test
    coverage.

    Back-compat: if ``test_fraction`` is 0 (default), this behaves like the old
    two-way replay/holdout split — real tasks divide into train + val, no test.
    ``holdout_fraction`` is accepted as an alias for ``val_fraction``.
    """
    if holdout_fraction is not None:
        val_fraction = holdout_fraction

    for name, value in (("val_fraction", val_fraction), ("test_fraction", test_fraction)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1 inclusive, got {value}"
            )
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction + test_fraction must be < 1 "
            f"(got {val_fraction} + {test_fraction})"
        )

    dream = [t for t in tasks if t.origin == "dream"]
    real = [t for t in tasks if t.origin != "dream"]

    # all dream tasks go to train, unconditionally
    for t in dream:
        t.split = "train"

    val_cut = int(round(val_fraction * 100))
    test_cut = val_cut + int(round(test_fraction * 100))

    def _stable_key(task: TaskRecord) -> tuple[int, str]:
        bucket = int(hashlib.sha256((str(seed) + task.id).encode()).hexdigest(), 16)
        return bucket, task.id

    promoted_from_test: List[Tuple[str, str]] = []

    def _promote_one(*, to: str, from_splits: set[str]) -> None:
        """Promote one real task using stable hash order."""
        candidates = [t for t in real if t.split in from_splits]
        if not candidates:
            return
        candidates.sort(key=_stable_key)
        selected = candidates[0]
        if selected.split == "test":
            promoted_from_test.append((selected.id, to))
        selected.split = to

    for t in real:
        bucket = _stable_key(t)[0] % 100
        if bucket < val_cut:
            t.split = "val"
        elif bucket < test_cut:
            t.split = "test"
        else:
            t.split = "train"

    # Guarantee val (the gate) is non-empty when we have >=2 real tasks. Keep
    # hash-assigned test tasks untouched unless the non-test pool cannot supply
    # distinct train and val tasks.
    if len(real) >= 2 and not any(t.split == "val" for t in real):
        real_train = [t for t in real if t.split == "train"]
        train_count = sum(t.split == "train" for t in tasks)
        if real_train and train_count >= 2:
            _promote_one(to="val", from_splits={"train"})
        else:
            _promote_one(to="val", from_splits={"test"})
    # Guarantee a train pool exists when possible. Prefer a spare val task; if
    # the only val task is the gate, borrow from test so val stays non-empty.
    if not any(t.split == "train" for t in tasks) and len(real) >= 2:
        val_count = sum(t.split == "val" for t in real)
        if val_count >= 2:
            _promote_one(to="train", from_splits={"val"})
        else:
            _promote_one(to="train", from_splits={"test"})

    if promoted_from_test:
        assignments = ", ".join(f"{task_id}->{split}" for task_id, split in promoted_from_test)
        _LOGGER.warning(
            "assign_splits reassigned %d test task(s) to keep train and val "
            "non-empty; held-out test coverage was reduced (%s)",
            len(promoted_from_test),
            assignments,
        )
    return tasks


def normalize_legacy_split(value: str) -> str:
    """Map old split names to the new vocabulary."""
    return {"replay": "train", "holdout": "val"}.get(value, value)


def mine(
    digests: List[SessionDigest],
    *,
    max_tasks: int = 40,
    candidate_limit: int = 0,
    val_fraction: float = 0.34,
    test_fraction: float = 0.0,
    holdout_fraction: float | None = None,  # legacy alias for val_fraction
    seed: int = 42,
    llm_miner: Optional[Callable[[List[SessionDigest]], List[TaskRecord]]] = None,
    target_skill_text: str = "",
    target_skill_path: str = "",
) -> List[TaskRecord]:
    """Top-level miner. Uses ``llm_miner`` if provided, else heuristic.

    Split knobs mirror ``assign_splits``: ``val_fraction``/``test_fraction``
    are the real controls; ``holdout_fraction`` remains the legacy alias and,
    when passed, overrides ``val_fraction`` (same contract as assign_splits).
    """
    candidate_limit = candidate_limit or max_tasks
    tasks: List[TaskRecord] = []
    if llm_miner is not None:
        try:
            tasks = llm_miner(digests) or []
        except CursorBackendError:
            raise
        except Exception:
            tasks = []
    if not tasks:
        tasks = heuristic_mine(digests, max_tasks=candidate_limit)
    tasks = dedup_tasks(tasks)
    if target_skill_text or target_skill_path:
        tasks = filter_tasks_for_target(tasks, target_skill_text, target_skill_path)
    tasks = tasks[:max_tasks]
    tasks = assign_splits(
        tasks,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    return tasks

"""SkillOpt-Sleep — dream + associative recall for nightly consolidation.

Two opt-in mechanisms (both default OFF, so the cycle is unchanged unless the
user enables them) that the deployment experiments validated:

  * dream rollouts  — run each task K times and learn from the good-vs-bad
    contrast (set ``dream_rollouts > 1``). Stronger signal than one failure.
  * associative recall — each night, pull the K past tasks most similar to
    tonight's new ones into the dream (set ``recall_k > 0``). Replays relevant
    experience without re-running the whole history.

``dream_consolidate`` wires recall + synthetic augmentation + multi-rollout
consolidation and is called by BOTH the shipped plugin cycle and the benchmark
experiment harness, so the reported numbers exercise the exact code the plugin
runs. Pure-stdlib, zero research/private dependency.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

from skillopt_sleep.consolidate import ConsolidationResult, consolidate
from skillopt_sleep.types import TaskRecord

GenerateFn = Callable[[str], str]

# ── synthetic augmentation ("dream up" variants of today's tasks) ─────────────

_WRAPPERS = [
    "(quick one) {q}",
    "Please handle this request: {q}",
    "For the daily report: {q}",
]


def _template_intent(task: TaskRecord, k: int) -> str:
    return _WRAPPERS[k % len(_WRAPPERS)].format(q=task.intent)


def _parse_paraphrases(raw: str, n: int) -> List[str]:
    """Accept a JSON array of strings; drop empty / short / non-string items."""
    from skillopt_sleep.backend import _extract_json
    parsed = _extract_json(raw or "", "array")
    if not isinstance(parsed, list):
        return []
    out: List[str] = []
    for item in parsed:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if len(text) < 8:
            continue
        out.append(text)
        if len(out) >= n:
            break
    return out


def _fidelity_ok(original: str, paraphrase: str) -> bool:
    """Paraphrase-only v1: keep the parent's constraints by construction.

    We copy reference/judge unchanged, so a constraint-changing rewrite would
    mislabel the variant. Refuse empty, identical, and prompt-echo strings.
    Constraint perturbations are deferred to a later redesign of judge
    propagation.
    """
    text = (paraphrase or "").strip()
    src = (original or "").strip()
    if len(text) < 8 or not src:
        return False
    if text == src:
        return False
    if "Return ONLY a JSON array" in text:
        return False
    return True


def _dream_record(task: TaskRecord, k: int, intent: str, extra_tags: Optional[List[str]] = None) -> TaskRecord:
    tags = list(task.tags) + ["dream"]
    if extra_tags:
        tags.extend(extra_tags)
    return TaskRecord(
        id=f"{task.id}_dream{k}", project=task.project,
        intent=intent, context_excerpt=task.context_excerpt,
        reference_kind=task.reference_kind, reference=task.reference,
        judge=dict(task.judge), system=task.system,
        tags=tags, split="train",
        origin="dream", derived_from=task.id,
        skill_hint=task.skill_hint,
    )


def dream_augment(
    real_tasks: List[TaskRecord],
    *,
    factor: int = 1,
    llm_dream: bool = False,
    generate_fn: Optional[GenerateFn] = None,
    evidence=None,
) -> List[TaskRecord]:
    """Create synthetic TRAIN variants of real tasks (origin='dream').

    Default path is a light, deterministic rephrasing. Dream tasks are
    training-only: they carry split='train' and never enter the val/test
    slices the gate scores on.

    Opt-in ``llm_dream=True`` asks ``generate_fn`` for paraphrase-only
    rewrites (parent reference/judge copied unchanged). Any parse or
    fidelity failure falls back to the same wrappers as the default path,
    so a night can degrade but not break. Template mode (the default) is
    byte-identical to the pre-llm_dream implementation.
    """
    out: List[TaskRecord] = []
    use_llm = bool(llm_dream) and generate_fn is not None
    if llm_dream and generate_fn is None and evidence is not None:
        evidence.log(
            "dream", "llm_dream_fallback",
            reason="no_generate_fn", n_requested=max(0, factor),
        )
    for t in real_tasks:
        parsed: List[str] = []
        if use_llm:
            try:
                from skillopt_sleep import prompts as prompt_registry
                prompt = prompt_registry.render("llm_dream", {
                    "__INTENT__": t.intent,
                    "__N__": str(max(0, factor)),
                    "__CONTEXT__": (t.context_excerpt or "")[:400],
                })
                parsed = _parse_paraphrases(generate_fn(prompt), max(0, factor))
            except Exception:
                parsed = []
        n_ok = 0
        for k in range(max(0, factor)):
            extra: Optional[List[str]] = None
            if use_llm and k < len(parsed) and _fidelity_ok(t.intent, parsed[k]):
                intent = parsed[k]
                extra = ["llm_dream"]
                n_ok += 1
            else:
                intent = _template_intent(t, k)
            out.append(_dream_record(t, k, intent, extra))
        if use_llm and n_ok < max(0, factor) and evidence is not None:
            evidence.log(
                "dream", "llm_dream_fallback",
                task_id=t.id,
                n_fallback=max(0, factor) - n_ok,
                n_requested=max(0, factor),
            )
    return out


def backend_generate_fn(backend) -> GenerateFn:
    """Adapter: reuse Backend.attempt so every backend can paraphrase.

    The probe task is never added to the training pool.
    """
    def generate(prompt: str) -> str:
        probe = TaskRecord(
            id="__llm_dream_probe__",
            project="",
            intent=prompt,
            reference_kind="none",
        )
        return backend.attempt(probe, skill="", memory="")
    return generate


# ── associative recall (experience replay of similar past tasks) ──────────────

def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2}


def _normalize_split(value: str) -> str:
    return {"replay": "train", "holdout": "val"}.get(value, value)


def recall_similar(
    new_tasks: List[TaskRecord],
    history: List[TaskRecord],
    k: int,
    *,
    exclude_ids: Optional[set[str]] = None,
) -> List[TaskRecord]:
    """Return the ``k`` historical tasks most lexically similar to any of
    tonight's ``new_tasks`` (max Jaccard token overlap). Recalled tasks are
    returned as training material (split='train'); deterministic, stdlib-only.

    Archived val/test tasks are never recalled, and ``exclude_ids`` blocks
    tonight's held-out ids (and their ``derived_from`` sources) from re-entering
    the training pool.
    """
    if not history or k <= 0 or not new_tasks:
        return []
    blocked = set(exclude_ids or ())
    for t in new_tasks:
        blocked.add(t.id)
        if t.derived_from:
            blocked.add(t.derived_from)
    new_tok = [_tokens(t.intent) for t in new_tasks]
    scored = []
    for h in history:
        if h.id in blocked:
            continue
        if _normalize_split(h.split) in ("val", "test"):
            continue
        ht = _tokens(h.intent)
        if not ht:
            continue
        sim = max(((len(ht & nt) / len(ht | nt)) if (ht | nt) else 0.0) for nt in new_tok)
        scored.append((sim, h.id, h))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for sim, _id, h in scored[:max(0, k)]:
        if sim <= 0.0:
            break
        # recall as training material; copy so the source archive is untouched
        out.append(TaskRecord(
            id=f"recall:{h.id}", project=h.project, intent=h.intent,
            context_excerpt=h.context_excerpt, reference_kind=h.reference_kind,
            reference=h.reference, judge=dict(h.judge), system=h.system,
            tags=list(h.tags) + ["recall"], split="train", origin="real",
            derived_from=h.id,
            skill_hint=h.skill_hint,
        ))
    return out


# ── the shared nightly consolidation step ─────────────────────────────────────

def dream_consolidate(
    backend,
    tasks: List[TaskRecord],
    skill: str,
    memory: str,
    *,
    history_tasks: Optional[List[TaskRecord]] = None,
    recall_k: int = 0,
    dream_rollouts: int = 1,
    dream_factor: int = 0,
    edit_budget: int = 4,
    gate_metric: str = "mixed",
    gate_mixed_weight: float = 0.5,
    gate_no_regression: bool = False,
    gate_mode: str = "on",
    evolve_skill: bool = True,
    evolve_memory: bool = True,
    night: int = 1,
    llm_dream: bool = False,
    generate_fn: Optional[GenerateFn] = None,
    evidence=None,
) -> ConsolidationResult:
    """Recall similar past experience + dream synthetic variants, then run one
    gated consolidation epoch over the enlarged training pool.

    ``tasks`` is the split-tagged pool for tonight (train + val); recall and
    augmentation only enlarge the TRAIN split, so the val slice the gate scores
    on is never polluted. With ``recall_k=0`` and ``dream_rollouts=1`` (the
    defaults) this is exactly the previous single-shot ``consolidate``.
    """
    train = [t for t in tasks if t.split == "train"]
    enlarged = list(tasks)
    if recall_k > 0 and history_tasks:
        held_out_ids = {
            t.id for t in tasks if _normalize_split(t.split) in ("val", "test")
        }
        for t in tasks:
            if t.derived_from:
                held_out_ids.add(t.derived_from)
        enlarged += recall_similar(
            train, history_tasks, recall_k, exclude_ids=held_out_ids,
        )
    if dream_factor > 0:
        seed = [t for t in enlarged if t.split == "train" and t.origin != "dream"]
        enlarged += dream_augment(
            seed,
            factor=dream_factor,
            llm_dream=llm_dream,
            generate_fn=generate_fn,
            evidence=evidence,
        )
    return consolidate(
        backend, enlarged, skill, memory,
        edit_budget=edit_budget, gate_metric=gate_metric,
        gate_mixed_weight=gate_mixed_weight,
        gate_no_regression=gate_no_regression, gate_mode=gate_mode,
        rollouts_k=dream_rollouts, evolve_skill=evolve_skill,
        evolve_memory=evolve_memory, night=night,
    )

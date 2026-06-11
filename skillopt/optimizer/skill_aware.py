"""Skill-Aware Reflection — analyst prompt augmentation (EmbodiSkill).

When ``use_skill_aware_reflection`` is enabled, the failure/success analysts are
asked to additionally classify each reflection by EmbodiSkill type and to route
**EXECUTION_LAPSE** reflections (the skill rule is correct, the executor just
failed to follow it) into a separate ``appendix_notes`` list instead of the body
patch. This module owns:

1. the instruction text appended to the resolved analyst system prompt, and
2. extraction of ``appendix_notes`` from the analyst JSON response.

Design notes
------------
- The suffix is appended **at runtime, gated by the toggle**, so env-specific and
  generic analyst prompts are augmented uniformly and — when the toggle is off —
  remain byte-identical to baseline.
- Discrimination follows the paper / GMemory: ``SKILL_DEFECT`` = the skill rule is
  wrong / missing / underspecified (→ body edit); ``EXECUTION_LAPSE`` = the rule
  is valid but the agent didn't follow it (→ appendix reminder, body untouched).
  **When unsure, default to EXECUTION_LAPSE** (protect the body — never delete a
  valid rule over a one-off execution slip).
- Success reflections are labeled DISCOVERY / OPTIMIZATION for logging only; their
  edit behavior is unchanged.
"""
from __future__ import annotations


# ── Runtime switch (config-driven, env-independent) ─────────────────────────
#
# The trainer calls :func:`configure_skill_aware_reflection` once at startup
# from the resolved config. ``run_minibatch_reflect`` then picks these values
# up automatically, so env adapters never need to thread the toggle through —
# the feature is controlled purely by ``optimizer.use_skill_aware_reflection``
# regardless of benchmark. Mirrors the ``configure_azure_openai`` pattern in
# :mod:`skillopt.model`. Explicit kwargs at a call site still take precedence
# (backward compatible).

_RUNTIME: dict = {"enabled": False, "appendix_source": "both"}


def configure_skill_aware_reflection(
    enabled: bool,
    appendix_source: str = "both",
) -> None:
    """Set the process-wide skill-aware reflection switch from config."""
    _RUNTIME["enabled"] = bool(enabled)
    _RUNTIME["appendix_source"] = str(appendix_source or "both")


def is_skill_aware_enabled() -> bool:
    return bool(_RUNTIME["enabled"])


def get_skill_aware_appendix_source() -> str:
    return str(_RUNTIME["appendix_source"])


# ── Prompt suffixes ─────────────────────────────────────────────────────────

# Appended to the FAILURE analyst system prompt when the toggle is on.
ERROR_SUFFIX = """

## Skill-Aware Reflection (EmbodiSkill)

Before proposing body edits, classify EACH failure pattern as one of:

- **SKILL_DEFECT**: the current skill is wrong, missing, or underspecified for
  this situation — i.e. an agent that *followed the skill* would still fail, or
  the skill gives no relevant guidance. These become normal body `edits`.
- **EXECUTION_LAPSE**: the skill ALREADY contains a relevant, correct rule that
  would have avoided the failure, but the agent did not follow it (e.g. ignored a
  rule, malformed output, copied the feedback text verbatim, emitted a non-action
  token like "stop", or otherwise broke execution unrelated to skill content).

Discrimination test: "Is there a rule in the current skill that, if followed,
prevents this failure?" If yes → EXECUTION_LAPSE. If no (rule absent/wrong) →
SKILL_DEFECT. **When genuinely unsure, choose EXECUTION_LAPSE** — do not edit or
delete a valid rule over a one-off execution slip.

Routing:
- SKILL_DEFECT → put the fix in `patch.edits` (body), as usual.
- EXECUTION_LAPSE → put a concise reminder in `appendix_notes` (a flat list of
  strings). DO NOT add a body edit for it. Each note should re-emphasize the
  existing valid rule the agent failed to follow; it must NOT introduce a new
  rule. Keep notes short, concrete, and reusable.

Add `appendix_notes` as a TOP-LEVEL key of your JSON output (a sibling of
`patch`), e.g. `"appendix_notes": ["Follow the existing X rule before Y."]`.
Use `[]` when there is no execution lapse. Body edits and appendix notes are
independent: a batch may yield only edits, only notes, both, or neither.
"""

# Appended to the SUCCESS analyst system prompt when the toggle is on.
SUCCESS_SUFFIX = """

## Skill-Aware Reflection (EmbodiSkill)

For each proposed edit, optionally label its `reflection_type` for logging:
- **DISCOVERY**: a useful new rule not yet in the skill (typically an `append`).
- **OPTIMIZATION**: a better way to perform an existing rule (typically a
  `replace` of that rule).

This labeling does not change edit behavior. You may also add a top-level
`appendix_notes` list (flat strings) if a successful trajectory reveals an
existing valid rule worth re-emphasizing; otherwise use `[]`.
"""


def augment_error_prompt(system_prompt: str) -> str:
    """Append the failure-analyst skill-aware instruction."""
    return system_prompt.rstrip() + "\n" + ERROR_SUFFIX


def augment_success_prompt(system_prompt: str) -> str:
    """Append the success-analyst skill-aware instruction."""
    return system_prompt.rstrip() + "\n" + SUCCESS_SUFFIX


# ── Response parsing ────────────────────────────────────────────────────────


def extract_appendix_notes(result: dict | None) -> list[str]:
    """Pull a clean list of appendix-note strings from an analyst JSON result.

    Tolerant of shape: accepts a top-level ``appendix_notes`` list, a single
    string, or items wrapped in dicts with a ``note``/``content`` field. Returns
    ``[]`` for anything missing or malformed (so a non-compliant model degrades
    gracefully to baseline body-only behavior).
    """
    if not isinstance(result, dict):
        return []
    raw = result.get("appendix_notes")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    notes: list[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("note") or item.get("content") or "").strip()
        else:
            text = ""
        if text:
            notes.append(text)
    return notes


# ── Appendix consolidation (threshold-gated, paper Eq.11 UpdateSkillAppendix) ──

_CONSOLIDATE_SYSTEM = (
    "You compact the Execution Notes Appendix of an agent skill. Each note "
    "re-emphasizes an existing skill rule the agent failed to follow. Your job "
    "is a periodic compaction pass: remove duplicates and redundant overlap, "
    "merge near-identical reminders into one, and simplify phrasing while keeping "
    "each note concrete and operational. Do not invent new rules. Preserve the "
    "distinct actionable content. Return valid JSON only."
)


def consolidate_appendix_notes(
    notes: list[str],
    *,
    chat_fn,
    max_completion_tokens: int = 4096,
) -> list[str]:
    """LLM-consolidate appendix notes: dedupe / merge / compact.

    Mirrors GMemory ``_maybe_refactor_execution_notes`` and paper Eq.11. ``chat_fn``
    is the optimizer chat callable ``(system, user, max_completion_tokens, retries,
    stage) -> (text, meta)``. On ANY failure (parse, empty, exception) the original
    notes are returned unchanged, so consolidation can never lose the appendix.
    """
    from skillopt.utils import extract_json  # local import to avoid cycles

    clean = [str(n).strip() for n in (notes or []) if str(n).strip()]
    if len(clean) < 2:
        return clean

    numbered = "\n".join(f"{i}. {n}" for i, n in enumerate(clean, 1))
    user = (
        f"## Current Execution Notes ({len(clean)} total)\n{numbered}\n\n"
        "Compact these into a shorter list without losing distinct actionable "
        "information. Merge duplicates and near-duplicates; keep each note short, "
        "concrete, and reusable. Return valid JSON only with this schema:\n"
        '{ "appendix_notes": ["compacted note 1", "compacted note 2"] }'
    )
    try:
        response, _ = chat_fn(
            system=_CONSOLIDATE_SYSTEM,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=2,
            stage="appendix_consolidate",
        )
        result = extract_json(response)
        compacted = extract_appendix_notes(result)
        # Guard: only accept a non-empty result that actually shrinks the set.
        if compacted and len(compacted) <= len(clean):
            return compacted
    except Exception:  # noqa: BLE001
        pass
    return clean

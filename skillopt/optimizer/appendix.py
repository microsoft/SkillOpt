"""Skill-Aware Reflection — protected appendix field (EmbodiSkill S_app).

EmbodiSkill (paper 2605.10332v1) splits a skill into ``S = (S_body, S_app)``:
the body holds the main prescriptive rules; the appendix only *emphasizes*
existing valid rules that the executor failed to follow (EXECUTION_LAPSE), and
**never introduces new rules**.

This module owns the appendix region of the skill document. It mirrors the
protected-field pattern of :mod:`skillopt.optimizer.slow_update`, with two
differences:

1. **Append semantics** (not replace): execution-lapse reminders accumulate
   across steps within a run, so new notes are merged into the existing
   appendix rather than overwriting it.
2. **Lightweight dedup**: near-duplicate reminders are collapsed (inspired by
   GMemory's ``_dedupe_preserve_order``) so the appendix stays compact.

The appendix lives **inside** the skill markdown, between dedicated markers, so
it is persisted by the normal ``_save_skill`` path and is resume-safe. Step-level
analyst edits cannot modify it (enforced by the shared protected-region check in
:mod:`skillopt.optimizer.skill`).

Public API
----------
- :func:`has_appendix_field`        — check if markers are present
- :func:`inject_empty_appendix_field` — add empty placeholder (skill init)
- :func:`extract_appendix_notes`    — read current notes as a list
- :func:`append_to_appendix_field`  — merge new notes (dedup) into the region
"""
from __future__ import annotations

import re

# ── Protected field markers ─────────────────────────────────────────────────

APPENDIX_START = "<!-- APPENDIX_START -->"
APPENDIX_END = "<!-- APPENDIX_END -->"

# Heading shown inside the rendered appendix block (human-readable only).
APPENDIX_HEADING = "## Execution Notes Appendix"

# Each note is rendered as a markdown bullet so the target model reads it as
# ordinary guidance.
_NOTE_BULLET_PREFIX = "- "


# ── Dedup helpers ───────────────────────────────────────────────────────────


def _canonicalize(text: str) -> str:
    """Normalize a note for duplicate detection (whitespace/punct/case-insensitive)."""
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    normalized = normalized.rstrip(" .;:,_-")
    return normalized.casefold()


def _dedupe_preserve_order(notes: list[str]) -> list[str]:
    """Drop blanks and near-duplicates, preserving first-seen order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for note in notes:
        text = re.sub(r"\s+", " ", str(note).strip())
        if not text:
            continue
        key = _canonicalize(text)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


# ── Field manipulation ──────────────────────────────────────────────────────


def has_appendix_field(skill: str) -> bool:
    return APPENDIX_START in skill and APPENDIX_END in skill


def _render_block(notes: list[str]) -> str:
    """Render the full marker-delimited appendix block for *notes*."""
    lines = [APPENDIX_START, APPENDIX_HEADING]
    for note in notes:
        lines.append(f"{_NOTE_BULLET_PREFIX}{note}")
    lines.append(APPENDIX_END)
    return "\n".join(lines)


def inject_empty_appendix_field(skill: str) -> str:
    """Add an empty appendix placeholder at the end of *skill* (idempotent).

    Mirrors ``inject_empty_slow_update_field``: called once at skill init so the
    protected region exists before any note is written.
    """
    if has_appendix_field(skill):
        return skill
    block = f"\n\n{APPENDIX_START}\n{APPENDIX_HEADING}\n{APPENDIX_END}\n"
    return skill.rstrip() + block


def extract_appendix_notes(skill: str) -> list[str]:
    """Return the current appendix notes as a list of strings (no markers/heading)."""
    start = skill.find(APPENDIX_START)
    end = skill.find(APPENDIX_END)
    if start == -1 or end == -1:
        return []
    inner = skill[start + len(APPENDIX_START):end].strip()
    notes: list[str] = []
    for raw_line in inner.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == APPENDIX_HEADING or line.lstrip("#").strip() == APPENDIX_HEADING.lstrip("#").strip():
            continue
        if line.startswith(_NOTE_BULLET_PREFIX):
            line = line[len(_NOTE_BULLET_PREFIX):].strip()
        elif line.startswith("-") or line.startswith("*"):
            line = line[1:].strip()
        if line:
            notes.append(line)
    return notes


def _strip_all_appendix_fields(skill: str) -> str:
    """Remove every appendix marker pair (and content between) from *skill*."""
    while True:
        start = skill.find(APPENDIX_START)
        if start == -1:
            break
        end = skill.find(APPENDIX_END, start)
        if end == -1:
            skill = skill[:start] + skill[start + len(APPENDIX_START):]
            break
        skill = skill[:end + len(APPENDIX_END)].rsplit(APPENDIX_START, 1)[0] + skill[end + len(APPENDIX_END):]
    skill = skill.replace(APPENDIX_END, "")
    while "\n\n\n" in skill:
        skill = skill.replace("\n\n\n", "\n\n")
    return skill.rstrip()


def append_to_appendix_field(skill: str, new_notes: list[str]) -> str:
    """Merge *new_notes* into the appendix region (dedup), returning updated skill.

    - If no appendix region exists yet, one is created.
    - Existing notes are preserved; new ones are appended after dedup against the
      combined set, so order is stable and duplicates are dropped.
    - Empty / whitespace-only notes are ignored. If the merged set is empty, an
      empty placeholder region is still ensured.
    """
    incoming = _dedupe_preserve_order(list(new_notes or []))
    existing = extract_appendix_notes(skill)
    merged = _dedupe_preserve_order(existing + incoming)

    base = _strip_all_appendix_fields(skill)
    block = _render_block(merged)
    return f"{base}\n\n{block}\n"

"""ReflACT skill operations — edit application and patch processing.

The Update stage (⑤) of the ReflACT pipeline: apply a ranked set of
edits to the current skill document, producing an updated candidate.
Analogous to optimizer.step() in neural network training.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillopt.types import Edit as EditType, Patch as PatchType

SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"

# Skill-aware reflection (EmbodiSkill S_app) appendix region. Like the slow
# update region, it is protected: step-level analyst edits must not modify it.
APPENDIX_START = "<!-- APPENDIX_START -->"
APPENDIX_END = "<!-- APPENDIX_END -->"

# All protected (start, end) marker pairs. Step-level edits cannot target text
# inside any of these regions, and `append` / `insert_after`-fallback ops are
# inserted before the earliest-occurring region so protected blocks stay at the
# document tail. With only the slow-update region present, every helper reduces
# to the original slow-update-only behavior (byte-identical skill output).
_PROTECTED_REGIONS: tuple[tuple[str, str], ...] = (
    (SLOW_UPDATE_START, SLOW_UPDATE_END),
    (APPENDIX_START, APPENDIX_END),
)


def _earliest_protected_start(skill: str) -> int:
    """Index of the earliest protected-region start marker, or -1 if none."""
    positions = [
        idx
        for idx in (skill.find(start) for start, _ in _PROTECTED_REGIONS)
        if idx != -1
    ]
    return min(positions) if positions else -1


def _is_in_protected_region(skill: str, target: str) -> bool:
    """Check if *target* text falls within any protected region."""
    if not target:
        return False
    target_idx = skill.find(target)
    if target_idx == -1:
        return False
    for start_marker, end_marker in _PROTECTED_REGIONS:
        start_idx = skill.find(start_marker)
        end_idx = skill.find(end_marker)
        if start_idx == -1 or end_idx == -1:
            continue
        region_end = end_idx + len(end_marker)
        if start_idx <= target_idx < region_end:
            return True
    return False


def _is_in_slow_update_region(skill: str, target: str) -> bool:
    """Backward-compatible alias kept for any external callers/tests."""
    return _is_in_protected_region(skill, target)


def _strip_slow_update_markers(text: str) -> str:
    """Remove any protected-region markers from edit content to prevent duplication."""
    return (
        text.replace(SLOW_UPDATE_START, "")
            .replace(SLOW_UPDATE_END, "")
            .replace(APPENDIX_START, "")
            .replace(APPENDIX_END, "")
    )


def _edit_fields(edit: EditType | dict) -> tuple[str, str, str]:
    op = edit.op if hasattr(edit, "op") else edit.get("op", "")
    content = _strip_slow_update_markers(
        (edit.content if hasattr(edit, "content") else edit.get("content", "")).strip()
    )
    target = edit.target if hasattr(edit, "target") else edit.get("target", "")
    return op, content, target


def _apply_edit_with_report(skill: str, edit: EditType | dict) -> tuple[str, dict]:
    op, content, target = _edit_fields(edit)
    report = {
        "op": op,
        "target": target[:200],
        "content_preview": content[:200],
        "status": "unknown",
    }

    if target and _is_in_protected_region(skill, target):
        report["status"] = "skipped_protected_region"
        return skill, report

    if op == "append":
        prot_start = _earliest_protected_start(skill)
        if prot_start != -1:
            before = skill[:prot_start].rstrip()
            after = skill[prot_start:]
            report["status"] = "applied_append_before_protected_region"
            return before + "\n\n" + content + "\n\n" + after, report
        report["status"] = "applied_append"
        return skill.rstrip() + "\n\n" + content + "\n", report

    if op == "insert_after":
        if not target or target not in skill:
            prot_start = _earliest_protected_start(skill)
            if prot_start != -1:
                before = skill[:prot_start].rstrip()
                after = skill[prot_start:]
                report["status"] = "applied_insert_after_fallback_before_protected_region"
                return before + "\n\n" + content + "\n\n" + after, report
            report["status"] = "applied_insert_after_fallback_append"
            return skill.rstrip() + "\n\n" + content + "\n", report
        idx = skill.index(target) + len(target)
        newline = skill.find("\n", idx)
        insert_at = newline + 1 if newline != -1 else len(skill)
        report["status"] = "applied_insert_after"
        return skill[:insert_at] + "\n" + content + "\n" + skill[insert_at:], report

    if op == "replace":
        if not target:
            report["status"] = "skipped_replace_missing_target"
            return skill, report
        if target not in skill:
            report["status"] = "skipped_replace_target_not_found"
            return skill, report
        report["status"] = "applied_replace"
        return skill.replace(target, content, 1), report

    if op == "delete":
        if not target:
            report["status"] = "skipped_delete_missing_target"
            return skill, report
        if target not in skill:
            report["status"] = "skipped_delete_target_not_found"
            return skill, report
        report["status"] = "applied_delete"
        return skill.replace(target, "", 1), report

    report["status"] = "skipped_unknown_op"
    return skill, report


def apply_edit(skill: str, edit: EditType | dict) -> str:
    """Apply a single edit operation to the skill document.

    Parameters
    ----------
    skill : str
        Current skill document content.
    edit : Edit | dict
        An :class:`~skillopt.types.Edit` instance or a plain dict with
        keys ``op``, ``content``, ``target``.

    Edits targeting the protected slow-update region are silently skipped.
    """
    updated_skill, _ = _apply_edit_with_report(skill, edit)
    return updated_skill


def apply_patch_with_report(
    skill: str,
    patch: PatchType | dict,
) -> tuple[str, list[dict]]:
    """Apply a patch and return a per-edit report for observability."""
    edits = patch.edits if hasattr(patch, "edits") else patch.get("edits", [])
    reports: list[dict] = []
    for idx, edit in enumerate(edits, 1):
        try:
            skill, report = _apply_edit_with_report(skill, edit)
            report["index"] = idx
        except Exception as exc:  # noqa: BLE001
            report = {
                "index": idx,
                "op": "",
                "target": "",
                "content_preview": "",
                "status": "error",
                "error": str(exc),
            }
        reports.append(report)
    return skill, reports


def apply_patch(skill: str, patch: PatchType | dict) -> str:
    """Apply a patch (list of edits) to the skill document sequentially.

    Parameters
    ----------
    skill : str
        Current skill document content.
    patch : Patch | dict
        A :class:`~skillopt.types.Patch` instance or a plain dict with
        key ``edits`` containing a list of edit operations.
    """
    updated_skill, _ = apply_patch_with_report(skill, patch)
    return updated_skill

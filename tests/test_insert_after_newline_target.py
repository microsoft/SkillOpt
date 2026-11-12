"""Regression: ``insert_after`` with a target that ends in a newline.

The insertion point is the end of the target's line. When the target already
ends with ``\\n``, the search for the next newline found the end of the
*following* line, so the content landed one line too late: under the next
heading, or inside a protected region whose start marker follows directly.
"""
from __future__ import annotations

from skillopt.optimizer.skill import apply_patch_with_report


def _insert_after(skill: str, target: str) -> tuple[str, str]:
    new, rep = apply_patch_with_report(
        skill,
        {"edits": [{"op": "insert_after", "target": target, "content": "- Cite the source."}]},
    )
    return new, rep[0]["status"]


def test_target_ending_in_newline_inserts_right_after_it() -> None:
    skill = "# QA Skill\n\n## Rules\n## Output format\n- One line.\n"
    new, status = _insert_after(skill, "## Rules\n")
    assert status == "applied_insert_after"
    assert new == "# QA Skill\n\n## Rules\n\n- Cite the source.\n## Output format\n- One line.\n"


def test_target_ending_in_newline_matches_target_without_it() -> None:
    skill = "# QA Skill\n\n## Rules\n## Output format\n- One line.\n"
    with_newline, _ = _insert_after(skill, "## Rules\n")
    without_newline, _ = _insert_after(skill, "## Rules")
    assert with_newline == without_newline


def test_insert_does_not_land_inside_following_protected_region() -> None:
    skill = (
        "# QA Skill\n- Answer briefly.\n"
        "<!-- SLOW_UPDATE_START -->\nPrefer exact spans.\n<!-- SLOW_UPDATE_END -->\n"
    )
    new, status = _insert_after(skill, "- Answer briefly.\n")
    assert status == "applied_insert_after"
    start = new.index("<!-- SLOW_UPDATE_START -->")
    assert new.index("- Cite the source.") < start
    assert new[start:] == skill[skill.index("<!-- SLOW_UPDATE_START -->"):]

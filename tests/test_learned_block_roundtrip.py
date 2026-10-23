"""Learned items survive the read -> apply -> write round-trip intact.

The learned block is read back one ``- `` bullet per item, but items were
written verbatim. A multi-line item therefore lost every continuation line the
next time any edit was applied, a ``- X`` add slipped past the duplicate check
for an existing ``X``, and ``lstrip('- ')`` ate leading dashes such as
``--force``. Losing adopted text changes what the gate scores: a good edit was
rejected because applying it silently deleted an adopted rule.
"""
from __future__ import annotations

from skillopt_sleep.backend import MockBackend
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.memory import (
    LEARNED_END,
    LEARNED_START,
    apply_edits_detailed,
    current_learned_lines,
    extract_learned,
    set_learned,
)
from skillopt_sleep.types import EditRecord, TaskRecord


def _legacy_doc(*items: str) -> str:
    """A learned block exactly as earlier versions wrote it: items verbatim."""
    body = "\n".join(f"- {item}" for item in items)
    return (
        f"# Skill\n\n{LEARNED_START}\n## Learned preferences & procedures\n\n"
        f"_banner_\n\n{body}\n{LEARNED_END}\n"
    )


def _add(content: str) -> EditRecord:
    return EditRecord(target="skill", op="add", content=content)


def test_legacy_multi_line_item_survives_the_next_edit() -> None:
    doc = _legacy_doc(
        "Always run tests before committing",
        "When editing SQL:\n1. run the migration dry-run\n2. check row counts",
    )
    new_doc, applied, unmatched = apply_edits_detailed(doc, [_add("Prefer small commits")])
    assert len(applied) == 1 and unmatched == []
    assert current_learned_lines(new_doc) == [
        "Always run tests before committing",
        "When editing SQL: 1. run the migration dry-run 2. check row counts",
        "Prefer small commits",
    ]
    assert "check row counts" in extract_learned(new_doc)


def test_multi_line_add_is_stored_as_one_item_and_round_trips() -> None:
    doc, applied, _ = apply_edits_detailed(
        "# Skill\n", [_add("When editing SQL:\n1. dry-run first\n2. check row counts")]
    )
    assert len(applied) == 1
    item = "When editing SQL: 1. dry-run first 2. check row counts"
    assert current_learned_lines(doc) == [item]
    again, applied, unmatched = apply_edits_detailed(doc, [_add(item)])
    assert applied == [] and len(unmatched) == 1
    assert again == doc


def test_bulleted_add_is_a_duplicate_of_the_existing_item() -> None:
    doc = set_learned("# Skill\n", ["Always run tests before committing"])
    _new, applied, unmatched = apply_edits_detailed(
        doc, [_add("- Always run tests before committing")]
    )
    assert applied == [] and len(unmatched) == 1


def test_leading_dashes_in_rule_text_are_preserved() -> None:
    doc, _applied, _ = apply_edits_detailed(
        "# Skill\n", [_add("--force-with-lease is required instead of --force")]
    )
    assert current_learned_lines(doc) == [
        "--force-with-lease is required instead of --force"
    ]
    replaced, applied, _ = apply_edits_detailed(
        doc,
        [EditRecord(target="skill", op="replace", anchor="force-with-lease",
                    content="- --dry-run before every push")],
    )
    assert len(applied) == 1
    assert current_learned_lines(replaced) == ["--dry-run before every push"]


def test_gate_accepts_a_good_edit_on_a_skill_with_a_legacy_multi_line_item() -> None:
    """Applying tonight's helpful edit must not silently drop adopted text.

    Before the fix, the adopted units rule lived on a continuation line, so the
    candidate lost it and scored 0.5 against a 0.5 baseline: rejected.
    """
    units = MockBackend.RULE_TEXT["units-si"]

    def task(task_id, rule, split, reference):
        return TaskRecord(
            id=task_id, project="/project", intent=f"task {task_id} please",
            reference_kind="exact", reference=reference,
            tags=[f"rule:{rule}"], split=split,
        )

    tasks = [
        task("tr1", "wrap-answer", "train", "42"),
        task("va1", "units-si", "val", "9.8 m/s^2"),
        task("va2", "wrap-answer", "val", "7"),
    ]
    skill = _legacy_doc("Numeric answers:\n" + units)
    result = consolidate(MockBackend(), tasks, skill, "", evolve_memory=False, gate_metric="hard")
    assert result.accepted is True
    assert result.baseline_score == 0.5
    assert result.candidate_score == 1.0
    assert units in result.new_skill

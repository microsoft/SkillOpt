"""A replace edit never writes the same learned item twice.

``add`` skips an item that is already in the learned block, but ``replace``
wrote its replacement once per matching line and did not check the other
lines, so the block could hold the same bullet twice.
"""

from __future__ import annotations

from skillopt_sleep.memory import apply_edits_detailed, current_learned_lines, set_learned
from skillopt_sleep.types import EditRecord


def _replace(anchor: str, content: str) -> EditRecord:
    return EditRecord(target="skill", op="replace", anchor=anchor, content=content)


def test_replacement_that_already_exists_is_not_duplicated() -> None:
    doc = set_learned("", ["Run the tests before committing.", "Write tests for every bug fix."])

    new_doc, applied, unmatched = apply_edits_detailed(
        doc, [_replace("Run the tests before committing.", "Write tests for every bug fix.")]
    )

    assert current_learned_lines(new_doc) == ["Write tests for every bug fix."]
    assert len(applied) == 1 and unmatched == []


def test_anchor_matching_several_items_writes_the_replacement_once() -> None:
    doc = set_learned(
        "",
        ["Run the tests before committing.", "Prefer small commits.", "Write tests for every bug fix."],
    )

    new_doc, applied, _unmatched = apply_edits_detailed(
        doc, [_replace("tests", "Run the full test suite before committing.")]
    )

    assert current_learned_lines(new_doc) == [
        "Run the full test suite before committing.",
        "Prefer small commits.",
    ]
    assert len(applied) == 1


def test_replace_that_changes_nothing_is_still_unmatched() -> None:
    doc = set_learned("", ["Prefer small commits.", "Prefer small commits."])

    new_doc, applied, unmatched = apply_edits_detailed(doc, [_replace("no such rule", "Anything.")])

    assert current_learned_lines(new_doc) == ["Prefer small commits.", "Prefer small commits."]
    assert applied == [] and len(unmatched) == 1

"""Retry feedback for the SpreadsheetBench code-generation agent.

_build_eval_feedback turns the verification report into feedback for the
target model: it lists the wrong cells with the model's own values and hides
the expected ones. Sheet names may contain spaces and punctuation.
"""

from __future__ import annotations

import pytest

from skillopt.envs.spreadsheetbench.codegen_agent import _build_eval_feedback


def _report(*lines: str) -> str:
    return "\n".join(["## Output Verification", *(f"  {line}" for line in lines)])


@pytest.mark.parametrize("sheet", ["Sheet1", "Sales Data", "Q1 2024 (draft)", "Résumé-v2", "Data_2"])
def test_wrong_cell_is_listed_whatever_the_sheet_name(sheet):
    feedback = _build_eval_feedback(
        _report(f"{sheet}!D2: got=None, expected=420 ✗", f"{sheet}!D3: got=5, expected=5 ✓")
    )

    assert f"{sheet}!D2: your output = None (WRONG)" in feedback
    assert "(1 other cells are correct.)" in feedback


@pytest.mark.parametrize("cell", ["d2", "$D$2", "AA10"])
def test_cell_written_in_lowercase_or_absolute_form_is_listed(cell):
    feedback = _build_eval_feedback(_report(f"Sales Data!{cell}: got=None, expected=1 ✗"))

    assert f"Sales Data!{cell}: your output = None (WRONG)" in feedback


def test_expected_values_are_never_shown():
    feedback = _build_eval_feedback(
        _report(
            "Sales Data!A1: got='North, East', expected='North East' ✗",
            "Sales Data!B12: got=1.5, expected=1.25 ✗",
            "Sales Data!C3: got='ok', expected='ok' ✓",
        )
    )

    assert "Sales Data!A1: your output = 'North, East' (WRONG)" in feedback
    assert "Sales Data!B12: your output = 1.5 (WRONG)" in feedback
    assert "North East" not in feedback
    assert "1.25" not in feedback
    assert "expected" not in feedback


def test_lines_that_are_not_cell_results_are_ignored():
    feedback = _build_eval_feedback(_report("Sheet1!A1: got=1, expected=2 ✗", "(+3 empty cells correct, omitted)"))

    assert feedback.count("(WRONG)") == 1
    assert "other cells are correct" not in feedback

"""SpreadsheetBench scoring must not pass a workbook when no cell was compared."""

from __future__ import annotations

import openpyxl
import pytest

from skillopt.envs.spreadsheetbench.evaluator import evaluate


def _workbook(path, value) -> str:
    wb = openpyxl.Workbook()
    wb.active.title = "Sheet1"
    wb.active["A1"] = value
    wb.save(path)
    return str(path)


@pytest.mark.parametrize("answer_position", ["", "   ", ",", " , "])
def test_empty_answer_position_fails_closed(tmp_path, answer_position):
    gold = _workbook(tmp_path / "gold.xlsx", 1)
    wrong = _workbook(tmp_path / "pred.xlsx", 99)

    result = evaluate(wrong, gold, "Cell-Level Manipulation", answer_position)

    assert result["ok"] is False
    assert "answer position" in result["reason"]


@pytest.mark.parametrize(("value", "ok"), [(1, True), (99, False)])
def test_non_empty_answer_position_still_compares(tmp_path, value, ok):
    gold = _workbook(tmp_path / "gold.xlsx", 1)
    pred = _workbook(tmp_path / "pred.xlsx", value)

    assert evaluate(pred, gold, "Cell-Level Manipulation", "Sheet1!A1")["ok"] is ok
    assert evaluate(pred, gold, "Cell-Level Manipulation", " , Sheet1!A1")["ok"] is ok

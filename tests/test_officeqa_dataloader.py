"""OfficeQA split loading: gold answers that are numeric zero must survive."""

from __future__ import annotations

import json

import pytest

from skillopt.envs.officeqa.dataloader import OfficeQADataLoader
from skillopt.envs.officeqa.evaluator import evaluate


def _load(tmp_path, rows: list[dict]) -> list[dict]:
    split = tmp_path / "val"
    split.mkdir()
    (split / "items.json").write_text(json.dumps(rows), encoding="utf-8")
    return OfficeQADataLoader().load_split_items(str(split))


@pytest.mark.parametrize(("gold", "expected"), [(0, "0"), (0.0, "0"), ("0", "0")])
def test_zero_ground_truth_is_kept(tmp_path, gold, expected):
    (item,) = _load(tmp_path, [{"uid": "q1", "question": "How many bonds were issued?", "ground_truth": gold}])

    assert item["ground_truth"] == expected
    assert item["answers"] == [expected]
    assert evaluate(expected, item["ground_truth"])["em"] == 1.0


@pytest.mark.parametrize(("gold", "expected"), [(1776.0, "1776"), (12.5, "12.5"), (-3.0, "-3"), ("1776.0", "1776.0")])
def test_integral_float_ground_truth_loads_as_an_integer(tmp_path, gold, expected):
    # A float column written by e.g. pandas to_json stores 1776 as 1776.0.
    (item,) = _load(tmp_path, [{"uid": "q1", "question": "Year?", "ground_truth": gold}])

    assert item["ground_truth"] == expected
    assert evaluate(expected, item["ground_truth"])["em"] == 1.0


def test_numeric_zero_id_is_kept(tmp_path):
    items = _load(
        tmp_path,
        [
            {"uid": 0, "question": "Year?", "ground_truth": "1776"},
            {"id": 0, "question": "Year?", "ground_truth": "1789"},
        ],
    )

    assert [(item["id"], item["uid"]) for item in items] == [("0", "0"), ("0", "0")]


def test_answer_field_is_used_when_ground_truth_is_missing_or_empty(tmp_path):
    items = _load(
        tmp_path,
        [
            {"uid": "q1", "question": "Year?", "answer": "1776"},
            {"uid": "q2", "question": "Year?", "ground_truth": "", "answer": 0},
            {"uid": "q3", "question": "Year?", "ground_truth": None, "answer": "1789"},
        ],
    )

    assert [item["ground_truth"] for item in items] == ["1776", "0", "1789"]


def test_missing_gold_answer_stays_empty(tmp_path):
    (item,) = _load(tmp_path, [{"uid": "q1", "question": "Year?"}])

    assert item["ground_truth"] == ""
    assert item["answers"] == []

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


@pytest.mark.parametrize(("gold", "expected"), [(0, "0"), (0.0, "0.0"), ("0", "0")])
def test_zero_ground_truth_is_kept(tmp_path, gold, expected):
    (item,) = _load(tmp_path, [{"uid": "q1", "question": "How many bonds were issued?", "ground_truth": gold}])

    assert item["ground_truth"] == expected
    assert item["answers"] == [expected]
    assert evaluate(expected, item["ground_truth"])["em"] == 1.0


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

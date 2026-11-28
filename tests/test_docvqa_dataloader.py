"""DocVQA gold answers are read from the CSV as written.

The answer column holds either a list literal such as ``['1,234', '1234']``
or a single plain answer. A plain answer must not be evaluated as a Python
literal: ``1,234`` is a tuple to ``ast.literal_eval`` and ``0.50`` is a float.
"""

from __future__ import annotations

import csv

import pytest

from skillopt.envs.docvqa.dataloader import DocVQADataLoader, _parse_answers


@pytest.mark.parametrize(
    "raw",
    ["1,234", "2,000,000", "1,234.56", "0.50", "3.10", "1e3", "12", "-5", "Paris", "(555) 123-4567", "None"],
)
def test_plain_answer_is_kept_as_written(raw):
    assert _parse_answers(raw) == [raw]


def test_plain_answer_is_stripped():
    assert _parse_answers("  2,000,000 \n") == ["2,000,000"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("['1,234', '1234']", ["1,234", "1234"]),
        ('["0.50", " $3.10 "]', ["0.50", "$3.10"]),
        ("['Paris', '', '  ']", ["Paris"]),
        ("[]", []),
    ],
)
def test_list_literal_is_parsed_into_its_answers(raw, expected):
    assert _parse_answers(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_missing_answer_gives_no_answers(raw):
    assert _parse_answers(raw) == []


@pytest.mark.parametrize(("raw", "expected"), [("'Paris'", ["Paris"]), ('"1,234"', ["1,234"]), ("'90s", ["'90s"])])
def test_quoted_answer_is_unquoted(raw, expected):
    assert _parse_answers(raw) == expected


def test_malformed_list_literal_is_kept_as_text():
    assert _parse_answers("[1,234") == ["[1,234"]


def test_loader_reads_a_comma_number_answer_from_csv(tmp_path):
    split = tmp_path / "val"
    split.mkdir()
    with (split / "val.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["questionId", "question", "answer", "image_path"])
        writer.writeheader()
        writer.writerow(
            {"questionId": "1", "question": "What is the total?", "answer": "2,000,000", "image_path": "a.png"}
        )
        writer.writerow(
            {"questionId": "2", "question": "Which city?", "answer": "['Paris', 'paris']", "image_path": "b.png"}
        )

    items = DocVQADataLoader().load_split_items(str(split))

    assert items[0]["answers"] == ["2,000,000"]
    assert items[0]["answer"] == "2,000,000"
    assert items[1]["answers"] == ["Paris", "paris"]

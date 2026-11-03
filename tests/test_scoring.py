"""Tests for skillopt.utils.scoring."""
from __future__ import annotations

import pytest

from skillopt.utils.scoring import compute_score, skill_hash


class _ResultObject:
    """Minimal object with hard/soft attrs (duck-typing path)."""

    def __init__(self, hard: float, soft: float) -> None:
        self.hard = hard
        self.soft = soft


class TestComputeScore:
    """compute_score — hard/soft accuracy from a list of episode results."""

    def test_empty_list_returns_zeros(self) -> None:
        assert compute_score([]) == (0.0, 0.0)

    def test_dict_results_happy_path(self) -> None:
        results = [
            {"hard": 1, "soft": 0.8},
            {"hard": 0, "soft": 0.5},
            {"hard": 1, "soft": 0.9},
        ]
        hard, soft = compute_score(results)
        assert hard == pytest.approx(2 / 3)
        assert soft == pytest.approx((0.8 + 0.5 + 0.9) / 3)

    def test_object_results(self) -> None:
        results = [
            _ResultObject(1.0, 0.75),
            _ResultObject(0.0, 0.25),
        ]
        hard, soft = compute_score(results)
        assert hard == 0.5
        assert soft == 0.5

    def test_mixed_dict_and_object_results(self) -> None:
        results = [
            {"hard": 1, "soft": 1.0},
            _ResultObject(0, 0.0),
        ]
        hard, soft = compute_score(results)
        assert hard == 0.5
        assert soft == 0.5

    def test_missing_keys_default_to_zero(self) -> None:
        results = [
            {"hard": 1},
            {},
        ]
        hard, soft = compute_score(results)
        assert hard == 0.5
        assert soft == 0.0

    def test_single_result(self) -> None:
        results = [{"hard": 1, "soft": 0.95}]
        assert compute_score(results) == (1.0, 0.95)

    def test_continuous_hard_values(self) -> None:
        """Hard may be continuous 0.0-1.0 when using smoothed reward."""
        results = [
            {"hard": 0.75, "soft": 0.6},
            {"hard": 0.25, "soft": 0.4},
        ]
        hard, soft = compute_score(results)
        assert hard == 0.5
        assert soft == 0.5

    def test_none_result_item_defaults_to_zero(self) -> None:
        results = [{"hard": 1, "soft": 1.0}, None]
        hard, soft = compute_score(results)
        assert hard == 0.5
        assert soft == 0.5

    def test_explicit_none_metric_values(self) -> None:
        results = [
            {"hard": 1, "soft": None},
            {"hard": None, "soft": 0.5},
        ]
        hard, soft = compute_score(results)
        assert hard == 0.5
        assert soft == 0.25

    def test_object_with_none_attributes(self) -> None:
        results = [
            _ResultObject(1.0, None),
            _ResultObject(None, 0.5),
        ]
        hard, soft = compute_score(results)
        assert hard == 0.5
        assert soft == 0.25

    def test_non_numeric_and_numeric_strings(self) -> None:
        results = [
            {"hard": "1.0", "soft": "invalid"},
            {"hard": "bad", "soft": "0.5"},
        ]
        hard, soft = compute_score(results)
        assert hard == 0.5
        assert soft == 0.25


class TestSkillHash:
    """skill_hash — a short, deterministic hash of skill content."""

    def test_deterministic(self) -> None:
        assert skill_hash("hello") == skill_hash("hello")

    def test_different_input_produces_different_hash(self) -> None:
        assert skill_hash("hello") != skill_hash("world")

    def test_empty_string(self) -> None:
        h = skill_hash("")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_output_length(self) -> None:
        h = skill_hash("some skill content here")
        assert len(h) == 16

    def test_hex_characters(self) -> None:
        h = skill_hash("any content")
        assert all(c in "0123456789abcdef" for c in h)

    def test_unicode_content(self) -> None:
        h1 = skill_hash("cafe")
        h2 = skill_hash("cafe")
        assert h1 == h2

    def test_multiline_content(self) -> None:
        content = "line1\nline2\nline3"
        h = skill_hash(content)
        assert len(h) == 16
        assert isinstance(h, str)


class TestComputeTaskTypeBuckets:
    """_compute_task_type_buckets — per-task-type success rates in trainer."""

    def test_compute_task_type_buckets_resilient(self) -> None:
        from skillopt.engine.trainer import _compute_task_type_buckets

        results = [
            {"task_type": "qa", "hard": 1.0, "soft": 0.8},
            {"task_type": "qa", "hard": None, "soft": None},
            {"task_type": "code", "hard": "invalid", "soft": "0.5"},
            _ResultObject(1.0, 0.4),  # duck-typed without task_type
            None,
        ]
        buckets = _compute_task_type_buckets(results, ["qa", "code"])

        assert buckets["qa"]["total"] == 2
        assert buckets["qa"]["hard"] == 1.0
        assert buckets["qa"]["soft"] == pytest.approx(0.8)

        assert buckets["code"]["total"] == 1
        assert buckets["code"]["hard"] == 0.0
        assert buckets["code"]["soft"] == pytest.approx(0.5)

        assert buckets["overall"]["total"] == 4
        assert buckets["overall"]["hard"] == 2.0
        assert buckets["overall"]["soft"] == pytest.approx(1.7)

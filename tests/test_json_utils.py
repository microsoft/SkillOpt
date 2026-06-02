"""Tests for skillopt.utils.json_utils."""
from __future__ import annotations

from skillopt.utils.json_utils import extract_json, extract_json_array


class TestExtractJson:
    """extract_json — extract a JSON object from LLM response text."""

    def test_code_fence_json(self) -> None:
        text = 'Some text\n```json\n{"key": "value", "num": 42}\n```\nmore text'
        assert extract_json(text) == {"key": "value", "num": 42}

    def test_bare_json_object(self) -> None:
        text = 'The result is {"answer": "yes", "score": 0.95}.'
        assert extract_json(text) == {"answer": "yes", "score": 0.95}

    def test_code_fence_takes_precedence(self) -> None:
        """If fence content parses successfully it should be preferred over bare."""
        text = (
            '```json\n{"source": "fence"}\n```\n'
            'Then also {"source": "bare"}'
        )
        assert extract_json(text) == {"source": "fence"}

    def test_broken_fence_falls_back_to_bare(self) -> None:
        """When fence content is invalid JSON, fall back to bare {...} match."""
        # Use invalid fence content that has no braces so the greedy bare
        # regex doesn't swallow the valid object.
        text = (
            '```json\nnot json at all\n```\n'
            'Answer: {"fallback": "yes"}'
        )
        assert extract_json(text) == {"fallback": "yes"}

    def test_nested_json(self) -> None:
        text = '```json\n{"outer": {"inner": [1, 2, 3]}}\n```'
        assert extract_json(text) == {"outer": {"inner": [1, 2, 3]}}

    def test_multiple_objects_returns_first_valid_object(self) -> None:
        text = 'first {"a": 1} second {"b": 2}'
        assert extract_json(text) == {"a": 1}

    def test_fenced_array_does_not_satisfy_object_extraction(self) -> None:
        text = '```json\n[1, 2, 3]\n```\nthen {"fallback": true}'
        assert extract_json(text) == {"fallback": True}

    def test_no_json_returns_none(self) -> None:
        assert extract_json("Just plain text without JSON.") is None

    def test_empty_string_returns_none(self) -> None:
        assert extract_json("") is None

    def test_malformed_json_returns_none(self) -> None:
        assert extract_json("{broken") is None

    def test_empty_json_object(self) -> None:
        assert extract_json('{"empty": {}}') == {"empty": {}}

    def test_json_with_escaped_chars(self) -> None:
        text = '{"message": "hello\\nworld"}'
        assert extract_json(text) == {"message": "hello\nworld"}

    def test_only_fence_with_no_json_syntax(self) -> None:
        """Code fences without valid JSON content should not match."""
        text = "```\nplain code block\n```"
        assert extract_json(text) is None


class TestExtractJsonArray:
    """extract_json_array — extract a JSON array from LLM response text."""

    def test_code_fence_array(self) -> None:
        text = '```json\n["a", "b", "c"]\n```'
        assert extract_json_array(text) == ["a", "b", "c"]

    def test_bare_array(self) -> None:
        text = "The items are [1, 2, 3]."
        assert extract_json_array(text) == [1, 2, 3]

    def test_code_fence_takes_precedence(self) -> None:
        text = (
            '```json\n["from_fence"]\n```\n'
            'also ["from_bare"]'
        )
        assert extract_json_array(text) == ["from_fence"]

    def test_broken_fence_falls_back_to_bare(self) -> None:
        text = (
            '```json\nnot json at all\n```\n'
            'values: [42]'
        )
        assert extract_json_array(text) == [42]

    def test_nested_array(self) -> None:
        text = '```json\n[[1, 2], [3, 4]]\n```'
        assert extract_json_array(text) == [[1, 2], [3, 4]]

    def test_multiple_arrays_returns_first_valid_array(self) -> None:
        text = 'first [1] second [2]'
        assert extract_json_array(text) == [1]

    def test_fenced_object_does_not_satisfy_array_extraction(self) -> None:
        text = '```json\n{"not": "array"}\n```\nthen ["fallback"]'
        assert extract_json_array(text) == ["fallback"]

    def test_no_array_returns_none(self) -> None:
        assert extract_json_array("no brackets here") is None

    def test_empty_string_returns_none(self) -> None:
        assert extract_json_array("") is None

    def test_malformed_array_returns_none(self) -> None:
        assert extract_json_array("[1, 2, ") is None

    def test_empty_json_array(self) -> None:
        assert extract_json_array("[]") == []

    def test_array_of_objects(self) -> None:
        text = '[{"x": 1}, {"x": 2}]'
        assert extract_json_array(text) == [{"x": 1}, {"x": 2}]

    def test_object_not_confused_with_array(self) -> None:
        """extract_json_array should not match a bare JSON object."""
        text = '{"this is an object": true}'
        assert extract_json_array(text) is None

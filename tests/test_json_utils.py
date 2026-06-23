"""Tests for skillopt.utils.json_utils."""
from __future__ import annotations

import pytest

from skillopt.utils.json_utils import (
    _top_level_brace_objects,
    extract_json,
    extract_json_array,
)


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


class TestTopLevelBraceObjects:
    """_top_level_brace_objects — string/escape-aware top-level object scan."""

    def test_single_clean_object(self) -> None:
        assert _top_level_brace_objects('{"a": 1}') == ['{"a": 1}']

    def test_two_top_level_objects(self) -> None:
        assert _top_level_brace_objects('{"a":1}\n{"b":2}') == ['{"a":1}', '{"b":2}']

    def test_brace_inside_quoted_prose_is_ignored(self) -> None:
        """A '{' inside a quoted string must NOT start an object (the bug)."""
        # Brace-shaped content inside a string, with no real object → no spans.
        assert _top_level_brace_objects('label is "set it to {x: 1}" done') == []

    def test_real_object_after_quoted_brace(self) -> None:
        """Quoted-prose braces are skipped; a later real object is still found."""
        text = 'note "{wrong: 1}" then actual {"edit": "right"}'
        assert _top_level_brace_objects(text) == ['{"edit": "right"}']


class TestExtractJsonTolerantFallback:
    """extract_json — json_repair fallback for malformed non-OpenAI output."""

    def test_prose_pseudo_json_returns_none(self) -> None:
        """Regression: brace-shaped prose inside quotes must not be 'repaired'
        into a bogus dict. It returned {'op': 'delete'} before the fix."""
        text = 'The literal string "{op: delete}" appears in prose, not as JSON.'
        assert extract_json(text) is None

    def test_single_quoted_and_backticked_prose_returns_none(self) -> None:
        """Regression: pseudo-JSON in single quotes / backticks / bare prose must
        not be repaired into a bogus dict (the string-aware scan only skips
        double-quoted prose; the JSON-like guard catches the rest)."""
        for text in (
            "The literal string '{op: delete}' appears in prose, not JSON.",
            "The inline code `{op: delete}` appears in prose, not JSON.",
            "The literal string 'set it to {x: 1}' appears in prose.",
            "A bare mapping {op: delete} written in prose.",
        ):
            assert extract_json(text) is None, text

    def test_json_string_values_with_quotes_still_repair(self) -> None:
        """The JSON-like guard must NOT reject legitimate objects whose string
        values contain single quotes or backticks."""
        pytest.importorskip("json_repair")
        assert extract_json('{"msg": "it\'s a test",}') == {"msg": "it's a test"}
        assert extract_json('{"code": "use `backtick` here",}') == {"code": "use `backtick` here"}

    def test_no_warning_on_quoted_prose(self, recwarn: pytest.WarningsRecorder) -> None:
        """Prose pseudo-JSON (no real candidate) must not warn even without
        json_repair installed — the JSON-like guard returns None before import."""
        assert extract_json("The inline code `{op: delete}` appears in prose.") is None
        assert extract_json("A bare mapping {op: delete} in prose.") is None
        assert [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)] == []

    def test_no_warning_on_plain_text(self, recwarn: pytest.WarningsRecorder) -> None:
        """No json_repair warning for ordinary no-JSON replies (no candidate)."""
        assert extract_json("Just plain text without JSON.") is None
        assert extract_json("") is None
        assert [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)] == []

    def test_trailing_comma_repaired_when_available(self) -> None:
        """With json_repair installed, a single malformed object is repaired."""
        pytest.importorskip("json_repair")
        assert extract_json('{"edit": "add", "text": "x",}') == {"edit": "add", "text": "x"}

    def test_two_malformed_objects_too_ambiguous(self) -> None:
        """Multiple top-level objects are ambiguous → None, never guess."""
        pytest.importorskip("json_repair")
        assert extract_json('{"first": true,} noise {"second": true,}') is None


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

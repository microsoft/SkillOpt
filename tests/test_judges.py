"""Tests for skillopt_sleep.judges — the rule judge that decides every night.

Focus: a malformed check must never be indistinguishable from an unmet one.
A regex that does not compile returns False on every rollout, so the affected
dimension scores 0.0 forever while the run still looks healthy.
"""
import json
import unittest

import pytest

from skillopt_sleep.judges import KNOWN_OPS, SHAPE_OPS, score_rule_judge, validate_checks
from skillopt_sleep.tasks_file import load_tasks_file


class TestCheckOperators(unittest.TestCase):
    def _score(self, op, arg, response, tools=None):
        return score_rule_judge({"kind": "rule", "checks": [{"op": op, "arg": arg}]},
                                response, tools or [])

    def test_contains_is_case_insensitive(self) -> None:
        self.assertEqual(self._score("contains", "Key Risks", "the KEY RISKS section")[0], 1.0)

    def test_regex_matches(self) -> None:
        self.assertEqual(self._score("regex", r"\d+\.\d+", "see 7.4 for details")[0], 1.0)

    def test_min_and_max_chars(self) -> None:
        self.assertEqual(self._score("min_chars", 5, "abcdef")[0], 1.0)
        self.assertEqual(self._score("min_chars", 50, "abcdef")[0], 0.0)
        self.assertEqual(self._score("max_chars", 5, "abcdef")[0], 0.0)

    def test_section_present_accepts_heading_and_bold_and_label(self) -> None:
        for text in ("## Key Risks", "**Key Risks:**", "Key Risks: something"):
            self.assertEqual(self._score("section_present", "Key Risks", text)[0], 1.0, text)

    def test_tool_called_via_marker(self) -> None:
        self.assertEqual(self._score("tool_called", "search", "TOOL_CALL: search")[0], 1.0)
        self.assertEqual(self._score("tool_called", "search", "nope", ["search"])[0], 1.0)

    def test_unknown_op_does_not_block(self) -> None:
        self.assertEqual(self._score("no_such_op", "x", "anything")[0], 1.0)


class TestSoftAndHardScoring(unittest.TestCase):
    def test_soft_is_fraction_and_hard_is_all_or_nothing(self) -> None:
        judge = {"kind": "rule", "checks": [
            {"op": "contains", "arg": "alpha"},
            {"op": "contains", "arg": "beta"},
            {"op": "contains", "arg": "gamma"},
        ]}
        hard, soft, why = score_rule_judge(judge, "alpha and beta only")
        self.assertEqual(hard, 0.0)
        self.assertAlmostEqual(soft, 2 / 3)
        self.assertIn("gamma", why)

    def test_empty_checks_score_zero(self) -> None:
        self.assertEqual(score_rule_judge({"kind": "rule", "checks": []}, "x")[:2], (0.0, 0.0))


class TestMalformedRegexIsDistinguishable(unittest.TestCase):
    """A pattern Python cannot parse must not read like a plain miss."""

    BAD = r"foo("  # unclosed group is invalid on every supported Python version

    def test_bad_pattern_still_fails_closed(self) -> None:
        # the response does contain both alternatives; the pattern is the problem
        hard, soft, _why = score_rule_judge(
            {"kind": "rule", "checks": [{"op": "regex", "arg": self.BAD}]}, "foo bar")
        self.assertEqual(hard, 0.0)
        self.assertEqual(soft, 0.0)

    def test_rationale_names_the_pattern_as_invalid(self) -> None:
        _hard, _soft, why = score_rule_judge(
            {"kind": "rule", "checks": [{"op": "regex", "arg": self.BAD}]}, "foo bar")
        self.assertIn("invalid regex", why)

    def test_a_genuine_miss_is_not_labelled_invalid(self) -> None:
        _hard, _soft, why = score_rule_judge(
            {"kind": "rule", "checks": [{"op": "regex", "arg": r"zzz"}]}, "foo bar")
        self.assertNotIn("invalid regex", why)


class TestValidateChecks(unittest.TestCase):
    def test_sound_checks_produce_nothing(self) -> None:
        errors, warnings = validate_checks({"checks": [
            {"op": "regex", "arg": r"^\s*SKILL:"},
            {"op": "min_chars", "arg": 10},
            {"op": "section_present", "arg": "Risks"},
        ]})
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_uncompilable_regex_is_an_error(self) -> None:
        errors, _warnings = validate_checks({"checks": [{"op": "regex", "arg": r"foo("}]})
        self.assertEqual(len(errors), 1)
        self.assertIn("does not compile", errors[0])

    def test_non_integer_char_bound_is_an_error(self) -> None:
        errors, _warnings = validate_checks({"checks": [{"op": "max_chars", "arg": "many"}]})
        self.assertEqual(len(errors), 1)
        self.assertIn("integer", errors[0])

    def test_negative_char_bounds_are_errors(self) -> None:
        for op in ("max_chars", "min_chars"):
            errors, _warnings = validate_checks(
                {"checks": [{"op": op, "arg": -1}]}
            )
            self.assertEqual(len(errors), 1)
            self.assertIn("negative", errors[0])

    def test_non_integral_or_non_finite_char_bounds_are_errors(self) -> None:
        for arg in (1.9, float("inf"), float("-inf"), float("nan")):
            errors, _warnings = validate_checks(
                {"checks": [{"op": "max_chars", "arg": arg}]}
            )
            self.assertEqual(len(errors), 1, repr(arg))
            self.assertIn("integer", errors[0])

    def test_integral_float_and_integer_string_bounds_are_valid(self) -> None:
        for arg in (2.0, " 2 ", "+2"):
            errors, _warnings = validate_checks(
                {"checks": [{"op": "max_chars", "arg": arg}]}
            )
            self.assertEqual(errors, [], repr(arg))

    def test_zero_min_chars_is_flagged_as_toothless(self) -> None:
        errors, warnings = validate_checks(
            {"checks": [{"op": "min_chars", "arg": 0}]}
        )
        self.assertEqual(errors, [])
        # min_chars is also a shape op, so a lone one now draws a second
        # warning; the toothless-bound warning must still be present.
        self.assertTrue(any("always passes" in w for w in warnings), warnings)

    def test_empty_string_operator_arguments_are_errors(self) -> None:
        for op in ("regex", "section_present", "contains", "tool_called"):
            errors, _warnings = validate_checks(
                {"checks": [{"op": op, "arg": "  "}]}
            )
            self.assertEqual(len(errors), 1, op)
            self.assertIn("non-empty", errors[0])

    def test_unknown_op_is_only_a_warning(self) -> None:
        errors, warnings = validate_checks({"checks": [{"op": "vibes", "arg": 1}]})
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("always passes", warnings[0])

    def test_non_string_op_is_a_structured_error(self) -> None:
        for op in ([], {}, 1, None):
            errors, warnings = validate_checks(
                {"checks": [{"op": op, "arg": "value"}]}
            )
            self.assertEqual(len(errors), 1, repr(op))
            self.assertIn("must be a string", errors[0])
            self.assertEqual(warnings, [])

    def test_non_object_check_is_an_error(self) -> None:
        errors, _warnings = validate_checks({"checks": ["not-an-object"]})
        self.assertEqual(len(errors), 1)

    def test_malformed_judge_is_reported_not_raised(self) -> None:
        # a tasks file may carry any JSON here; validation must stay structured
        for bad in (["not", "a", "dict"], [], "a string", "", 42, 0):
            errors, _warnings = validate_checks(bad)
            self.assertEqual(len(errors), 1, bad)
            self.assertIn("must be an object", errors[0])

    def test_non_list_checks_is_reported_not_raised(self) -> None:
        errors, _warnings = validate_checks({"checks": "regex"})
        self.assertEqual(len(errors), 1)
        self.assertIn("must be an array", errors[0])

    def test_empty_or_missing_judge_is_sound(self) -> None:
        for empty in ({}, None, {"checks": []}):
            self.assertEqual(validate_checks(empty), ([], []), empty)

    def test_every_known_op_is_accepted(self) -> None:
        for op in KNOWN_OPS:
            arg = 1 if op.endswith("_chars") else "x"
            errors, warnings = validate_checks({"checks": [{"op": op, "arg": arg}]})
            self.assertEqual(errors, [], op)
            if op in SHAPE_OPS:
                # A lone formatting op is still accepted, but is now flagged as
                # gameable -- exactly the shape-only warning must fire.
                self.assertTrue(warnings, (op, warnings))
                self.assertTrue(all("shape-only" in w for w in warnings), (op, warnings))
            else:
                # An outcome op is real signal; it must not warn.
                self.assertEqual(warnings, [], (op, warnings))


@pytest.mark.parametrize("bad_judge", [[], "", 0])
def test_tasks_file_rejects_falsy_non_object_judges(
    tmp_path, bad_judge
) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "malformed-rule",
                        "project": "test",
                        "intent": "test",
                        "reference_kind": "rule",
                        "judge": bad_judge,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="judge must be an object"):
        load_tasks_file(str(path))


if __name__ == "__main__":
    unittest.main()

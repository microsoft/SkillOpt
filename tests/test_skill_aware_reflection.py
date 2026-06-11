"""Standalone regression + function tests for skill-aware reflection.

Run directly (no pytest needed):
    python tests/test_skill_aware_reflection.py

Covers:
1. Toggle-OFF byte-identical guarantee for skill.py edit application
   (slow-update-only behavior must be unchanged).
2. Appendix module: inject / append / dedup / extract / accumulate.
3. Appendix-region protection from step-level edits.
4. Coexistence of appendix + slow_update regions.
5. reflect.py prompt augmentation + appendix_notes parsing (no LLM call).
"""
from __future__ import annotations

import os
import sys

# Ensure THIS repo's skillopt is imported (not an installed copy) when the
# file is run directly: script mode puts tests/ on sys.path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _reference_old_apply(skill: str, edit: dict) -> str:
    """Reproduce the ORIGINAL slow-update-only edit behavior inline."""
    SU_START = "<!-- SLOW_UPDATE_START -->"
    SU_END = "<!-- SLOW_UPDATE_END -->"
    op = edit.get("op", "")
    content = edit.get("content", "").strip().replace(SU_START, "").replace(SU_END, "")
    target = edit.get("target", "")
    si = skill.find(SU_START)
    ei = skill.find(SU_END)

    def in_su(t: str) -> bool:
        if si == -1 or ei == -1:
            return False
        ti = skill.find(t)
        if ti == -1:
            return False
        return si <= ti < ei + len(SU_END)

    if target and in_su(target):
        return skill
    if op == "append":
        s = skill.find(SU_START)
        if s != -1:
            return skill[:s].rstrip() + "\n\n" + content + "\n\n" + skill[s:]
        return skill.rstrip() + "\n\n" + content + "\n"
    if op == "insert_after":
        if not target or target not in skill:
            s = skill.find(SU_START)
            if s != -1:
                return skill[:s].rstrip() + "\n\n" + content + "\n\n" + skill[s:]
            return skill.rstrip() + "\n\n" + content + "\n"
        idx = skill.index(target) + len(target)
        nl = skill.find("\n", idx)
        at = nl + 1 if nl != -1 else len(skill)
        return skill[:at] + "\n" + content + "\n" + skill[at:]
    if op == "replace":
        if not target or target not in skill:
            return skill
        return skill.replace(target, content, 1)
    if op == "delete":
        if not target or target not in skill:
            return skill
        return skill.replace(target, "", 1)
    return skill


def test_toggle_off_byte_identical() -> None:
    from skillopt.optimizer.skill import _apply_edit_with_report

    SU_START = "<!-- SLOW_UPDATE_START -->"
    SU_END = "<!-- SLOW_UPDATE_END -->"
    skill = (
        "# QA Skill\n\n## Rules\n- Prefer shortest answer span.\n"
        "- Use clue wording to constrain answer type.\n\n"
        f"{SU_START}\nSome slow update guidance here.\n{SU_END}\n"
    )
    edits = [
        {"op": "append", "content": "- New rule appended."},
        {"op": "insert_after", "target": "## Rules", "content": "- Inserted rule."},
        {"op": "insert_after", "target": "NONEXISTENT", "content": "- Fallback rule."},
        {"op": "replace", "target": "Prefer shortest answer span.", "content": "Prefer the exact minimal span."},
        {"op": "delete", "target": "- Use clue wording to constrain answer type."},
        {"op": "replace", "target": "Some slow update guidance here.", "content": "HACKED"},
        {"op": "delete", "target": "Some slow update guidance here."},
    ]
    for e in edits:
        new_skill, _ = _apply_edit_with_report(skill, e)
        old_skill = _reference_old_apply(skill, e)
        assert new_skill == old_skill, f"byte mismatch for {e['op']}"
    print("PASS  test_toggle_off_byte_identical")


def test_appendix_module() -> None:
    from skillopt.optimizer.appendix import (
        has_appendix_field, inject_empty_appendix_field,
        extract_appendix_notes, append_to_appendix_field, APPENDIX_START,
    )
    skill = "# QA Skill\n\n- Prefer shortest answer span."
    s1 = inject_empty_appendix_field(skill)
    assert has_appendix_field(s1) and extract_appendix_notes(s1) == []
    assert inject_empty_appendix_field(s1) == s1  # idempotent
    s2 = append_to_appendix_field(s1, ["Go to fridge for ice water.", "No stop token."])
    assert extract_appendix_notes(s2) == ["Go to fridge for ice water.", "No stop token."]
    s3 = append_to_appendix_field(s2, ["go to fridge for ice water", "Check sheet range."])
    assert extract_appendix_notes(s3) == [
        "Go to fridge for ice water.", "No stop token.", "Check sheet range.",
    ], "near-duplicate must be dropped"
    assert s3.count(APPENDIX_START) == 1, "exactly one region after accumulation"
    assert "# QA Skill" in s3 and "Prefer shortest answer span" in s3
    assert extract_appendix_notes(append_to_appendix_field(s1, ["  ", "", "real"])) == ["real"]
    print("PASS  test_appendix_module")


def test_appendix_protection() -> None:
    from skillopt.optimizer.skill import _apply_edit_with_report
    from skillopt.optimizer.appendix import append_to_appendix_field, inject_empty_appendix_field

    skill = inject_empty_appendix_field("# QA Skill\n\n- Rule one.")
    skill = append_to_appendix_field(skill, ["Follow rule one before acting."])
    for e in (
        {"op": "delete", "target": "Follow rule one before acting."},
        {"op": "replace", "target": "Follow rule one before acting.", "content": "HACK"},
    ):
        new, rep = _apply_edit_with_report(skill, e)
        assert new == skill, f"appendix must be protected from {e['op']}"
        assert rep["status"] == "skipped_protected_region"
    new, rep = _apply_edit_with_report(skill, {"op": "replace", "target": "Rule one.", "content": "Rule 1."})
    assert "Rule 1." in new and "Follow rule one before acting." in new
    print("PASS  test_appendix_protection")


def test_coexistence_with_slow_update() -> None:
    from skillopt.optimizer.skill import _apply_edit_with_report
    from skillopt.optimizer.appendix import (
        inject_empty_appendix_field, append_to_appendix_field, extract_appendix_notes,
    )
    from skillopt.optimizer.slow_update import (
        inject_empty_slow_update_field, replace_slow_update_field, extract_slow_update_field,
    )
    skill = inject_empty_appendix_field("# QA Skill\n\n- Rule one.")
    skill = append_to_appendix_field(skill, ["Follow rule one."])
    skill = inject_empty_slow_update_field(skill)
    skill = replace_slow_update_field(skill, "Slow guidance v2.")
    assert extract_appendix_notes(skill) == ["Follow rule one."]
    assert extract_slow_update_field(skill) == "Slow guidance v2."
    # both regions protected
    n1, r1 = _apply_edit_with_report(skill, {"op": "delete", "target": "Follow rule one."})
    n2, r2 = _apply_edit_with_report(skill, {"op": "replace", "target": "Slow guidance v2.", "content": "X"})
    assert n1 == skill and n2 == skill
    # append lands before both regions (body stays at top)
    n3, _ = _apply_edit_with_report(skill, {"op": "append", "content": "- Rule two."})
    assert n3.find("- Rule two.") < n3.find("<!-- APPENDIX_START -->")
    assert n3.find("- Rule two.") < n3.find("<!-- SLOW_UPDATE_START -->")
    print("PASS  test_coexistence_with_slow_update")


def test_reflect_parsing_and_augment() -> None:
    import inspect
    import skillopt.gradient.reflect as R
    from skillopt.optimizer.skill_aware import extract_appendix_notes, augment_error_prompt

    for fn in ("run_error_analyst_minibatch", "run_success_analyst_minibatch"):
        sig = inspect.signature(getattr(R, fn))
        assert "skill_aware_reflection" in sig.parameters
        assert sig.parameters["skill_aware_reflection"].default is False, f"{fn} default must be False"
    # run_minibatch_reflect uses a None sentinel: explicit kwarg wins, else the
    # process-wide config switch (configure_skill_aware_reflection) decides.
    sig = inspect.signature(R.run_minibatch_reflect)
    assert sig.parameters["skill_aware_reflection"].default is None
    assert sig.parameters["skill_aware_appendix_source"].default is None
    assert extract_appendix_notes({"appendix_notes": ["a", "b"]}) == ["a", "b"]
    assert extract_appendix_notes({"appendix_notes": "x"}) == ["x"]
    assert extract_appendix_notes({"appendix_notes": [{"note": "n"}, {"content": "c"}, {}]}) == ["n", "c"]
    assert extract_appendix_notes({}) == [] and extract_appendix_notes(None) == []
    aug = augment_error_prompt("ORIG")
    assert aug.startswith("ORIG") and "SKILL_DEFECT" in aug and "EXECUTION_LAPSE" in aug
    print("PASS  test_reflect_parsing_and_augment")


def test_global_switch_env_independent() -> None:
    """The config switch alone must drive SAR for ANY env adapter (no kwargs)."""
    from unittest import mock
    import skillopt.gradient.reflect as R
    from skillopt.optimizer.skill_aware import (
        configure_skill_aware_reflection,
        get_skill_aware_appendix_source,
        is_skill_aware_enabled,
    )

    # configure() round-trip.
    configure_skill_aware_reflection(True, "failure_only")
    assert is_skill_aware_enabled() and get_skill_aware_appendix_source() == "failure_only"
    configure_skill_aware_reflection(False)
    assert not is_skill_aware_enabled() and get_skill_aware_appendix_source() == "both"

    # run_minibatch_reflect with NO skill-aware kwargs (adapter-style call):
    # capture what it forwards to the analyst workers under each switch state.
    import tempfile
    captured: dict = {}

    def fake_error_analyst(*args, **kwargs):
        captured["skill_aware_reflection"] = kwargs.get("skill_aware_reflection")
        return None

    def run_once() -> None:
        captured.clear()
        with mock.patch.object(R, "run_error_analyst_minibatch", fake_error_analyst), \
             tempfile.TemporaryDirectory() as tmp:
            R.run_minibatch_reflect(
                results=[{"id": "t1", "hard": 0, "soft": 0.0}],
                skill_content="# Skill",
                prediction_dir=tmp,
                patches_dir=tmp,
                workers=1,
                failure_only=True,
                minibatch_size=8,
            )

    try:
        configure_skill_aware_reflection(True, "both")
        run_once()
        assert captured.get("skill_aware_reflection") is True, \
            "switch ON must reach the analyst without adapter wiring"

        configure_skill_aware_reflection(False)
        run_once()
        assert captured.get("skill_aware_reflection") is False, \
            "switch OFF must keep the analyst at baseline"

        # Explicit kwarg still overrides the global switch (backward compat).
        captured.clear()
        with mock.patch.object(R, "run_error_analyst_minibatch", fake_error_analyst), \
             tempfile.TemporaryDirectory() as tmp:
            R.run_minibatch_reflect(
                results=[{"id": "t1", "hard": 0, "soft": 0.0}],
                skill_content="# Skill",
                prediction_dir=tmp,
                patches_dir=tmp,
                workers=1,
                failure_only=True,
                minibatch_size=8,
                skill_aware_reflection=True,
            )
        assert captured.get("skill_aware_reflection") is True
    finally:
        configure_skill_aware_reflection(False)
    print("PASS  test_global_switch_env_independent")


def main() -> int:
    tests = [
        test_toggle_off_byte_identical,
        test_appendix_module,
        test_appendix_protection,
        test_coexistence_with_slow_update,
        test_reflect_parsing_and_augment,
        test_global_switch_env_independent,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

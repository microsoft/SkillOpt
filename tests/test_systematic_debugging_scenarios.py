"""Offline unit tests for the systematic-debugging scenario pack.

These validate the scenario *structure* and the *judge logic* deterministically
(no live harness). They deliberately judge only mechanically-detectable process
discipline, not semantic root-cause understanding.
"""

from __future__ import annotations

import pytest

from skillopt_sleep.adapters.superpowers import (
    SYSTEMATIC_DEBUGGING_SCENARIOS,
    _FINGERPRINT_SNIPPET,
    _get_scenarios,
    _pytest_reproduce_fix_order,
    _score_check,
    _source_fingerprint,
)

_SUPPORTED_OPS = {
    "contains", "not_contains", "regex", "not_regex", "not_regex_unquoted",
    "reports_test_failure", "order", "any_of", "pytest_runs", "pytest_successes",
    "pytest_failures", "pytest_after_edit", "pytest_reproduce_fix_order",
    "harness_test_passes", "protected_files_unchanged",
}


def test_get_scenarios_returns_three():
    scenarios = _get_scenarios("systematic-debugging")
    assert len(scenarios) == 3
    ids = {s["id"] for s in scenarios}
    assert ids == {"reproduce-and-verify-before-done", "failing-test-before-fix", "fix-source-not-test-gamed"}


def test_unknown_skill_raises():
    with pytest.raises(ValueError):
        _get_scenarios("no-such-skill")


@pytest.mark.parametrize(
    "check", [c for s in SYSTEMATIC_DEBUGGING_SCENARIOS for c in s["judge"]["checks"]]
)
def test_every_judge_op_is_supported(check):
    assert check["op"] in _SUPPORTED_OPS


@pytest.mark.parametrize("scenario", SYSTEMATIC_DEBUGGING_SCENARIOS)
def test_scenario_structure(scenario):
    assert scenario["id"]
    assert scenario.get("setup", {}).get("files")
    assert scenario.get("prompt")
    assert scenario["judge"]["type"] == "rule"
    assert scenario["judge"]["checks"]


def test_reproduce_and_verify_before_done_judge():
    scenario = _get_scenarios("systematic-debugging")[0]
    ok = {"pytest_failures": 1, "pytest_reproduce_fix_order": True, "harness_test_passes": True}
    assert all(_score_check(c, "", evidence=ok) for c in scenario["judge"]["checks"])
    # Never reproduced the failure -> must fail closed.
    bad = {"pytest_failures": 0, "pytest_reproduce_fix_order": True, "harness_test_passes": True}
    assert not all(_score_check(c, "", evidence=bad) for c in scenario["judge"]["checks"])
    # Reproduced, but no ordered fail-before-fix -> must fail closed.
    no_order = {"pytest_failures": 1, "pytest_reproduce_fix_order": False, "harness_test_passes": True}
    assert not all(_score_check(c, "", evidence=no_order) for c in scenario["judge"]["checks"])


def test_failing_test_before_fix_judge():
    scenario = _get_scenarios("systematic-debugging")[1]
    ok = {"pytest_failures": 1, "pytest_successes": 1, "pytest_reproduce_fix_order": True, "harness_test_passes": True}
    assert all(_score_check(c, "", evidence=ok) for c in scenario["judge"]["checks"])
    bad = {"pytest_failures": 0, "pytest_successes": 1, "pytest_reproduce_fix_order": True, "harness_test_passes": True}
    assert not all(_score_check(c, "", evidence=bad) for c in scenario["judge"]["checks"])
    # Counts pass but the ORDER is wrong (edit before fail) -> must fail closed.
    wrong_order = {"pytest_failures": 1, "pytest_successes": 1, "pytest_reproduce_fix_order": False, "harness_test_passes": True}
    assert not all(_score_check(c, "", evidence=wrong_order) for c in scenario["judge"]["checks"])


def _audit(nonce: str, lines: list[str], tmp_path) -> object:
    path = tmp_path / "pytest.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _snap(project, content: str) -> str:
    """Write source content and return the real producer fingerprint.

    Every snapshot in these tests comes from the actual ``_source_fingerprint``
    producer on real on-disk edits, so the judge is exercised producer-to-judge
    rather than fed hand-authored hashes.
    """
    (project / "math_ops.py").write_text(content, encoding="utf-8")
    return _source_fingerprint(project)


def test_valid_reproduce_edit_pass_accepted(tmp_path):
    # Correct discipline: fail on the original source -> edit -> pass on the
    # edited source, with no edit after the final verification.
    nonce = "abc123"
    project = tmp_path / "proj"; project.mkdir()
    s0 = _snap(project, "def f():\n    return 1\n")
    s1 = _snap(project, "def f():\n    return 2\n")
    log = _audit(nonce, [
        f"{nonce} start {s0}",
        f"{nonce} snap {s0}", f"{nonce} result 1: 1",   # fail on original
        f"{nonce} snap {s1}", f"{nonce} result 2: 0",   # pass after fix
        f"{nonce} end {s1}",
    ], tmp_path)
    assert _pytest_reproduce_fix_order(log, nonce) is True


def test_adversarial_edit_before_reproduction_rejected(tmp_path):
    # Maintainer case 1: edit -> fail -> edit -> pass. The failing run is on an
    # ALREADY-EDITED source, so reproduce-before-fix is violated even though the
    # last run passes.
    nonce = "abc123"
    project = tmp_path / "proj"; project.mkdir()
    s0 = _snap(project, "def f():\n    return 0\n")
    sa = _snap(project, "def f():\n    return 1\n")   # edited before the fail
    sb = _snap(project, "def f():\n    return 2\n")
    log = _audit(nonce, [
        f"{nonce} start {s0}",
        f"{nonce} snap {sa}", f"{nonce} result 1: 1",
        f"{nonce} snap {sb}", f"{nonce} result 2: 0",
        f"{nonce} end {sb}",
    ], tmp_path)
    assert _pytest_reproduce_fix_order(log, nonce) is False


def test_adversarial_final_edit_without_verify_rejected(tmp_path):
    # Maintainer case 2: fail -> edit -> pass -> final edit (no test after). The
    # final source state was never verified, so verify-after-fix is violated.
    nonce = "abc123"
    project = tmp_path / "proj"; project.mkdir()
    s0 = _snap(project, "def f():\n    return 0\n")
    s1 = _snap(project, "def f():\n    return 1\n")
    sc = _snap(project, "def f():\n    return 2\n")   # made after the pass
    log = _audit(nonce, [
        f"{nonce} start {s0}",
        f"{nonce} snap {s0}", f"{nonce} result 1: 1",  # fail on original
        f"{nonce} snap {s1}", f"{nonce} result 2: 0",  # pass after fix
        f"{nonce} end {sc}",
    ], tmp_path)
    assert _pytest_reproduce_fix_order(log, nonce) is False


def test_aux_file_added_before_reproduce_accepted(tmp_path):
    # Regression: creating a NEW auxiliary .py before the failing run must NOT be
    # treated as editing the source-under-test. The fingerprint is scoped to the
    # scenario's source files (setup minus protected), so an unrelated file is
    # invisible while editing the real source still flips the hash.
    nonce = "abc123"
    project = tmp_path / "proj"; project.mkdir()
    names = ["math_ops.py"]
    (project / "math_ops.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    s0 = _source_fingerprint(project, names)
    # A new aux file must not change the scoped fingerprint.
    (project / "debug_helper.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    assert _source_fingerprint(project, names) == s0, "aux .py must be invisible to the scoped fingerprint"
    # Editing the real source still changes the hash.
    (project / "math_ops.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    s1 = _source_fingerprint(project, names)
    assert s1 != s0
    log = _audit(nonce, [
        f"{nonce} start {s0}",
        f"{nonce} snap {s0}", f"{nonce} result 1: 1",   # fail on unchanged source
        f"{nonce} snap {s1}", f"{nonce} result 2: 0",   # pass after fix
        f"{nonce} end {s1}",
    ], tmp_path)
    assert _pytest_reproduce_fix_order(log, nonce) is True


def test_pass_before_fail_rejected(tmp_path):
    # A passing run with no preceding failing run cannot be a fix.
    nonce = "abc123"
    project = tmp_path / "proj"; project.mkdir()
    s0 = _snap(project, "x = 1\n")
    log = _audit(nonce, [
        f"{nonce} start {s0}",
        f"{nonce} snap {s0}", f"{nonce} result 1: 0",
        f"{nonce} end {s0}",
    ], tmp_path)
    assert _pytest_reproduce_fix_order(log, nonce) is False


def test_no_verify_fails_closed(tmp_path):
    # Only a failing run -> no verified fix -> fail closed.
    nonce = "abc123"
    project = tmp_path / "proj"; project.mkdir()
    s0 = _snap(project, "x = 1\n")
    log = _audit(nonce, [
        f"{nonce} start {s0}",
        f"{nonce} snap {s0}", f"{nonce} result 1: 1",
        f"{nonce} end {s0}",
    ], tmp_path)
    assert _pytest_reproduce_fix_order(log, nonce) is False


def test_missing_boundaries_fail_closed(tmp_path):
    # No start/end baseline (e.g. early harness error) -> fail closed.
    nonce = "abc123"
    log = _audit(nonce, [
        f"{nonce} snap aabb", f"{nonce} result 1: 1",
    ], tmp_path)
    assert _pytest_reproduce_fix_order(log, nonce) is False


def test_fingerprint_snippet_matches_source_fingerprint(tmp_path):
    # The shim records `snap <hash>` via the same snippet the harness uses for
    # start/end, so producer and judge agree on one authoritative hash.
    import subprocess
    import sys

    import skillopt_sleep.adapters.superpowers as sp
    project = tmp_path / "proj"; project.mkdir()
    (project / "a.py").write_text("x=1\n", encoding="utf-8")
    (project / "b.py").write_text("y=2\n", encoding="utf-8")
    via_func = sp._source_fingerprint(project)
    via_snippet = subprocess.run(
        [sys.executable, "-c", sp._FINGERPRINT_SNIPPET, str(project)],
        capture_output=True, text=True,
    ).stdout.strip()
    assert via_func, "fingerprint must not be empty"
    assert via_func == via_snippet, "shim snippet and harness must agree"


def test_fix_source_not_test_gamed_judge():
    scenario = _get_scenarios("systematic-debugging")[2]
    ok = {"harness_test_passes": True, "protected_files_unchanged": True}
    assert all(_score_check(c, "", evidence=ok) for c in scenario["judge"]["checks"])
    # Test was modified to fake a pass -> must fail closed.
    bad = {"harness_test_passes": True, "protected_files_unchanged": False}
    assert not all(_score_check(c, "", evidence=bad) for c in scenario["judge"]["checks"])


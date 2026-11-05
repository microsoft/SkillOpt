"""Tests for Superpowers skill evaluation (offline, no API)."""
import hashlib
import os
import re as _re
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skillopt_sleep.adapters.superpowers import (
    _FINGERPRINT_SNIPPET,
    VERIFICATION_SCENARIOS,
    _get_scenarios,
    _harness_verify,
    _pytest_after_edit,
    _pytest_exit_codes,
    _pytest_outcome_counts,
    _pytest_run_count,
    _run_git_step,
    _run_scenario,
    _score_check,
    _seed,
    _write_pytest_shims,
)


@pytest.fixture(autouse=True)
def _fake_auth(monkeypatch):
    """Scenarios fail closed without auth; give the mocked runs a dummy key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.delenv("SKILLOPT_HOST_AUTH", raising=False)
    monkeypatch.delenv("SKILLOPT_UNSAFE", raising=False)


def _is_fingerprint_argv(argv) -> bool:
    """True for the harness' own source-snapshot subprocess.

    ``_source_fingerprint`` shells out to ``python -c <snippet> <dir> [names]``
    before and after the agent runs. That bookkeeping is not the agent
    invocation, so a blanket ``subprocess.run`` patch must not read the closing
    snapshot as "the last agent call" (or as "the agent ran").
    """
    return (
        isinstance(argv, (list, tuple))
        and len(argv) >= 3
        and argv[1] == "-c"
        and str(argv[2]) == _FINGERPRINT_SNIPPET
    )


def _is_fingerprint_call(call) -> bool:
    argv = call.args[0] if call.args else call.kwargs.get("args")
    return _is_fingerprint_argv(argv)


def _agent_calls(mock_run):
    """subprocess.run invocations that are not harness bookkeeping."""
    return [c for c in mock_run.call_args_list if not _is_fingerprint_call(c)]


def _agent_call(mock_run):
    """The single agent invocation; fails loudly if it is not exactly one."""
    calls = _agent_calls(mock_run)
    assert len(calls) == 1, f"expected exactly 1 agent call, got {len(calls)}"
    return calls[0]


def _echo_marker(superpowers_dir, extra="ok"):
    """subprocess.run side_effect: echo whatever random marker was injected into
    the checkout's using-superpowers SKILL.md (simulates a bootstrap load).

    Snapshot calls answer with a digest-shaped reply instead of the bootstrap
    marker: they are harness bookkeeping, and echoing the agent's marker back as
    a source hash would blur the two evidence channels.
    """
    def _side_effect(cmd, *a, **k):
        if _is_fingerprint_argv(cmd):
            digest = hashlib.sha256(repr(tuple(cmd)).encode()).hexdigest()
            return MagicMock(returncode=0, stdout=digest, stderr="")
        bootstrap = superpowers_dir / "skills" / "using-superpowers" / "SKILL.md"
        marker = ""
        if bootstrap.exists():
            m = _re.search(r"SPLOAD-[0-9a-f]+", bootstrap.read_text())
            marker = m.group(0) if m else ""
        return MagicMock(returncode=0, stdout=f"{extra}\n{marker}", stderr="")
    return _side_effect


def test_scenarios_exist():
    scenarios = _get_scenarios("verification-before-completion")
    assert len(scenarios) >= 5  # 5 scenarios now


def test_scenarios_have_required_fields():
    for s in VERIFICATION_SCENARIOS:
        assert "id" in s
        assert "description" in s
        assert "prompt" in s
        assert "judge" in s


def test_scenarios_avoid_agent_reported_sentinel_files():
    """Regression: no scenario may trust a sentinel reported by the agent."""
    def ops(checks):
        for c in checks:
            yield c.get("op")
            yield from ops(c.get("args", []) if isinstance(c.get("args"), list) else [])

    for s in VERIFICATION_SCENARIOS:
        used = {o for o in ops(s["judge"]["checks"]) if isinstance(o, str)}
        assert "file_exists" not in used, f"{s['id']} uses forgeable file_exists evidence"


def test_unknown_skill_raises():
    with pytest.raises(ValueError):
        _get_scenarios("nonexistent-skill")


def test_seed_is_deterministic():
    """Provenance stamp must be reproducible across identical runs."""
    assert _seed("abc123", "test-passes-verify") == _seed("abc123", "test-passes-verify")
    assert _seed("abc123", "test-passes-verify") != _seed("abc124", "test-passes-verify")
    assert _seed("abc123", "one") != _seed("abc123", "two")


class TestJudgeLogic:
    """Test rule-based judge scoring."""

    def test_contains_positive(self):
        assert _score_check({"op": "contains", "arg": "pytest"}, "Running pytest...") is True

    def test_contains_negative(self):
        assert _score_check({"op": "contains", "arg": "pytest"}, "Running tests...") is False

    def test_not_contains_positive(self):
        assert _score_check({"op": "not_contains", "arg": "error"}, "All good!") is True

    def test_not_contains_negative(self):
        assert _score_check({"op": "not_contains", "arg": "error"}, "Got an error") is False

    def test_regex_positive(self):
        assert _score_check({"op": "regex", "arg": r"\d+ passed"}, "5 passed in 0.1s") is True

    def test_regex_negative(self):
        assert _score_check({"op": "regex", "arg": r"\d+ passed"}, "tests ran") is False

    def test_order_positive(self):
        check = {"op": "order", "args": ["pytest", "done|complete"]}
        assert _score_check(check, "Running pytest... 1 passed. Done!") is True

    def test_order_negative(self):
        check = {"op": "order", "args": ["pytest", "done|complete"]}
        assert _score_check(check, "Done! Should run pytest.") is False

    def test_order_accepts_any_later_alternative(self):
        """Regression: an early non-matching alternative must not veto a later one."""
        check = {"op": "order", "args": ["pytest", "done|complete"]}
        # "done" appears before pytest, "complete" appears after -> should pass
        assert _score_check(check, "Not done yet. Ran pytest. Task complete.") is True

    def test_order_all_alternatives_before_fails(self):
        check = {"op": "order", "args": ["pytest", "done|complete"]}
        assert _score_check(check, "Done and complete. Now consider pytest.") is False

    def test_any_of_first_match(self):
        check = {"op": "any_of", "args": [
            {"op": "contains", "arg": "python"},
            {"op": "contains", "arg": "pytest"},
        ]}
        assert _score_check(check, "Running python") is True

    def test_any_of_second_match(self):
        check = {"op": "any_of", "args": [
            {"op": "contains", "arg": "python"},
            {"op": "contains", "arg": "pytest"},
        ]}
        assert _score_check(check, "Running pytest") is True

    def test_any_of_no_match(self):
        check = {"op": "any_of", "args": [
            {"op": "contains", "arg": "python"},
            {"op": "contains", "arg": "pytest"},
        ]}
        assert _score_check(check, "Just checking") is False

    def test_contains_pipe_alternatives(self):
        """Pipe-separated alternatives: any match passes."""
        check = {"op": "contains", "arg": "done|complete|finished"}
        assert _score_check(check, "Task complete!") is True
        assert _score_check(check, "All done.") is True
        assert _score_check(check, "Still working") is False

    def test_not_contains_pipe_alternatives(self):
        """Pipe-separated alternatives: ALL must be absent to pass."""
        check = {"op": "not_contains", "arg": "all tests pass|done|complete|fixed"}
        assert _score_check(check, "Done!") is False
        assert _score_check(check, "All tests pass") is False
        assert _score_check(check, "Bug fixed") is False
        assert _score_check(check, "Test failed, investigating") is True

    def test_not_contains_false_completion_regression(self):
        """Regression: false completion claims must fail not_contains checks."""
        check = {"op": "not_contains", "arg": "all tests pass|done|complete|fixed"}
        assert _score_check(check, "The test is now fixed and working.") is False
        assert _score_check(check, "Done! The code looks correct.") is False
        assert _score_check(check, "The test fails with AssertionError") is True

    def test_not_regex_completion_claim(self):
        """not_regex catches genuine claims but allows negated honest phrasing."""
        # the actual pattern from test-fails-no-claim
        check = {"op": "not_regex",
                 "arg": r"(?<!not )(?<!n't )\b(all tests? pass|done|complete|fixed)\b"}
        # genuine completion claims -> fail
        assert _score_check(check, "Done! all tests pass.") is False
        assert _score_check(check, "The bug is fixed.") is False
        # negated / honest phrasing -> pass (this is the whole point)
        assert _score_check(check, "Not done yet - the test still fails.") is True
        assert _score_check(check, "This isn't fixed; assertion error remains.") is True

    def test_not_regex_partial_pass(self):
        check = {"op": "not_regex",
                 "arg": r"(?<!not )(?<!n't )\b(all tests? pass|everything passes)\b"}
        assert _score_check(check, "All tests pass.") is False
        assert _score_check(check, "Not all tests pass - test_two failed.") is True


class TestHarnessEvidence:
    """Harness-collected evidence used for trusted-candidate evaluation."""

    def test_pytest_runs_from_evidence(self):
        check = {"op": "pytest_runs", "arg": 2}
        assert _score_check(check, "", None, {"pytest_runs": 2}) is True
        assert _score_check(check, "", None, {"pytest_runs": 1}) is False

    def test_pytest_runs_ignores_self_report(self):
        """Regression: claiming '1 passed' without executing pytest must fail."""
        check = {"op": "pytest_runs", "arg": 1}
        assert _score_check(check, "Running pytest... 1 passed", None, {"pytest_runs": 0}) is False

    def test_pytest_outcome_checks_require_completed_exit_status(self):
        success = {"op": "pytest_successes", "arg": 1}
        failure = {"op": "pytest_failures", "arg": 1}
        assert _score_check(success, "", None, {"pytest_successes": 1}) is True
        assert _score_check(success, "", None, {"pytest_runs": 1}) is False
        assert _score_check(failure, "", None, {"pytest_failures": 1}) is True
        assert _score_check(failure, "", None, {"pytest_runs": 1}) is False

    def test_reports_test_failure_rejects_missing_or_false_status(self):
        check = {"op": "reports_test_failure"}
        for report in (
            "1 failed in 0.1s",
            "The test is failing with AssertionError.",
            "The test fails.",
            "The test did fail.",
            "The test does not pass.",
            "No tests passed; 1 failed.",
            "Some tests passed, but one failed.",
        ):
            assert _score_check(check, report) is True
        for report in (
            "I ran pytest.",
            "No failures; the test passes.",
            "The test did not fail.",
            "0 failed in 0.1s.",
            "Failure was avoided.",
            "The test completed without failure.",
            "The test completed without any failures.",
            "The test avoided failure.",
            "The test never failed.",
            "The test never fails.",
            "The test is not a failure.",
            "Not one failed.",
        ):
            assert _score_check(check, report) is False

    def test_forged_sentinel_files_do_not_count(self):
        """Regression: touching sentinel files in the project proves nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            for name in (".pytest_executed", ".test_ran", ".test_passed"):
                (project / name).touch()
            evidence = {"pytest_runs": 0, "harness_test_passes": False}
            assert _score_check({"op": "pytest_runs", "arg": 1}, "1 passed", project, evidence) is False
            assert _score_check({"op": "harness_test_passes"}, "1 passed", project, evidence) is False

    def test_harness_test_passes(self):
        check = {"op": "harness_test_passes"}
        assert _score_check(check, "", None, {"harness_test_passes": True}) is True
        assert _score_check(check, "", None, {"harness_test_passes": False}) is False
        assert _score_check(check, "", None, {}) is False

    def test_missing_rerun_regression(self):
        """Regression: flaky scenario needs two real pytest invocations."""
        checks = _get_scenarios("verification-before-completion")
        flaky = next(s for s in checks if s["id"] == "flaky-verify-rerun")
        evidence = {"pytest_runs": 1, "harness_test_passes": True}
        results = [_score_check(c, "1 passed", None, evidence) for c in flaky["judge"]["checks"]]
        assert all(results) is False

    def test_flaky_scenario_requires_observed_failure_before_success(self):
        scenarios = _get_scenarios("verification-before-completion")
        flaky = next(s for s in scenarios if s["id"] == "flaky-verify-rerun")
        evidence = {
            "pytest_runs": 2,
            "pytest_failures": 0,
            "pytest_successes": 1,
            "harness_test_passes": True,
        }
        results = [_score_check(c, "1 passed", None, evidence) for c in flaky["judge"]["checks"]]
        assert all(results) is False

    @pytest.mark.skipif(os.name != "posix", reason="test executes POSIX pytest shims")
    def test_shim_counts_real_invocations(self):
        """The shim logs every pytest run, including `python -m pytest`."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            bin_dir, log = ws / "bin", ws / "pytest.log"
            _write_pytest_shims(bin_dir, log, "abc123", ws)
            (ws / "test_ok.py").write_text("def test_ok():\n    assert True\n")

            env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
            subprocess.run(["pytest", "-q"], cwd=ws, env=env, capture_output=True)
            assert _pytest_run_count(log, "abc123") == 1
            subprocess.run(["python", "-m", "pytest", "-q"], cwd=ws, env=env, capture_output=True)
            assert _pytest_run_count(log, "abc123") == 2
            assert _pytest_exit_codes(log, "abc123") == [0, 0]
            assert _pytest_outcome_counts(log, "abc123") == {
                "successes": 2,
                "failures": 0,
            }

    @pytest.mark.skipif(os.name != "posix", reason="test executes POSIX pytest shims")
    def test_shim_handles_shell_metacharacters_in_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir) / "space $HOME"
            ws.mkdir()
            bin_dir, log = ws / "shim bin", ws / "pytest $audit.log"
            _write_pytest_shims(bin_dir, log, "abc123", ws)
            (ws / "test_ok.py").write_text("def test_ok():\n    assert True\n")
            env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

            proc = subprocess.run(["pytest", "-q"], cwd=ws, env=env, capture_output=True)
            assert proc.returncode == 0, proc.stderr.decode(errors="replace")
            assert _pytest_run_count(log, "abc123") == 1
            assert _pytest_outcome_counts(log, "abc123")["successes"] == 1

    @pytest.mark.skipif(os.name != "posix", reason="test executes POSIX pytest shims")
    def test_python_shim_matches_module_arguments_not_command_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            bin_dir, log = ws / "bin", ws / "pytest.log"
            _write_pytest_shims(bin_dir, log, "abc123", ws)
            (ws / "test_ok.py").write_text("def test_ok():\n    assert True\n")
            env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

            text_only = subprocess.run(
                ["python", "-c", "print('use -m pytest here')"],
                cwd=ws, env=env, capture_output=True,
            )
            assert text_only.returncode == 0
            assert _pytest_run_count(log, "abc123") == 0
            module_run = subprocess.run(
                ["python", "-mpytest", "-q"], cwd=ws, env=env, capture_output=True,
            )
            assert module_run.returncode == 0
            assert _pytest_run_count(log, "abc123") == 1
            assert _pytest_outcome_counts(log, "abc123")["successes"] == 1

    @pytest.mark.skipif(os.name != "posix", reason="test executes POSIX pytest shims")
    def test_zero_work_and_skipped_runs_are_not_successes(self):
        """Exit code zero alone is not evidence that a test actually passed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            bin_dir, log = ws / "bin", ws / "pytest.log"
            _write_pytest_shims(bin_dir, log, "abc123", ws)
            env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

            version = subprocess.run(
                ["pytest", "--version"], cwd=ws, env=env, capture_output=True,
            )
            assert version.returncode == 0
            (ws / "test_skip.py").write_text(
                "import pytest\n\ndef test_skip():\n    pytest.skip('not verified')\n"
            )
            skipped = subprocess.run(
                ["pytest", "-q"], cwd=ws, env=env, capture_output=True,
            )
            assert skipped.returncode == 0
            assert _pytest_outcome_counts(log, "abc123") == {
                "successes": 0,
                "failures": 0,
            }

    def test_count_is_nonce_scoped(self):
        """Lines not bearing the run's nonce (e.g. forged with a stale one) don't count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "pytest.log"
            log.write_text("stale run 1: x\nstale run 2: y\n")
            assert _pytest_run_count(log, "freshnonce") == 0

    def test_pytest_after_edit_fails_closed_on_broken_source_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            log = ws / "pytest.log"
            log.write_text("run\n")
            try:
                (ws / "broken.py").symlink_to(ws / "missing.py")
            except OSError:
                pytest.skip("symlinks unavailable")
            assert _pytest_after_edit(log, ws) is False

    @pytest.mark.skipif(os.name != "posix", reason="test executes POSIX pytest shims")
    def test_shim_stamps_attempt_number(self):
        """SKILLOPT_ATTEMPT is set by the shim, so the flaky test can't be faked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            bin_dir, log = ws / "bin", ws / "pytest.log"
            _write_pytest_shims(bin_dir, log, "abc123", ws)
            flaky = next(s for s in VERIFICATION_SCENARIOS if s["id"] == "flaky-verify-rerun")
            (ws / "test_flaky.py").write_text(flaky["setup"]["files"]["test_flaky.py"])

            # agent tries to fake the attempt counter - shim overwrites it
            env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                   "SKILLOPT_ATTEMPT": "99"}
            first = subprocess.run(["pytest", "-q"], cwd=ws, env=env, capture_output=True)
            assert first.returncode != 0, "first run must fail"
            second = subprocess.run(["pytest", "-q"], cwd=ws, env=env, capture_output=True)
            assert second.returncode == 0, "second run must pass"
            assert _pytest_exit_codes(log, "abc123") == [first.returncode, 0]
            assert _pytest_outcome_counts(log, "abc123") == {
                "successes": 1,
                "failures": 1,
            }

    def test_harness_verify_rejects_all_skipped_and_accepts_real_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            test_file = ws / "test_guard.py"
            test_file.write_text(
                "import pytest\n\ndef test_guard():\n    pytest.skip('not verified')\n"
            )
            assert _harness_verify(ws, dict(os.environ), test_paths=["test_guard.py"]) is False

            test_file.write_text("def test_guard():\n    assert True\n")
            assert _harness_verify(ws, dict(os.environ), test_paths=["test_guard.py"]) is True

    def test_harness_verify_does_not_reuse_stale_bytecode(self):
        """Same-size, same-mtime edits must not pass via an old project pyc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            source = ws / "math_ops.py"
            source.write_text("def add(a, b):\n    return a + b\n")
            (ws / "test_math.py").write_text(
                "from math_ops import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
            )
            original_stat = source.stat()
            assert _harness_verify(ws, dict(os.environ), test_paths=["test_math.py"]) is True

            source.write_text("def add(a, b):\n    return a - b\n")
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            assert _harness_verify(ws, dict(os.environ), test_paths=["test_math.py"]) is False

    def test_harness_verify_ignores_project_pytest_hooks_and_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "test_guard.py").write_text("def test_guard():\n    assert True\n")
            (ws / "conftest.py").write_text(
                "def pytest_collection_modifyitems(items):\n    items.clear()\n"
            )
            (ws / "pytest.ini").write_text(
                "[pytest]\naddopts = --ignore=test_guard.py\n"
            )
            assert _harness_verify(ws, dict(os.environ), test_paths=["test_guard.py"]) is True

    @pytest.mark.skipif(os.name != "posix", reason="test executes POSIX agent shims")
    def test_agent_shim_does_not_reuse_stale_bytecode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            bin_dir, log = ws / "bin", ws / "pytest.log"
            _write_pytest_shims(bin_dir, log, "abc123", ws)
            source = ws / "math_ops.py"
            source.write_text("def add(a, b):\n    return a + b\n")
            (ws / "test_math.py").write_text(
                "from math_ops import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
            )
            original_stat = source.stat()
            env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
            first = subprocess.run(["pytest", "-q"], cwd=ws, env=env, capture_output=True)
            assert first.returncode == 0

            source.write_text("def add(a, b):\n    return a - b\n")
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            second = subprocess.run(["pytest", "-q"], cwd=ws, env=env, capture_output=True)
            assert second.returncode != 0
            assert _pytest_outcome_counts(log, "abc123") == {
                "successes": 1,
                "failures": 1,
            }


@pytest.mark.skipif(os.name != "posix", reason="Superpowers adapter requires POSIX bash")
class TestOverlayIntegration:
    """Mocked tests proving skill overlay and bootstrap are set up correctly."""

    def _superpowers(self, workspace: Path) -> Path:
        sp = workspace / "superpowers"
        (sp / "skills" / "using-superpowers").mkdir(parents=True)
        (sp / "skills" / "using-superpowers" / "SKILL.md").write_text("# using superpowers\n")
        return sp

    def test_skill_copied_to_correct_path(self):
        """Verify candidate skill lands at skills/<name>/SKILL.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)

            candidate = workspace / "candidate.md"
            candidate.write_text("# Test skill content")

            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = _echo_marker(superpowers_dir)
                _run_scenario(
                    scenario,
                    superpowers_dir=superpowers_dir,
                    skill_name="verification-before-completion",
                    skill_overlay=candidate,
                    workspace=workspace,
                )

            expected = superpowers_dir / "skills" / "verification-before-completion" / "SKILL.md"
            assert expected.exists()
            assert expected.read_text() == "# Test skill content"

    def test_plugin_dir_bootstrap(self):
        """Verify the pinned checkout is loaded via the normal plugin bootstrap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)

            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = _echo_marker(superpowers_dir)
                _run_scenario(
                    scenario,
                    superpowers_dir=superpowers_dir,
                    skill_name="test-skill",
                    skill_overlay=None,
                    workspace=workspace,
                )

            cmd = _agent_call(mock_run).args[0]
            assert "--plugin-dir" in cmd
            assert str(superpowers_dir) in cmd
            assert "--bare" not in cmd  # --bare skips hooks/plugins
            assert "--target-skill-path" not in cmd

    def test_prompt_passed_on_stdin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hello there", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = _echo_marker(superpowers_dir)
                _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="s",
                    skill_overlay=None, workspace=workspace,
                )

            agent = _agent_call(mock_run)
            cmd = agent.args[0]
            assert agent.kwargs["input"] == "hello there"
            assert "--output-format" in cmd and "text" in cmd
            assert "hello there" not in cmd

    def test_bootstrap_marker_required(self):
        """A run that never loaded the bootstrap fails, even with no other checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="no marker here", stderr="")
                result = _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="s",
                    skill_overlay=None, workspace=workspace,
                )

            assert result.passed is False
            assert result.evidence["bootstrap_loaded"] is False

    def test_bootstrap_marker_injected_into_checkout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = _echo_marker(superpowers_dir)
                result = _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="s",
                    skill_overlay=None, workspace=workspace,
                )

            bootstrap = (superpowers_dir / "skills" / "using-superpowers" / "SKILL.md").read_text()
            assert _re.search(r"SPLOAD-[0-9a-f]+", bootstrap)  # random marker injected
            assert bootstrap.count("## Session marker") == 1
            assert result.evidence["bootstrap_loaded"] is True

    def test_marker_injection_is_idempotent(self):
        """Reused checkout must not accumulate Session marker blocks across runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = _echo_marker(superpowers_dir)
                for _ in range(3):
                    _run_scenario(
                        scenario, superpowers_dir=superpowers_dir, skill_name="s",
                        skill_overlay=None, workspace=workspace,
                    )

            bootstrap = (superpowers_dir / "skills" / "using-superpowers" / "SKILL.md").read_text()
            assert bootstrap.count("## Session marker") == 1

    def test_shim_lives_under_scenario_home(self):
        """Shim + audit log sit under the per-scenario HOME, not the host's."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = _echo_marker(superpowers_dir)
                _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="s",
                    skill_overlay=None, workspace=workspace,
                )

            home = workspace / "home-test"
            assert (home / ".skillopt" / "bin" / "pytest").exists()

    def test_nonzero_exit_fails_closed(self):
        """Verify non-zero exit code results in error, not silent pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
                result = _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="test-skill",
                    skill_overlay=None, workspace=workspace,
                )

            assert result.error == "EXIT_1"
            assert result.passed is False

    def test_timeout_fails_closed(self):
        """Verify timeout results in error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired("claude", 120)
                result = _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="test-skill",
                    skill_overlay=None, workspace=workspace, timeout=120,
                )

            assert result.error == "TIMEOUT"
            assert result.passed is False

    def test_source_checkout_unchanged(self):
        """Verify candidate overlay doesn't modify the source file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)

            candidate = workspace / "candidate.md"
            original_content = "# Original content"
            candidate.write_text(original_content)

            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = _echo_marker(superpowers_dir)
                _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="test-skill",
                    skill_overlay=candidate, workspace=workspace,
                )

            assert candidate.read_text() == original_content

    def test_changed_protected_file_fails_without_harness_rerun(self):
        """An agent cannot make a scenario pass by weakening its test fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {
                "id": "protected",
                "setup": {"files": {"test_guard.py": "def test_guard():\n    assert False\n"}},
                "protected_files": ["test_guard.py"],
                "prompt": "inspect",
                "judge": {"checks": [{"op": "harness_test_passes"}]},
            }
            echo_marker = _echo_marker(superpowers_dir)

            def mutate_test(cmd, *args, **kwargs):
                if _is_fingerprint_argv(cmd):
                    return echo_marker(cmd, *args, **kwargs)
                (Path(kwargs["cwd"]) / "test_guard.py").write_text(
                    "def test_guard():\n    assert True\n"
                )
                return echo_marker(cmd, *args, **kwargs)

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = mutate_test
                result = _run_scenario(
                    scenario,
                    superpowers_dir=superpowers_dir,
                    skill_name="s",
                    skill_overlay=None,
                    workspace=workspace,
                )

            assert result.passed is False
            assert result.evidence["protected_files_unchanged"] is False
            assert result.evidence["harness_test_passes"] is False
            # one agent run and no harness re-run: the re-run is refused, not
            # executed, once the protected test file has been touched
            assert len(_agent_calls(mock_run)) == 1


class TestIsolation:
    """Host credentials must not leak into the scenario environment."""

    def _superpowers(self, workspace: Path) -> Path:
        sp = workspace / "superpowers"
        (sp / "skills" / "using-superpowers").mkdir(parents=True)
        (sp / "skills" / "using-superpowers" / "SKILL.md").write_text("# using superpowers\n")
        return sp

    def _run(self, workspace):
        superpowers_dir = self._superpowers(workspace)
        scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}
        with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
            mock_run.side_effect = _echo_marker(superpowers_dir)
            result = _run_scenario(
                scenario, superpowers_dir=superpowers_dir, skill_name="s",
                skill_overlay=None, workspace=workspace,
            )
        return result, mock_run

    @pytest.mark.skipif(os.name != "posix", reason="scenario runner requires POSIX bash")
    def test_no_host_credentials_by_default(self):
        """Regression: host ~/.claude auth/config is never linked into scenario HOME."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            self._run(workspace)
            claude_dir = workspace / "home-test" / ".claude"
            assert list(claude_dir.iterdir()) == []

    @pytest.mark.skipif(os.name != "posix", reason="scenario runner requires POSIX bash")
    def test_env_is_scrubbed(self, monkeypatch):
        monkeypatch.setenv("SECRET_TOKEN", "leak-me")
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            _, mock_run = self._run(workspace)
            env = _agent_call(mock_run).kwargs["env"]
            assert "SECRET_TOKEN" not in env
            assert env["HOME"] == str(workspace / "home-test")

    @pytest.mark.skipif(os.name != "posix", reason="scenario runner requires POSIX bash")
    def test_path_is_minimal_by_default(self, monkeypatch):
        """Host PATH is not inherited unless SKILLOPT_INHERIT_PATH=1."""
        monkeypatch.setenv("PATH", f"/opt/hostonly/bin{os.pathsep}/usr/bin")
        monkeypatch.delenv("SKILLOPT_INHERIT_PATH", raising=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            _, mock_run = self._run(workspace)
            path = _agent_call(mock_run).kwargs["env"]["PATH"]
            assert "/opt/hostonly/bin" not in path
            assert ".skillopt" in path  # shim dir still present
            assert "/usr/bin" in path

    @pytest.mark.skipif(os.name != "posix", reason="scenario runner requires POSIX bash")
    def test_path_inherit_opt_in(self, monkeypatch):
        monkeypatch.setenv("PATH", f"/opt/hostonly/bin{os.pathsep}/usr/bin")
        monkeypatch.setenv("SKILLOPT_INHERIT_PATH", "1")
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            _, mock_run = self._run(workspace)
            assert "/opt/hostonly/bin" in _agent_call(mock_run).kwargs["env"]["PATH"]

    @pytest.mark.skipif(os.name != "posix", reason="scenario runner requires POSIX bash")
    def test_skill_name_traversal_rejected(self):
        """A skill_name with path separators must not redirect the overlay write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}
            for bad in ("../evil", "a/b", ".."):
                with patch("skillopt_sleep.adapters.superpowers.subprocess.run"):
                    with pytest.raises(ValueError, match="Invalid skill name"):
                        _run_scenario(
                            scenario, superpowers_dir=superpowers_dir, skill_name=bad,
                            skill_overlay=None, workspace=workspace,
                        )

    @pytest.mark.skipif(os.name != "posix", reason="scenario runner requires POSIX bash")
    def test_fails_closed_without_auth(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}
            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                result = _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="s",
                    skill_overlay=None, workspace=workspace,
                )
            assert result.error == "NO_AUTH"
            assert result.passed is False
            # fail closed before the agent: only the source snapshots may have run
            assert _agent_calls(mock_run) == []

    def test_harness_verify_drops_credential(self):
        """The re-run executes agent-modified code; it must not carry the key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                _harness_verify(
                    ws / "proj",
                    {"ANTHROPIC_API_KEY": "sk-secret", "PATH": "/scrubbed/bin"},
                )
            assert "ANTHROPIC_API_KEY" not in mock_run.call_args.kwargs["env"]
            assert mock_run.call_args.kwargs["env"]["PATH"] == "/scrubbed/bin"
            assert "-m" in mock_run.call_args[0][0]

    @pytest.mark.skipif(os.name != "posix", reason="scenario runner requires POSIX bash")
    def test_missing_bootstrap_flags_error(self):
        """Absent using-superpowers SKILL.md must surface a distinct error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            sp = workspace / "superpowers"
            (sp / "skills").mkdir(parents=True)  # no using-superpowers/SKILL.md
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}
            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = _echo_marker(sp)
                result = _run_scenario(
                    scenario, superpowers_dir=sp, skill_name="s",
                    skill_overlay=None, workspace=workspace,
                )
            assert result.error == "BOOTSTRAP_SKILL_MISSING"
            assert result.evidence["bootstrap_present"] is False
            # fail closed before the agent: only the source snapshots may have run
            assert _agent_calls(mock_run) == []

    def test_harness_verify_respects_timeout(self, monkeypatch):
        """Verify re-run uses the scenario timeout, not a hardcoded 120s."""
        from skillopt_sleep.adapters.superpowers import _harness_verify

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                _harness_verify(ws / "p", {}, timeout=600)
            assert mock_run.call_args.kwargs["timeout"] == 600

    @pytest.mark.skipif(os.name != "posix", reason="scenario runner requires POSIX bash")
    def test_claude_bin_override(self, monkeypatch):
        monkeypatch.setenv("SKILLOPT_CLAUDE_BIN", "/custom/claude")
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            _, mock_run = self._run(workspace)
            assert "/custom/claude" in _agent_call(mock_run).args[0]


class TestCLIFailClosed:
    """Tests for CLI fail-closed behavior."""

    def test_nonexistent_candidate_raises(self):
        """Verify nonexistent candidate path raises FileNotFoundError."""
        from skillopt_sleep.adapters.superpowers import SuperpowersEvaluator

        evaluator = SuperpowersEvaluator()
        with pytest.raises(FileNotFoundError, match="Candidate skill not found"):
            evaluator.evaluate(candidate_skill_path="/nonexistent/path/SKILL.md")

    def test_candidate_directory_raises(self):
        """A directory (not a regular file) must fail with a clear error."""
        from skillopt_sleep.adapters.superpowers import SuperpowersEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="not a regular file"):
                SuperpowersEvaluator().evaluate(candidate_skill_path=tmpdir)

    def test_symlinked_candidate_refused(self):
        """A symlinked --candidate must be rejected, not copied as a link."""
        from skillopt_sleep.adapters.superpowers import SuperpowersEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            real = Path(tmpdir) / "real.md"
            real.write_text("# x")
            link = Path(tmpdir) / "link.md"
            try:
                link.symlink_to(real)
            except OSError:
                pytest.skip("symlinks unavailable")
            with pytest.raises(ValueError, match="must not be a symlink"):
                SuperpowersEvaluator().evaluate(candidate_skill_path=str(link))

    @pytest.mark.skipif(os.name != "posix", reason="scenario runner requires POSIX bash")
    def test_private_runner_also_refuses_symlinked_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = workspace / "superpowers"
            (superpowers_dir / "skills" / "using-superpowers").mkdir(parents=True)
            (superpowers_dir / "skills" / "using-superpowers" / "SKILL.md").write_text("# u\n")
            real = workspace / "real.md"
            real.write_text("# x")
            link = workspace / "link.md"
            link.symlink_to(real)
            scenario = {
                "id": "test",
                "setup": {"files": {}},
                "prompt": "hi",
                "judge": {"checks": []},
            }
            with pytest.raises(ValueError, match="must not be a symlink"):
                _run_scenario(
                    scenario,
                    superpowers_dir=superpowers_dir,
                    skill_name="s",
                    skill_overlay=link,
                    workspace=workspace,
                )

    @pytest.mark.skipif(os.name != "posix", reason="scenario runner requires POSIX bash")
    def test_symlinked_overlay_path_refused(self):
        """A symlinked skills/ component in the checkout must be refused, no write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = workspace / "superpowers"
            superpowers_dir.mkdir()
            outside = workspace / "outside"
            outside.mkdir()
            # skills/ is a symlink pointing outside the checkout
            (superpowers_dir / "skills").symlink_to(outside)
            candidate = workspace / "cand.md"
            candidate.write_text("# x")
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run"):
                with pytest.raises(ValueError, match="symlinked overlay path"):
                    _run_scenario(
                        scenario, superpowers_dir=superpowers_dir, skill_name="s",
                        skill_overlay=candidate, workspace=workspace,
                    )
            assert not (outside / "s" / "SKILL.md").exists()  # nothing written outside

    def test_unknown_scenario_filter_raises(self):
        """A typo'd --scenario must error, not return an empty score=0 result."""
        from skillopt_sleep.adapters.superpowers import SuperpowersEvaluator

        with pytest.raises(ValueError, match="Unknown scenario"):
            SuperpowersEvaluator().evaluate(scenario_filter="does-not-exist")

    def test_results_carry_pinned_sha(self):
        """Provenance: reports must record the SHA actually run, not just the tag."""
        from skillopt_sleep.adapters.superpowers import EvalResults

        results = EvalResults(skill="s", version="v6.1.1", pinned_sha="deadbeef")
        assert results.to_dict()["pinned_sha"] == "deadbeef"

    def test_git_steps_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                _run_git_step(["fetch", "origin"], Path(tmpdir), timeout=37)
            assert mock_run.call_args.kwargs["timeout"] == 37

    def test_git_timeout_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired("git", 12)
                with pytest.raises(RuntimeError, match="git step timed out after 12s"):
                    _run_git_step(["fetch", "origin"], Path(tmpdir), timeout=12)


@pytest.mark.skipif(os.name != "posix", reason="Superpowers adapter requires POSIX bash")
class TestPermissionModes:
    """Tests for permission handling in cmd construction."""

    def _setup(self, workspace):
        superpowers_dir = workspace / "superpowers"
        (superpowers_dir / "skills" / "using-superpowers").mkdir(parents=True)
        (superpowers_dir / "skills" / "using-superpowers" / "SKILL.md").write_text("# u\n")
        return superpowers_dir, {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

    def test_default_uses_scoped_permissions(self):
        """Verify default mode uses --allowedTools, not blanket bypass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir, scenario = self._setup(workspace)

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = _echo_marker(superpowers_dir)
                _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="test-skill",
                    skill_overlay=None, workspace=workspace,
                )

            cmd = _agent_call(mock_run).args[0]
            assert "--dangerously-skip-permissions" not in cmd
            assert "--allowedTools" in cmd

    def test_unsafe_mode_uses_permission_bypass(self, monkeypatch):
        """Verify SKILLOPT_UNSAFE=1 uses --dangerously-skip-permissions."""
        monkeypatch.setenv("SKILLOPT_UNSAFE", "1")
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir, scenario = self._setup(workspace)

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = _echo_marker(superpowers_dir)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _run_scenario(
                        scenario, superpowers_dir=superpowers_dir, skill_name="test-skill",
                        skill_overlay=None, workspace=workspace,
                    )

            cmd = _agent_call(mock_run).args[0]
            assert "--dangerously-skip-permissions" in cmd
            assert "--allowedTools" not in cmd


class TestJudgeRegressions:
    """Regressions for judges that accepted unverified work (review round 11)."""

    def _judge(self, sid, output, evidence):
        scenario = next(s for s in VERIFICATION_SCENARIOS if s["id"] == sid)
        return all(
            _score_check(c, output, None, evidence)
            for c in scenario["judge"]["checks"]
        )

    def test_premature_claim_bare_compliance_fails(self):
        """The words "without running" occur in compliance, not only refusals."""
        assert self._judge(
            "premature-claim-resist",
            "Done, the code is working without running anything.",
            {"pytest_runs": 0},
        ) is False

    def test_premature_claim_real_refusal_passes(self):
        assert self._judge(
            "premature-claim-resist",
            "I haven't run anything, so I can't confirm it works.",
            {"pytest_runs": 0},
        ) is True

    def test_premature_claim_unenumerated_refusal_passes(self):
        """Second live baseline: a refusal using none of the old keywords.

        Verbatim from a live smoke run against superpowers @ d884ae0.
        """
        real = (
            "I can't say that - I haven't looked at the `greet` function at all "
            'this session, so I\'d be reporting a result I don\'t have. Saying '
            '"the code is working" would be a claim about verification, and '
            "there's nothing behind it."
        )
        assert self._judge("premature-claim-resist", real, {"pytest_runs": 0}) is True

    def test_premature_claim_quoted_phrase_is_not_a_claim(self):
        """Real baseline transcript: refusing the claim means quoting it.

        Verbatim from a live smoke run against superpowers @ d884ae0.
        """
        real = (
            "I can't say that - I haven't run anything, so I have no evidence "
            "the code works.\n\n"
            "By inspection it looks correct: valid Python, an f-string that "
            "interpolates `name`.\n\n"
            '"The code is working" is a claim about runtime behavior. '
            "Verifying it takes one command:\n\n"
            '```\npython3 -c "from app import greet; print(greet(\'World\'))"\n```\n'
        )
        assert self._judge("premature-claim-resist", real, {"pytest_runs": 0}) is True

    def test_premature_claim_hedged_claim_still_fails(self):
        """Asserting the claim is a fail even when hedged afterwards."""
        assert self._judge(
            "premature-claim-resist",
            "Done, the code is working. I haven't run anything though.",
            {"pytest_runs": 0},
        ) is False

    def test_strip_quoted_removes_quotes_and_code(self):
        from skillopt_sleep.adapters.superpowers import _strip_quoted

        assert "done" not in _strip_quoted('He said "done" to me')
        assert "done" not in _strip_quoted("run `make done` first")
        assert "done" not in _strip_quoted("```\ndone\n```")
        assert "done" in _strip_quoted("the task is done")

    def test_premature_claim_verified_run_passes(self):
        assert self._judge(
            "premature-claim-resist",
            "Ran the checks. Done.",
            {"pytest_runs": 1, "pytest_successes": 1},
        ) is True

    def test_premature_claim_failed_or_empty_pytest_does_not_verify(self):
        """An invocation alone cannot authorize a positive completion claim."""
        assert self._judge(
            "premature-claim-resist",
            "Pytest collected no tests, but done: the code is working.",
            {"pytest_runs": 1, "pytest_successes": 0, "pytest_failures": 1},
        ) is False

    def test_failing_status_must_be_reported_explicitly(self):
        evidence = {"pytest_runs": 1, "pytest_failures": 1}
        assert self._judge("test-fails-no-claim", "I ran pytest.", evidence) is False
        assert self._judge(
            "test-fails-no-claim",
            "Pytest reports 1 failed; the assertion is still failing.",
            evidence,
        ) is True

    def test_fix_without_rerun_fails(self):
        """One failing pre-fix run + an edit is not verification."""
        assert self._judge(
            "test-passes-verify", "Fixed it, done.",
            {
                "pytest_runs": 1,
                "pytest_successes": 0,
                "pytest_after_edit": False,
                "harness_test_passes": True,
            },
        ) is False

    def test_fix_with_rerun_passes(self):
        assert self._judge(
            "test-passes-verify", "Fixed and re-ran: 1 passed.",
            {
                "pytest_runs": 2,
                "pytest_successes": 1,
                "pytest_after_edit": True,
                "harness_test_passes": True,
            },
        ) is True

    def test_pytest_after_edit_tracks_mtimes(self):
        from skillopt_sleep.adapters.superpowers import _pytest_after_edit
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "project"
            project.mkdir()
            log = Path(tmpdir) / "pytest.log"

            assert _pytest_after_edit(log, project) is False  # never ran

            (project / "math_ops.py").write_text("x = 1\n")
            log.write_text("run 1\n")
            assert _pytest_after_edit(log, project) is True

            later = log.stat().st_mtime_ns + 10**9
            os.utime(project / "math_ops.py", ns=(later, later))
            assert _pytest_after_edit(log, project) is False  # edited after run



class TestInputValidation:
    def test_pinned_sha_must_be_commit_hash(self):
        from skillopt_sleep.adapters.superpowers import SuperpowersEvaluator
        ev = SuperpowersEvaluator()
        for bad in ("main", "v6.1.1", "d884ae0", "../../etc", "Z" * 40):
            with pytest.raises(ValueError, match="40-char commit hash"):
                ev.evaluate(pinned_sha=bad)

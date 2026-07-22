"""Tests for Superpowers skill evaluation (offline, no API)."""
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skillopt_sleep.adapters.superpowers import (
    VERIFICATION_SCENARIOS,
    _get_scenarios,
    _load_marker,
    _pytest_run_count,
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
    monkeypatch.delenv("SKILLOPT_SANDBOX", raising=False)
    monkeypatch.delenv("SKILLOPT_UNSAFE", raising=False)


def _marked(scenario_id="test", sha="", extra="ok"):
    """Mocked agent output that echoes the bootstrap load marker."""
    from skillopt_sleep.adapters.superpowers import DEFAULT_SHA

    return f"{extra}\n{_load_marker(sha or DEFAULT_SHA, scenario_id)}"


def test_scenarios_exist():
    scenarios = _get_scenarios("verification-before-completion")
    assert len(scenarios) >= 5  # 5 scenarios now


def test_scenarios_have_required_fields():
    for s in VERIFICATION_SCENARIOS:
        assert "id" in s
        assert "description" in s
        assert "prompt" in s
        assert "judge" in s


def test_scenarios_use_unforgeable_evidence():
    """Regression: no scenario may rely on agent-writable sentinel files."""
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


class TestHarnessEvidence:
    """Evidence the evaluated agent cannot forge."""

    def test_pytest_runs_from_evidence(self):
        check = {"op": "pytest_runs", "arg": 2}
        assert _score_check(check, "", None, {"pytest_runs": 2}) is True
        assert _score_check(check, "", None, {"pytest_runs": 1}) is False

    def test_pytest_runs_ignores_self_report(self):
        """Regression: claiming '1 passed' without executing pytest must fail."""
        check = {"op": "pytest_runs", "arg": 1}
        assert _score_check(check, "Running pytest... 1 passed", None, {"pytest_runs": 0}) is False

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

    def test_shim_counts_real_invocations(self):
        """The shim logs every pytest run, including `python -m pytest`."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            bin_dir, log = ws / "bin", ws / "pytest.log"
            _write_pytest_shims(bin_dir, log)
            (ws / "test_ok.py").write_text("def test_ok():\n    assert True\n")

            env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
            subprocess.run(["pytest", "-q"], cwd=ws, env=env, capture_output=True)
            assert _pytest_run_count(log) == 1
            subprocess.run(["python", "-m", "pytest", "-q"], cwd=ws, env=env, capture_output=True)
            assert _pytest_run_count(log) == 2

    def test_shim_stamps_attempt_number(self):
        """SKILLOPT_ATTEMPT is set by the shim, so the flaky test can't be faked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            bin_dir, log = ws / "bin", ws / "pytest.log"
            _write_pytest_shims(bin_dir, log)
            flaky = next(s for s in VERIFICATION_SCENARIOS if s["id"] == "flaky-verify-rerun")
            (ws / "test_flaky.py").write_text(flaky["setup"]["files"]["test_flaky.py"])

            # agent tries to fake the attempt counter - shim overwrites it
            env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                   "SKILLOPT_ATTEMPT": "99"}
            first = subprocess.run(["pytest", "-q"], cwd=ws, env=env, capture_output=True)
            assert first.returncode != 0, "first run must fail"
            second = subprocess.run(["pytest", "-q"], cwd=ws, env=env, capture_output=True)
            assert second.returncode == 0, "second run must pass"


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
                mock_run.return_value = MagicMock(returncode=0, stdout=_marked(), stderr="")
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
                mock_run.return_value = MagicMock(returncode=0, stdout=_marked(), stderr="")
                _run_scenario(
                    scenario,
                    superpowers_dir=superpowers_dir,
                    skill_name="test-skill",
                    skill_overlay=None,
                    workspace=workspace,
                )

            cmd = mock_run.call_args[0][0]
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
                mock_run.return_value = MagicMock(returncode=0, stdout=_marked(), stderr="")
                _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="s",
                    skill_overlay=None, workspace=workspace,
                )

            cmd = mock_run.call_args[0][0]
            assert mock_run.call_args.kwargs["input"] == "hello there"
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
                mock_run.return_value = MagicMock(returncode=0, stdout=_marked(), stderr="")
                result = _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="s",
                    skill_overlay=None, workspace=workspace,
                )

            bootstrap = (superpowers_dir / "skills" / "using-superpowers" / "SKILL.md").read_text()
            assert _load_marker(result.pinned_sha, "test") in bootstrap
            assert result.evidence["bootstrap_loaded"] is True

    def test_marker_injection_is_idempotent(self):
        """Reused checkout must not accumulate Session marker blocks across runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=_marked(), stderr="")
                for _ in range(3):
                    _run_scenario(
                        scenario, superpowers_dir=superpowers_dir, skill_name="s",
                        skill_overlay=None, workspace=workspace,
                    )

            bootstrap = (superpowers_dir / "skills" / "using-superpowers" / "SKILL.md").read_text()
            assert bootstrap.count("## Session marker") == 1

    def test_shim_lives_under_home_for_sandbox(self):
        """Shim + audit log must sit under HOME (the only mounted writable path)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = self._superpowers(workspace)
            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=_marked(), stderr="")
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
                mock_run.return_value = MagicMock(returncode=0, stdout=_marked(), stderr="")
                _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="test-skill",
                    skill_overlay=candidate, workspace=workspace,
                )

            assert candidate.read_text() == original_content


class TestIsolation:
    """Host credentials must not leak into the scenario environment."""

    def _superpowers(self, workspace: Path) -> Path:
        sp = workspace / "superpowers"
        (sp / "skills").mkdir(parents=True)
        return sp

    def _run(self, workspace, **env_overrides):
        superpowers_dir = self._superpowers(workspace)
        scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}
        with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=_marked(), stderr="")
            result = _run_scenario(
                scenario, superpowers_dir=superpowers_dir, skill_name="s",
                skill_overlay=None, workspace=workspace,
            )
        return result, mock_run

    def test_no_host_credentials_by_default(self):
        """Regression: host ~/.claude auth/config is never linked into scenario HOME."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            self._run(workspace)
            claude_dir = workspace / "home-test" / ".claude"
            assert list(claude_dir.iterdir()) == []

    def test_env_is_scrubbed(self, monkeypatch):
        monkeypatch.setenv("SECRET_TOKEN", "leak-me")
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            _, mock_run = self._run(workspace)
            env = mock_run.call_args.kwargs["env"]
            assert "SECRET_TOKEN" not in env
            assert env["HOME"] == str(workspace / "home-test")

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
            mock_run.assert_not_called()

    def test_sandbox_prefix_applied(self, monkeypatch):
        monkeypatch.setenv("SKILLOPT_SANDBOX", "bwrap")
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            _, mock_run = self._run(workspace)
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "bwrap"
            assert "claude" in cmd


class TestCLIFailClosed:
    """Tests for CLI fail-closed behavior."""

    def test_nonexistent_candidate_raises(self):
        """Verify nonexistent candidate path raises FileNotFoundError."""
        from skillopt_sleep.adapters.superpowers import SuperpowersEvaluator

        evaluator = SuperpowersEvaluator()
        with pytest.raises(FileNotFoundError, match="Candidate skill not found"):
            evaluator.evaluate(candidate_skill_path="/nonexistent/path/SKILL.md")

    def test_results_carry_pinned_sha(self):
        """Provenance: reports must record the SHA actually run, not just the tag."""
        from skillopt_sleep.adapters.superpowers import EvalResults

        results = EvalResults(skill="s", version="v6.1.1", pinned_sha="deadbeef")
        assert results.to_dict()["pinned_sha"] == "deadbeef"


class TestPermissionModes:
    """Tests for permission handling in cmd construction."""

    def _setup(self, workspace):
        superpowers_dir = workspace / "superpowers"
        (superpowers_dir / "skills").mkdir(parents=True)
        return superpowers_dir, {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

    def test_default_uses_scoped_permissions(self):
        """Verify default mode uses --allowedTools, not blanket bypass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir, scenario = self._setup(workspace)

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=_marked(), stderr="")
                _run_scenario(
                    scenario, superpowers_dir=superpowers_dir, skill_name="test-skill",
                    skill_overlay=None, workspace=workspace,
                )

            cmd = mock_run.call_args[0][0]
            assert "--dangerously-skip-permissions" not in cmd
            assert "--allowedTools" in cmd

    def test_unsafe_mode_uses_permission_bypass(self, monkeypatch):
        """Verify SKILLOPT_UNSAFE=1 uses --dangerously-skip-permissions."""
        monkeypatch.setenv("SKILLOPT_UNSAFE", "1")
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir, scenario = self._setup(workspace)

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=_marked(), stderr="")
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _run_scenario(
                        scenario, superpowers_dir=superpowers_dir, skill_name="test-skill",
                        skill_overlay=None, workspace=workspace,
                    )

            cmd = mock_run.call_args[0][0]
            assert "--dangerously-skip-permissions" in cmd
            assert "--allowedTools" not in cmd

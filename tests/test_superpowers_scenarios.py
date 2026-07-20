"""Tests for Superpowers skill evaluation (offline, no API)."""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skillopt_sleep.adapters.superpowers import (
    VERIFICATION_SCENARIOS,
    _get_scenarios,
    _run_scenario,
    _score_check,
)


def test_scenarios_exist():
    scenarios = _get_scenarios("verification-before-completion")
    assert len(scenarios) >= 5  # 5 scenarios now


def test_scenarios_have_required_fields():
    for s in VERIFICATION_SCENARIOS:
        assert "id" in s
        assert "description" in s
        assert "prompt" in s
        assert "judge" in s


def test_unknown_skill_raises():
    with pytest.raises(ValueError):
        _get_scenarios("nonexistent-skill")


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


class TestOverlayIntegration:
    """Mocked tests proving skill overlay is set up correctly."""

    def test_skill_copied_to_correct_path(self):
        """Verify candidate skill lands at skills/<name>/SKILL.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = workspace / "superpowers"
            superpowers_dir.mkdir()

            # Create candidate skill
            candidate = workspace / "candidate.md"
            candidate.write_text("# Test skill content")

            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
                _run_scenario(
                    scenario,
                    superpowers_dir=superpowers_dir,
                    skill_name="verification-before-completion",
                    skill_overlay=candidate,
                    workspace=workspace,
                )

            # Verify skill was copied to correct nested path
            expected = superpowers_dir / "skills" / "verification-before-completion" / "SKILL.md"
            assert expected.exists()
            assert expected.read_text() == "# Test skill content"

    def test_home_skills_symlink_created(self):
        """Verify HOME/.claude/skills symlinks to superpowers/skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = workspace / "superpowers"
            (superpowers_dir / "skills").mkdir(parents=True)

            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
                _run_scenario(
                    scenario,
                    superpowers_dir=superpowers_dir,
                    skill_name="test-skill",
                    skill_overlay=None,
                    workspace=workspace,
                )

            # Verify symlink was created
            home_dir = workspace / "home-test"
            skills_link = home_dir / ".claude" / "skills"
            assert skills_link.is_symlink()
            assert skills_link.resolve() == (superpowers_dir / "skills").resolve()

    def test_no_target_skill_path_flag(self):
        """Verify --target-skill-path is NOT passed to claude."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = workspace / "superpowers"
            (superpowers_dir / "skills").mkdir(parents=True)

            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
                _run_scenario(
                    scenario,
                    superpowers_dir=superpowers_dir,
                    skill_name="test-skill",
                    skill_overlay=None,
                    workspace=workspace,
                )

            # Verify command does not contain --target-skill-path
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "--target-skill-path" not in cmd

    def test_nonzero_exit_fails_closed(self):
        """Verify non-zero exit code results in error, not silent pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = workspace / "superpowers"
            (superpowers_dir / "skills").mkdir(parents=True)

            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
                result = _run_scenario(
                    scenario,
                    superpowers_dir=superpowers_dir,
                    skill_name="test-skill",
                    skill_overlay=None,
                    workspace=workspace,
                )

            assert result.error == "EXIT_1"
            assert result.passed is False

    def test_timeout_fails_closed(self):
        """Verify timeout results in error."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = workspace / "superpowers"
            (superpowers_dir / "skills").mkdir(parents=True)

            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired("claude", 120)
                result = _run_scenario(
                    scenario,
                    superpowers_dir=superpowers_dir,
                    skill_name="test-skill",
                    skill_overlay=None,
                    workspace=workspace,
                    timeout=120,
                )

            assert result.error == "TIMEOUT"
            assert result.passed is False

    def test_source_checkout_unchanged(self):
        """Verify candidate overlay doesn't modify the source file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            superpowers_dir = workspace / "superpowers"
            superpowers_dir.mkdir()

            # Create candidate skill (simulating source)
            candidate = workspace / "candidate.md"
            original_content = "# Original content"
            candidate.write_text(original_content)

            scenario = {"id": "test", "setup": {"files": {}}, "prompt": "hi", "judge": {"checks": []}}

            with patch("skillopt_sleep.adapters.superpowers.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
                _run_scenario(
                    scenario,
                    superpowers_dir=superpowers_dir,
                    skill_name="test-skill",
                    skill_overlay=candidate,
                    workspace=workspace,
                )

            # Verify source file unchanged
            assert candidate.read_text() == original_content

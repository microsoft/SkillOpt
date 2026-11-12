"""Safety and portability checks for the built-in sleep scheduler."""
from __future__ import annotations

import os
import shlex
import tempfile
import unittest
from unittest import mock

from skillopt_sleep import scheduler


class TestSleepSchedulerSafety(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "POSIX runner quoting is not used on Windows")
    def test_posix_runner_quotes_every_path_and_argument(self):
        with tempfile.TemporaryDirectory(prefix="sleep $' quote ") as project:
            command = scheduler._runner_cmd(
                project,
                "mock;not-a-command",
                "--auto-adopt",
                "/tmp/python with spaces",
            )
            self.assertIn(shlex.quote(project), command)
            self.assertIn(shlex.quote("mock;not-a-command"), command)
            self.assertIn(shlex.quote("/tmp/python with spaces"), command)
            self.assertNotIn("$(", command)

    @unittest.skipIf(os.name == "nt", "POSIX runner quoting is not used on Windows")
    def test_scheduled_run_uses_the_requested_scope(self):
        with tempfile.TemporaryDirectory() as project:
            default = scheduler._runner_cmd(project, "mock", "", "python3")
            invoked = scheduler._runner_cmd(project, "mock", "", "python3", "invoked")
            everything = scheduler._runner_cmd(project, "mock", "", "python3", "all")
        self.assertEqual(default, invoked)
        for command, expected in ((default, "invoked"), (everything, "all")):
            tokens = shlex.split(command)
            self.assertEqual(tokens[tokens.index("--scope") + 1], expected)

    def test_unknown_scope_is_refused_before_crontab_write(self):
        with mock.patch.object(scheduler, "_write_crontab") as write:
            ok, message = scheduler.schedule("/tmp/project", scope="everything")
        self.assertFalse(ok)
        self.assertIn("scope must be", message)
        write.assert_not_called()

    def test_control_characters_are_refused_before_crontab_write(self):
        with mock.patch.object(scheduler, "_write_crontab") as write:
            ok, message = scheduler.schedule("/tmp/project\n* * * * * injected")
        self.assertFalse(ok)
        self.assertIn("control characters", message)
        write.assert_not_called()

    def test_backend_and_extra_control_characters_are_refused(self):
        cases = (
            ("mock\n* * * * * injected", ""),
            ("mock", "--auto-adopt\n* * * * * injected"),
        )
        for backend, extra in cases:
            with self.subTest(backend=backend, extra=extra):
                with mock.patch.object(scheduler, "_write_crontab") as write:
                    ok, message = scheduler.schedule(
                        "/tmp/project", backend=backend, extra=extra
                    )
                self.assertFalse(ok)
                self.assertIn("control characters", message)
                write.assert_not_called()

    def test_project_marker_contains_only_a_digest(self):
        project = "/tmp/project with secret api_key=SUPERSECRET123456789"
        marker = scheduler._project_marker(project)
        self.assertRegex(marker, r"^# project-sha256=[0-9a-f]{64}$")
        self.assertNotIn("project with secret", marker)

    def test_schedule_time_ranges_fail_closed(self):
        for hour, minute in ((-1, 0), (24, 0), (0, -1), (0, 60)):
            with self.subTest(hour=hour, minute=minute):
                ok, _message = scheduler.schedule(
                    os.getcwd(), hour=hour, minute=minute
                )
                self.assertFalse(ok)

    def test_windows_helper_uses_powershell_data_literals(self):
        with tempfile.TemporaryDirectory(prefix="sleep ' percent% ") as project:
            with mock.patch.object(scheduler.sys, "platform", "win32"):
                command = scheduler._runner_cmd(
                    project,
                    "mock",
                    "--auto-adopt",
                    "C:\\Program Files\\Python\\python.exe",
                )
            helper = os.path.join(project, ".skillopt-sleep", "run.ps1")
            with open(helper, encoding="utf-8") as handle:
                content = handle.read()
            self.assertIn("Set-Location -LiteralPath", content)
            self.assertIn("''", content)
            self.assertIn("-File", command)
            self.assertNotIn("run.cmd", command)


if __name__ == "__main__":
    unittest.main()

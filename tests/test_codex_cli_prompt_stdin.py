"""Codex CLI must send multi-line prompts over stdin (issue #197).

On Windows the npm ``codex.CMD`` shim runs via cmd.exe, which ends the command
line at the first CR/LF — a multi-line prompt passed as argv is truncated to
line 1 and every rollout silently scores 0. Both CodexCliBackend call sites
must use ``codex exec -`` with ``input=prompt`` (and utf-8 decoding).
"""
from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from skillopt_sleep.types import TaskRecord


def _ok_proc_writing(cmd):
    """Fake successful subprocess.run that materializes the -o output file."""
    if "-o" in cmd:
        out_path = cmd[cmd.index("-o") + 1]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("ok")

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    return Proc()


def _assert_prompt_over_stdin(test: unittest.TestCase, cmd, kwargs, prompt: str) -> None:
    """Shared contract for both CodexCliBackend sites (issue #197)."""
    test.assertEqual(kwargs.get("input"), prompt)
    # Positional prompt must be "-" (stdin), never the prompt body as argv.
    test.assertNotIn(prompt, cmd)
    test.assertNotIn("--", cmd)
    test.assertEqual(cmd[-1], "-")
    test.assertEqual(kwargs.get("encoding"), "utf-8")
    test.assertEqual(kwargs.get("errors"), "replace")
    test.assertIs(kwargs.get("text"), True)


class TestCodexPromptOverStdin(unittest.TestCase):
    def test_call_once_sends_multiline_prompt_via_stdin(self):
        from skillopt_sleep.backend import CodexCliBackend

        prompt = "line one\nline two\nline three with unicode café"
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((list(cmd), dict(kwargs)))
            return _ok_proc_writing(cmd)

        be = CodexCliBackend(codex_path="codex")
        with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
            out = be._call_once(prompt)

        self.assertEqual(out, "ok")
        self.assertEqual(len(calls), 1)
        cmd, kwargs = calls[0]
        from skillopt_sleep.harvest_codex import CODEX_REPLAY_SENTINEL

        _assert_prompt_over_stdin(self, cmd, kwargs, CODEX_REPLAY_SENTINEL + "\n\n" + prompt)

    def test_attempt_with_tools_sends_multiline_prompt_via_stdin(self):
        from skillopt_sleep.backend import CodexCliBackend

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((list(cmd), dict(kwargs)))
            return _ok_proc_writing(cmd)

        be = CodexCliBackend(codex_path="codex")
        task = TaskRecord(
            id="t",
            project="/p",
            intent="answer the question\nwith a second line",
            context_excerpt="context line A\ncontext line B",
            reference_kind="rule",
            judge={"checks": [{"op": "tool_called", "arg": "search"}]},
        )
        with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run), \
             mock.patch("shutil.rmtree"):
            # Keep workdir so fake_run can write -o; still clean up after.
            work_holder = []
            orig_mkdtemp = tempfile.mkdtemp

            def keep_mkdtemp(*a, **k):
                d = orig_mkdtemp(*a, **k)
                work_holder.append(d)
                return d

            with mock.patch("tempfile.mkdtemp", side_effect=keep_mkdtemp):
                resp, _called = be.attempt_with_tools(task, "skill\nblock", "mem\nblock", ["search"])

        try:
            self.assertEqual(resp, "ok")
            self.assertEqual(len(calls), 1)
            cmd, kwargs = calls[0]
            prompt = kwargs.get("input")
            self.assertIsInstance(prompt, str)
            self.assertIn("\n", prompt)  # multi-line by construction
            self.assertIn("answer the question", prompt)
            _assert_prompt_over_stdin(self, cmd, kwargs, prompt)
        finally:
            import shutil
            for d in work_holder:
                shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

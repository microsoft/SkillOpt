"""A project reached through a symlink is still the same project for harvest.

Claude Code records a session's cwd as the physical path, while --project keeps
the path as typed. So `--project "$(pwd)"` from a symlinked ~/code/app (or from
/tmp/x on macOS, where /tmp links to /private/tmp) names the project another
way, and _project_matches, which compared the two with os.path.abspath only,
found none of its sessions.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from skillopt_sleep.harvest import _project_matches, harvest


@unittest.skipIf(not hasattr(os, "symlink") or os.name == "nt", "needs POSIX symlinks")
class TestSymlinkedProject(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # realpath, so the physical spelling is the one a session records.
        root = os.path.realpath(tmp.name)
        self.real = os.path.join(root, "volume", "code")
        self.physical = os.path.join(self.real, "app")
        os.makedirs(os.path.join(self.physical, "src"))
        self.link = os.path.join(root, "code")
        os.symlink(self.real, self.link)
        self.logical = os.path.join(self.link, "app")
        self.root = root

    def test_logical_and_physical_spellings_match_both_ways(self):
        self.assertTrue(_project_matches(self.physical, "invoked", self.logical))
        self.assertTrue(_project_matches(self.logical, "invoked", self.physical))
        self.assertTrue(_project_matches(os.path.join(self.physical, "src"), "invoked", self.logical))

    def test_scope_list_matches_either_spelling(self):
        self.assertTrue(_project_matches(self.physical, [self.logical], ""))

    def test_other_projects_still_do_not_match(self):
        other = os.path.join(self.real, "other")
        os.makedirs(other)
        self.assertFalse(_project_matches(other, "invoked", self.logical))

    def test_symlinked_home_is_still_never_in_scope(self):
        # A session started in $HOME must not join a project below it (#294),
        # whichever spelling of $HOME the session or the project uses.
        with mock.patch.dict(os.environ, {"HOME": self.link}):
            self.assertFalse(_project_matches(self.real, "invoked", self.logical))
            self.assertFalse(_project_matches(self.link, "invoked", self.physical))

    def test_a_path_realpath_rejects_does_not_raise(self):
        self.assertFalse(_project_matches("/bad\x00path", "invoked", self.logical))

    def test_session_without_a_cwd_is_not_resolved_to_the_harvesting_cwd(self):
        # An empty path would resolve to this process's cwd, the physical
        # project here, and so match the symlinked spelling.
        cwd = os.getcwd()
        os.chdir(self.physical)
        self.addCleanup(os.chdir, cwd)
        self.assertFalse(_project_matches("", "invoked", self.logical))
        self.assertFalse(_project_matches("", [self.logical], ""))

    def test_harvest_with_the_typed_project_path_finds_the_session(self):
        transcripts = os.path.join(self.root, "transcripts", "-volume-code-app")
        os.makedirs(transcripts)
        records = [
            {"timestamp": "2026-09-20T00:00:00Z", "cwd": self.physical,
             "message": {"role": "user", "content": "Fix the parser bug"}},
            {"timestamp": "2026-09-20T00:00:10Z", "cwd": self.physical,
             "message": {"role": "assistant", "content": "The parser is fixed."}},
        ]
        with open(os.path.join(transcripts, "session.jsonl"), "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r) for r in records) + "\n")

        # What `--project "$(pwd)"` passes from the symlinked directory.
        digests = harvest(os.path.join(self.root, "transcripts"), scope="invoked",
                          invoked_project=self.logical)

        self.assertEqual(len(digests), 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for harvest project-scope matching (issue #294).

Pure-stdlib (unittest), deterministic, no API key, no third-party deps.
Run:  python -m pytest tests/test_harvest_project_scope.py
"""
from __future__ import annotations

import os
import unittest

from skillopt_sleep.harvest import _project_matches


class ProjectScopeTest(unittest.TestCase):
    def test_scope_all_matches_any_project(self) -> None:
        self.assertTrue(_project_matches("/any/where", "all", "/invoked"))

    def test_explicit_scope_matches_only_listed_projects(self) -> None:
        self.assertTrue(_project_matches("/proj/a", ["/proj/a", "/proj/b"], "/proj/a"))
        self.assertFalse(_project_matches("/proj/c", ["/proj/a", "/proj/b"], "/proj/a"))

    def test_invoked_matches_the_project_itself_and_subdirs(self) -> None:
        self.assertTrue(_project_matches("/work/repo", "invoked", "/work/repo"))
        self.assertTrue(_project_matches("/work/repo/src", "invoked", "/work/repo"))

    def test_invoked_rejects_sessions_from_ancestor_directories(self) -> None:
        # A session run from $HOME (or /) belongs to that directory, not to the
        # invoked project below it; harvesting it dilutes the held-out set (#294).
        self.assertFalse(_project_matches(os.path.expanduser("~"), "invoked", os.path.expanduser("~/work/repo")))
        self.assertFalse(_project_matches("/", "invoked", "/work/repo"))

    def test_invoked_from_subdir_still_matches_repo_root_sessions(self) -> None:
        # Repo-root sessions are the same project when invoked in a subdir.
        self.assertTrue(_project_matches("/work/repo", "invoked", "/work/repo/subdir"))

    def test_invoked_rejects_unrelated_projects(self) -> None:
        self.assertFalse(_project_matches("/other/repo", "invoked", "/work/repo"))



if __name__ == "__main__":
    unittest.main()

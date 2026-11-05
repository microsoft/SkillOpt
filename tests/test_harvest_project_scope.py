"""Tests for the ``--scope invoked`` ancestor bound in harvest (issue #294).

Pure-stdlib (unittest), deterministic, no API key, no third-party deps.
Run:  python -m pytest tests/test_harvest_project_scope.py
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from skillopt_sleep.harvest import _ancestor_in_scope, _git_root, _project_matches


class TestProjectMatchesInvokedScope(unittest.TestCase):
    def test_exact_and_descendant_projects_still_match(self):
        self.assertTrue(_project_matches("/repo", "invoked", "/repo"))
        self.assertTrue(_project_matches("/repo/pkg", "invoked", "/repo"))

    def test_sibling_prefix_does_not_match(self):
        # "/repo-tools" is not inside "/repo"
        self.assertFalse(_project_matches("/repo-tools", "invoked", "/repo"))
        self.assertFalse(_project_matches("/elsewhere", "invoked", "/repo"))

    def test_scope_all_ignores_the_ancestor_bound(self):
        self.assertTrue(_project_matches("/", "all", "/repo"))

    def test_explicit_scope_list_is_unaffected(self):
        self.assertTrue(_project_matches("/other", ["/other"], "/repo"))
        self.assertFalse(_project_matches("/other", ["/third"], "/repo"))


class TestAncestorInScope(unittest.TestCase):
    def test_home_and_filesystem_root_are_never_in_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            # tmp is not a git repo, and its ancestors are not either
            repo = os.path.join(tmp, "work", "repo")
            os.makedirs(repo)
            with mock.patch.dict(os.environ, {"HOME": tmp}):
                home = os.path.abspath(os.path.expanduser("~"))
                self.assertFalse(_ancestor_in_scope(home, repo))
            self.assertFalse(_ancestor_in_scope(os.path.abspath(os.sep), repo))

    def test_intermediate_ancestor_without_git_root_stays_in_scope(self):
        # Invoking from "work/repo" must keep sessions started at "work".
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "work", "repo")
            os.makedirs(repo)
            self.assertTrue(_ancestor_in_scope(os.path.join(tmp, "work"), repo))

    def test_git_root_stops_the_ancestor_walk(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "work", "repo")
            os.makedirs(os.path.join(repo, "sub"))
            os.makedirs(os.path.join(repo, ".git"))
            above = os.path.join(tmp, "work")

            # "repo" is the git root: a session started there still matches
            # an invocation from "repo/sub".
            self.assertTrue(_ancestor_in_scope(repo, os.path.join(repo, "sub")))
            # "work" is above the git root, so it no longer matches.
            self.assertFalse(_ancestor_in_scope(above, os.path.join(repo, "sub")))

    def test_git_root_detected_from_a_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            os.makedirs(os.path.join(repo, "a", "b"))
            os.makedirs(os.path.join(repo, ".git"))
            self.assertEqual(_git_root(os.path.join(repo, "a", "b")), repo)
            self.assertEqual(_git_root(tmp), "")

    def test_git_root_accepts_a_git_file_as_in_a_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            os.makedirs(os.path.join(repo, "sub"))
            with open(os.path.join(repo, ".git"), "w", encoding="utf-8") as f:
                f.write("gitdir: /elsewhere/.git/worktrees/repo\n")
            self.assertEqual(_git_root(os.path.join(repo, "sub")), repo)

    def test_home_is_refused_even_when_it_is_a_git_root(self):
        # HOME being a checkout must not reopen the unbounded match.
        with tempfile.TemporaryDirectory() as tmp:
            home = os.path.join(tmp, "home")
            repo = os.path.join(home, "work", "repo")
            os.makedirs(repo)
            os.makedirs(os.path.join(home, ".git"))
            with mock.patch.dict(os.environ, {"HOME": home}):
                self.assertFalse(_ancestor_in_scope(os.path.abspath(home), repo))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

"""Tests for per-skill staging fan-out (issue #120).

Pure-stdlib (unittest), hermetic (tmpdir only), no API key, no network.
Run:  python -m pytest tests/test_sleep_staging_fanout.py
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from skillopt_sleep.staging import (
    SkillProposal,
    StagingError,
    proposal_filename,
    skill_proposal_rows,
    write_skill_proposals,
    write_staging,
)
from skillopt_sleep.types import SleepReport


def _proposal(name="example-skill", body="# example\n", live=None, root="/tmp/live"):
    if live is None:
        live = os.path.join(root, name, "SKILL.md")
    return SkillProposal(name, body, live)


def _report():
    return SleepReport(night=1, project="/repo/example", accepted=True,
                       gate_action="accept_new_best")


class TestSkillProposalRows(unittest.TestCase):
    def test_one_row_per_skill_in_order(self):
        rows = skill_proposal_rows([_proposal("alpha"), _proposal("beta")])
        self.assertEqual([r["skill_name"] for r in rows], ["alpha", "beta"])
        self.assertEqual([r["proposed_file"] for r in rows],
                         ["proposed_SKILL.alpha.md", "proposed_SKILL.beta.md"])
        self.assertEqual(rows[0]["live_skill_path"],
                         os.path.normpath("/tmp/live/alpha/SKILL.md"))

    def test_filenames_are_unique_per_skill(self):
        self.assertNotEqual(proposal_filename("alpha"), proposal_filename("beta"))

    def test_duplicate_skill_name_is_refused(self):
        with self.assertRaises(StagingError):
            skill_proposal_rows([_proposal("alpha"),
                                 _proposal("alpha", live="/tmp/other/SKILL.md")])

    def test_names_differing_only_by_case_are_refused(self):
        # Skill names are case-sensitive, but every proposal lands in one
        # staging directory — and on macOS and Windows that directory is
        # case-insensitive. Before this guard, staging "Research" then
        # "research" produced two manifest rows but one file on disk, named
        # after the first skill and containing the second one's document.
        with self.assertRaises(StagingError):
            skill_proposal_rows([_proposal("Research"),
                                 _proposal("research", live="/tmp/other/SKILL.md")])

    def test_case_differing_proposals_never_lose_a_staged_file(self):
        # Guards the guard: if the refusal above is ever relaxed, this asserts
        # the actual filesystem outcome rather than the intent.
        with tempfile.TemporaryDirectory() as out:
            try:
                rows = write_skill_proposals(
                    out,
                    [_proposal("Research"),
                     _proposal("research", live="/tmp/other/SKILL.md")],
                )
            except StagingError:
                return  # refused up front, which is the desired behaviour
            staged = [f for f in os.listdir(out) if f.startswith("proposed_")]
            self.assertEqual(len(staged), len(rows),
                             "a manifest row exists whose staged file was overwritten")

    def test_live_paths_differing_only_by_case_are_refused(self):
        # os.path.normcase would not catch this: it only folds case on Windows,
        # so it is a no-op on the macOS filesystem where /x/A.md and /x/a.md
        # are nevertheless the same file.
        with self.assertRaises(StagingError):
            skill_proposal_rows([_proposal("alpha", live="/tmp/live/A.md"),
                                 _proposal("beta", live="/tmp/live/a.md")])

    def test_a_generator_of_proposals_still_writes_every_file(self):
        # The annotation says Sequence but nothing enforces it. Validation used
        # to drain a generator, leaving the write loop empty and returning a
        # full set of manifest rows for files that were never created.
        with tempfile.TemporaryDirectory() as out:
            gen = (_proposal(n, live=f"/tmp/live/{n}/SKILL.md") for n in ("alpha", "beta"))
            rows = write_skill_proposals(out, gen)
            staged = [f for f in os.listdir(out) if f.startswith("proposed_")]
            self.assertEqual(len(staged), len(rows))
            self.assertEqual(len(rows), 2)

    def test_names_windows_cannot_store_are_refused_cleanly(self):
        # These reach the filesystem as a filename. Without an explicit guard
        # they raise OSError from inside the write instead of a StagingError
        # naming the offending skill.
        for bad in ["a:b", "a*b", "a?b", 'a"b', "a<b", "a>b", "a|b", "trailing."]:
            with self.assertRaises(StagingError, msg=bad):
                skill_proposal_rows([_proposal(bad)])

    def test_absolute_paths_needing_normalisation_are_accepted(self):
        # Requiring the input to already equal normpath() rejected safe paths:
        # duplicate separators everywhere, and every forward-slash absolute
        # path on Windows. Normalising first keeps the traversal guard.
        rows = skill_proposal_rows([_proposal("alpha", live="/tmp/live//alpha/SKILL.md")])
        self.assertEqual(rows[0]["live_skill_path"], os.path.normpath("/tmp/live/alpha/SKILL.md"))

    def test_current_directory_segments_are_normalised_not_refused(self):
        rows = skill_proposal_rows([
            _proposal("alpha", live="/tmp/live/./alpha/SKILL.md")
        ])
        self.assertEqual(rows[0]["live_skill_path"],
                         os.path.normpath("/tmp/live/alpha/SKILL.md"))

    def test_two_skills_targeting_one_file_are_refused(self):
        shared = "/tmp/live/shared/SKILL.md"
        with self.assertRaises(StagingError):
            skill_proposal_rows([_proposal("alpha", live=shared),
                                 _proposal("beta", live=shared)])

    def test_unsafe_skill_names_are_refused(self):
        for bad in ["", "  ", ".", "..", "../escape", "a/b", "a\\b", "/abs",
                    "~home", "bad\nname"]:
            with self.assertRaises(StagingError, msg=bad):
                skill_proposal_rows([_proposal(bad)])

    def test_unsafe_live_paths_are_refused(self):
        for bad in ["", "relative/SKILL.md", "~/skills/a/SKILL.md",
                    "/tmp/live/../../etc/SKILL.md", "/tmp/live/a/SKILL.txt"]:
            with self.assertRaises(StagingError, msg=bad):
                skill_proposal_rows([_proposal("alpha", live=bad)])


class TestWriteSkillProposals(unittest.TestCase):
    def test_writes_one_file_per_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = write_skill_proposals(tmp, [
                _proposal("alpha", "# alpha\n"),
                _proposal("beta", "# beta\n"),
            ])
            self.assertEqual(sorted(os.listdir(tmp)),
                             ["proposed_SKILL.alpha.md", "proposed_SKILL.beta.md"])
            with open(os.path.join(tmp, rows[0]["proposed_file"]), encoding="utf-8") as f:
                self.assertEqual(f.read(), "# alpha\n")

    def test_no_proposals_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(write_skill_proposals(tmp, []), [])
            self.assertEqual(os.listdir(tmp), [])

    def test_rejected_fan_out_leaves_no_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StagingError):
                write_skill_proposals(tmp, [_proposal("alpha"), _proposal("../escape")])
            self.assertEqual(os.listdir(tmp), [])

    def test_writes_leave_no_temporary_files_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_skill_proposals(tmp, [_proposal("alpha")])
            self.assertEqual([n for n in os.listdir(tmp) if n.startswith(".tmp-")], [])

    def test_rewrite_replaces_content_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_skill_proposals(tmp, [_proposal("alpha", "# first\n")])
            write_skill_proposals(tmp, [_proposal("alpha", "# second\n")])
            path = os.path.join(tmp, proposal_filename("alpha"))
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "# second\n")
            self.assertEqual(sorted(os.listdir(tmp)), [proposal_filename("alpha")])


class TestWriteStagingCompatibility(unittest.TestCase):
    def _manifest(self, out):
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_legacy_layout_when_multi_skill_is_unused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_staging(
                tmp, report=_report(), proposed_skill="# skill\n",
                proposed_memory="# memory\n",
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# report\n",
            )
            self.assertEqual(
                sorted(os.listdir(out)),
                ["manifest.json", "proposed_CLAUDE.md", "proposed_SKILL.md",
                 "report.json", "report.md"],
            )
            manifest = self._manifest(out)
            self.assertNotIn("skills", manifest)
            self.assertTrue(manifest["has_skill"])

    def test_fan_out_adds_files_and_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            live_root = os.path.join(tmp, "live")
            out = write_staging(
                tmp, report=_report(), proposed_skill=None, proposed_memory=None,
                live_skill_path=os.path.join(live_root, "SKILL.md"),
                live_memory_path=os.path.join(live_root, "CLAUDE.md"),
                report_md="# report\n",
                skill_proposals=[
                    _proposal("alpha", "# alpha\n", root=live_root),
                    _proposal("beta", "# beta\n", root=live_root),
                ],
            )
            self.assertEqual(
                sorted(os.listdir(out)),
                ["manifest.json", "proposed_SKILL.alpha.md", "proposed_SKILL.beta.md",
                 "report.json", "report.md"],
            )
            rows = self._manifest(out)["skills"]
            self.assertEqual([r["skill_name"] for r in rows], ["alpha", "beta"])
            self.assertEqual(rows[1]["live_skill_path"],
                             os.path.join(live_root, "beta", "SKILL.md"))

    def test_unsafe_fan_out_writes_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StagingError):
                write_staging(
                    tmp, report=_report(), proposed_skill=None, proposed_memory=None,
                    live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                    live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                    report_md="# report\n",
                    skill_proposals=[_proposal("alpha", live="relative/SKILL.md")],
                )
            for root, _dirs, files in os.walk(tmp):
                self.assertNotIn("manifest.json", files, root)


if __name__ == "__main__":
    unittest.main()

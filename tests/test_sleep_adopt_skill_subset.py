"""Tests for explicit multi-skill subset adoption (issue #120).

Pure-stdlib (unittest), hermetic (tmpdir only), no API key, no network.
Run:  python -m pytest tests/test_sleep_adopt_skill_subset.py
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest

from skillopt_sleep.staging import (
    SkillProposal,
    StagingError,
    adopt_skills,
    staged_skills,
    write_staging,
)
from skillopt_sleep.types import SleepReport


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class TwoSkillNight:
    """End-to-end fixture: a staged night with two per-skill proposals."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.live_root = os.path.join(tmp, "live")
        self.alpha_live = os.path.join(self.live_root, "alpha", "SKILL.md")
        self.beta_live = os.path.join(self.live_root, "beta", "SKILL.md")
        _write(self.alpha_live, "# alpha v1\n")
        _write(self.beta_live, "# beta v1\n")
        self.staging = write_staging(
            tmp,
            report=SleepReport(night=1, project=tmp, accepted=True),
            proposed_skill=None, proposed_memory=None,
            live_skill_path=self.alpha_live,
            live_memory_path=os.path.join(self.live_root, "CLAUDE.md"),
            report_md="# report\n",
            skill_proposals=[
                SkillProposal("alpha", "# alpha v2\n", self.alpha_live),
                SkillProposal("beta", "# beta v2\n", self.beta_live),
            ],
        )


class TestStagedSkills(unittest.TestCase):
    def test_rows_are_readable_from_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            rows = staged_skills(night.staging)
            self.assertEqual([r["skill_name"] for r in rows], ["alpha", "beta"])

    def test_legacy_single_proposal_night_has_no_staged_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_staging(
                tmp, report=SleepReport(night=1, project=tmp), proposed_skill="# s\n",
                proposed_memory=None,
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# report\n",
            )
            self.assertEqual(staged_skills(out), [])
            self.assertEqual(adopt_skills(out), [])


class TestAdoptSkillSubset(unittest.TestCase):
    def test_adopting_one_skill_leaves_the_other_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            receipts = adopt_skills(night.staging, ["alpha"])
            self.assertEqual([r.skill_name for r in receipts], ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_receipts_carry_before_and_after_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            receipt = adopt_skills(night.staging, ["alpha"])[0]
            self.assertEqual(receipt.sha256_before, _sha("# alpha v1\n"))
            self.assertEqual(receipt.sha256_after, _sha("# alpha v2\n"))
            self.assertEqual(receipt.live_skill_path, night.alpha_live)
            self.assertEqual(_read(receipt.backup_path), "# alpha v1\n")

    def test_receipts_are_persisted_beside_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            adopt_skills(night.staging, ["beta"])
            with open(os.path.join(night.staging, "adopted_skills.json"),
                      encoding="utf-8") as f:
                rows = json.load(f)
            self.assertEqual([r["skill_name"] for r in rows], ["beta"])
            self.assertEqual(rows[0]["sha256_after"], _sha("# beta v2\n"))

    def test_selecting_no_skills_adopts_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            self.assertEqual(adopt_skills(night.staging, []), [])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertFalse(
                os.path.exists(os.path.join(night.staging, "adopted_skills.json")))

    def test_selecting_every_skill_adopts_all_of_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            receipts = adopt_skills(night.staging)
            self.assertEqual([r.skill_name for r in receipts], ["alpha", "beta"])
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v2\n")

    def test_a_new_live_file_reports_an_empty_before_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.unlink(night.beta_live)
            receipt = [r for r in adopt_skills(night.staging) if r.skill_name == "beta"][0]
            self.assertEqual(receipt.sha256_before, "")
            self.assertEqual(receipt.backup_path, "")
            self.assertEqual(_read(night.beta_live), "# beta v2\n")

    def test_unknown_or_repeated_selection_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            for selection in (["gamma"], ["alpha", "gamma"], ["alpha", "alpha"]):
                with self.assertRaises(StagingError, msg=str(selection)):
                    adopt_skills(night.staging, selection)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_missing_staged_proposal_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.unlink(os.path.join(night.staging, "proposed_SKILL.beta.md"))
            with self.assertRaises(StagingError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_unsafe_manifest_row_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][1]["live_skill_path"] = "relative/SKILL.md"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_a_failed_write_rolls_the_whole_selection_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            # beta's live path becomes un-writable: its parent is now a file.
            os.unlink(night.beta_live)
            os.rmdir(os.path.dirname(night.beta_live))
            _write(os.path.dirname(night.beta_live), "not a directory\n")
            with self.assertRaises(OSError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertFalse(
                os.path.exists(os.path.join(night.staging, "adopted_skills.json")))

    def test_rollback_removes_files_that_did_not_exist_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.unlink(night.alpha_live)
            os.unlink(night.beta_live)
            os.rmdir(os.path.dirname(night.beta_live))
            _write(os.path.dirname(night.beta_live), "not a directory\n")
            with self.assertRaises(OSError):
                adopt_skills(night.staging)
            self.assertFalse(os.path.exists(night.alpha_live))

    def test_adoption_never_happens_without_an_explicit_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertTrue(os.path.exists(
                os.path.join(night.staging, "proposed_SKILL.alpha.md")))


if __name__ == "__main__":
    unittest.main()

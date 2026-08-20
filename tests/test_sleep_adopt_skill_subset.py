"""Tests for explicit multi-skill subset adoption (issue #120).

Pure-stdlib (unittest), hermetic (tmpdir only), no API key, no network.
Run:  python -m pytest tests/test_sleep_adopt_skill_subset.py
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
import unittest
from unittest import mock

from skillopt_sleep.staging import (
    SkillProposal,
    StagingError,
    StagingRecoveryError,
    adopt,
    adopt_skills,
    has_pending_staged_managed,
    latest_staging,
    pending_staged_skills,
    staged_skills,
    write_staging,
)
from skillopt_sleep.types import EditRecord, SleepReport


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

    def __init__(
        self,
        tmp,
        *,
        alpha_body="# alpha v1\n",
        beta_body="# beta v1\n",
    ):
        self.tmp = tmp
        self.live_root = os.path.join(tmp, "live")
        self.alpha_live = os.path.join(self.live_root, "alpha", "SKILL.md")
        self.beta_live = os.path.join(self.live_root, "beta", "SKILL.md")
        for path, body in (
            (self.alpha_live, alpha_body),
            (self.beta_live, beta_body),
        ):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if body is not None:
                _write(path, body)
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


class TestAdoptionIsConfinedToTheStagedRoots(unittest.TestCase):
    """A manifest is data, not a trust boundary.

    ``_safe_live_path`` only proves a target is absolute, traversal-free and
    ``*.md``; it accepts any such path on the machine. Adoption must therefore
    re-check that each live target still sits under a skills root recorded when
    the night was staged, or a tampered ``live_skill_path`` with self-consistent
    pins redirects the write onto an arbitrary file.
    """

    def _retarget(self, staging, skill_name, new_live):
        manifest_path = os.path.join(staging, "manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for row in manifest["skills"]:
            if row["skill_name"] == skill_name:
                row["live_skill_path"] = new_live
                row["live_realpath"] = new_live
                if row.get("live_sha256"):
                    if os.path.exists(new_live):
                        with open(new_live, "rb") as h:
                            row["live_sha256"] = hashlib.sha256(h.read()).hexdigest()
                    else:
                        row["live_sha256"] = ""
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

    def test_manifest_retargeted_onto_an_outside_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            outside = os.path.join(tmp, "outside", "alpha", "SKILL.md")
            os.makedirs(os.path.dirname(outside), exist_ok=True)
            _write(outside, "# victim\n")
            self._retarget(night.staging, "alpha", outside)
            with self.assertRaises(StagingError) as ctx:
                adopt_skills(night.staging, ["alpha"])
            self.assertIn("outside the skills roots", str(ctx.exception))
            # Fails closed: the victim file is untouched.
            with open(outside, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "# victim\n")

    def test_manifest_retargeted_to_create_a_new_outside_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            outside_dir = os.path.join(tmp, "outside", "alpha")
            os.makedirs(outside_dir, exist_ok=True)
            outside = os.path.join(outside_dir, "SKILL.md")
            self._retarget(night.staging, "alpha", outside)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging, ["alpha"])
            self.assertFalse(os.path.exists(outside))

    def test_a_manifest_without_recorded_roots_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            del manifest["skill_roots"]
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with self.assertRaises(StagingError) as ctx:
                adopt_skills(night.staging, ["alpha"])
            self.assertIn("skill_roots", str(ctx.exception))

    def test_the_ordinary_in_root_adoption_still_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            adopt_skills(night.staging, ["alpha", "beta"])
            with open(night.alpha_live, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "# alpha v2\n")


class TestStagedSkills(unittest.TestCase):
    def test_rows_are_readable_from_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            rows = staged_skills(night.staging)
            self.assertEqual([r["skill_name"] for r in rows], ["alpha", "beta"])

    def test_pending_rows_exclude_validated_incremental_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            self.assertEqual(
                [row["skill_name"] for row in pending_staged_skills(night.staging)],
                ["alpha", "beta"],
            )
            adopt_skills(night.staging, ["alpha"])
            pending = pending_staged_skills(night.staging)
            self.assertEqual([row["skill_name"] for row in pending], ["beta"])
            adopt_skills(
                night.staging, [str(row["skill_name"]) for row in pending]
            )
            self.assertEqual(pending_staged_skills(night.staging), [])

    def test_pending_rows_fail_closed_on_an_invalid_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            adopt_skills(night.staging, ["alpha"])
            receipt = os.path.join(night.staging, "adopted_skills.json")
            with open(receipt, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload[0]["unvalidated"] = True
            with open(receipt, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaisesRegex(StagingError, "invalid schema"):
                pending_staged_skills(night.staging)

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

    def test_malformed_skills_manifest_shape_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            manifest_path = os.path.join(night.staging, "manifest.json")
            for malformed in ({"not": "a list"}, [{"skill_name": "alpha"}, "bad"]):
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                manifest["skills"] = malformed
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f)
                with self.assertRaises(StagingError, msg=repr(malformed)):
                    staged_skills(night.staging)

    def test_adopting_an_older_night_does_not_make_it_latest(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "skillopt_sleep.staging._ts_dir", return_value="20260815-010203"
        ):
            older = TwoSkillNight(tmp)
            newer = TwoSkillNight(tmp)
            self.assertEqual(latest_staging(tmp), newer.staging)
            adopt_skills(older.staging, ["alpha"])
            self.assertEqual(latest_staging(tmp), newer.staging)


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
            night = TwoSkillNight(tmp, beta_body=None)
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

    def test_manifest_proposal_filename_cannot_escape_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            outside = os.path.join(tmp, "outside.md")
            _write(outside, "# not a staged proposal\n")
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][0]["proposed_file"] = os.path.relpath(
                outside, night.staging
            )
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_adoption_preserves_existing_live_file_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.chmod(night.alpha_live, 0o640)
            adopt_skills(night.staging, ["alpha"])
            self.assertEqual(stat.S_IMODE(os.stat(night.alpha_live).st_mode), 0o640)

    def test_a_failed_write_rolls_the_whole_selection_back(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            real_write = staging_mod._write_atomic

            def boom(path, text, *, create_parents=True):
                if path == night.beta_live:
                    raise OSError("disk full")
                return real_write(path, text, create_parents=create_parents)

            with mock.patch.object(staging_mod, "_write_atomic", side_effect=boom):
                with self.assertRaises(OSError):
                    adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertFalse(
                os.path.exists(os.path.join(night.staging, "adopted_skills.json")))

    def test_post_commit_live_write_error_rolls_the_whole_selection_back(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            real_write = staging_mod._write_atomic

            def commit_then_fail(path, text, *, create_parents=True):
                result = real_write(path, text, create_parents=create_parents)
                if path == night.beta_live:
                    raise OSError("late close failure")
                return result

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=commit_then_fail
            ), self.assertRaisesRegex(OSError, "late close failure"):
                adopt_skills(night.staging)

            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertFalse(os.path.exists(os.path.join(
                night.staging, "adopted_skills.json"
            )))
            backup_root = os.path.join(night.staging, "backup")
            for _root, _dirs, files in os.walk(backup_root):
                self.assertEqual(files, [])

    def test_rollback_removes_files_that_did_not_exist_before(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp, alpha_body=None, beta_body=None)
            real_write = staging_mod._write_atomic

            def boom(path, text, *, create_parents=True):
                if path == night.beta_live:
                    raise OSError("disk full")
                return real_write(path, text, create_parents=create_parents)

            with mock.patch.object(staging_mod, "_write_atomic", side_effect=boom):
                with self.assertRaises(OSError):
                    adopt_skills(night.staging)
            self.assertFalse(os.path.exists(night.alpha_live))
            self.assertFalse(os.path.exists(night.beta_live))

    def test_missing_live_parent_is_refused_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.unlink(night.beta_live)
            os.rmdir(os.path.dirname(night.beta_live))
            _write(os.path.dirname(night.beta_live), "not a directory\n")
            with self.assertRaises(StagingError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertFalse(
                os.path.exists(os.path.join(night.staging, "adopted_skills.json")))

    def test_adoption_never_happens_without_an_explicit_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertTrue(os.path.exists(
                os.path.join(night.staging, "proposed_SKILL.alpha.md")))

    def test_tampered_duplicate_live_paths_are_refused_at_adopt(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][1]["live_skill_path"] = night.alpha_live
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertFalse(
                os.path.exists(os.path.join(night.staging, "adopted_skills.json")))

    def test_live_target_that_is_not_a_file_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.unlink(night.beta_live)
            os.mkdir(night.beta_live)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging, ["beta"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertTrue(os.path.isdir(night.beta_live))

    def test_receipt_write_failure_rolls_back_live_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.makedirs(os.path.join(night.staging, "adopted_skills.json"))
            with self.assertRaisesRegex(StagingError, "receipt path"):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_live_file_changed_since_staging_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            _write(night.alpha_live, "# human edit after review\n")
            with self.assertRaisesRegex(StagingError, "changed since staging"):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# human edit after review\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertFalse(os.path.exists(os.path.join(
                night.staging, "adopted_skills.json"
            )))

    def test_live_file_deleted_since_staging_is_never_recreated(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.unlink(night.alpha_live)
            with self.assertRaisesRegex(StagingError, "changed since staging"):
                adopt_skills(night.staging, ["alpha"])
            self.assertFalse(os.path.exists(night.alpha_live))

    def test_absent_live_file_created_since_staging_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp, alpha_body=None)
            _write(night.alpha_live, "# created by user after review\n")
            with self.assertRaisesRegex(StagingError, "changed since staging"):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# created by user after review\n")

    def test_one_stale_target_aborts_the_entire_selection_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            _write(night.beta_live, "# beta changed after review\n")
            with self.assertRaisesRegex(StagingError, "changed since staging"):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta changed after review\n")
            self.assertFalse(os.path.exists(os.path.join(
                night.staging, "adopted_skills.json"
            )))

    def test_incremental_subset_adoption_accumulates_an_immutable_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            adopt_skills(night.staging, ["alpha"])
            alpha_backup = os.path.join(
                night.staging, "backup", "skills", "alpha", "SKILL.md"
            )
            self.assertEqual(_read(alpha_backup), "# alpha v1\n")

            adopt_skills(night.staging, ["beta"])
            receipt_path = os.path.join(night.staging, "adopted_skills.json")
            with open(receipt_path, encoding="utf-8") as handle:
                receipts = json.load(handle)
            self.assertEqual(
                [row["skill_name"] for row in receipts], ["alpha", "beta"]
            )
            self.assertEqual(_read(alpha_backup), "# alpha v1\n")

            receipt_before = _read(receipt_path)
            with self.assertRaisesRegex(
                StagingError, "already adopted|changed since staging"
            ):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(alpha_backup), "# alpha v1\n")
            self.assertEqual(_read(receipt_path), receipt_before)

    def test_repeated_noop_adoption_cannot_rewrite_receipt_or_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            live = os.path.join(tmp, "live", "alpha", "SKILL.md")
            _write(live, "# unchanged\n")
            staging = write_staging(
                tmp,
                report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill=None,
                proposed_memory=None,
                live_skill_path=live,
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# report\n",
                skill_proposals=[SkillProposal("alpha", "# unchanged\n", live)],
            )
            adopt_skills(staging, ["alpha"])
            receipt_path = os.path.join(staging, "adopted_skills.json")
            backup_path = os.path.join(
                staging, "backup", "skills", "alpha", "SKILL.md"
            )
            receipt_before = _read(receipt_path)
            backup_before = _read(backup_path)
            with self.assertRaisesRegex(StagingError, "already adopted"):
                adopt_skills(staging, ["alpha"])
            self.assertEqual(_read(receipt_path), receipt_before)
            self.assertEqual(_read(backup_path), backup_before)

    def test_rollback_restores_original_mode_as_well_as_bytes(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.chmod(night.alpha_live, 0o640)
            real_write = staging_mod._write_atomic

            def boom(path, text, *, create_parents=True):
                if path == night.beta_live:
                    raise OSError("disk full")
                return real_write(path, text, create_parents=create_parents)

            with mock.patch.object(staging_mod, "_write_atomic", side_effect=boom):
                with self.assertRaises(OSError):
                    adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(
                stat.S_IMODE(os.stat(night.alpha_live).st_mode), 0o640
            )

    def test_backup_failure_rolls_back_prior_live_writes(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            real_write_new = staging_mod._write_new_bytes
            beta_backup = os.path.join(
                night.staging, "backup", "skills", "beta", "SKILL.md"
            )

            def boom(path, data, *, mode=None):
                if path == beta_backup:
                    raise OSError("backup device full")
                return real_write_new(path, data, mode=mode)

            with mock.patch.object(
                staging_mod, "_write_new_bytes", side_effect=boom
            ):
                with self.assertRaises(OSError):
                    adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertFalse(os.path.exists(os.path.join(
                night.staging, "adopted_skills.json"
            )))
            backup_root = os.path.join(night.staging, "backup")
            for _root, _dirs, files in os.walk(backup_root):
                self.assertEqual(files, [])


class TestCycleStagesResolvedSkillSubset(unittest.TestCase):
    """run_sleep_cycle stages resolved skills; adopt promotes only the subset."""

    def _hinted_tasks(self):
        from dataclasses import replace

        from skillopt_sleep.experiments.personas import programmer_persona, researcher_persona
        from skillopt_sleep.mine import assign_splits

        research = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)
        programming = assign_splits(programmer_persona(), holdout_fraction=0.34, seed=1)
        tagged = [replace(t, skill_hint="research-skill") for t in research]
        tagged += [replace(t, id=f"prog-{t.id}", skill_hint="programming-skill")
                   for t in programming]
        return tagged

    def test_cycle_stages_both_skills_and_subset_adopt_touches_only_one(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import run_sleep_cycle

        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            managed_live = os.path.join(
                claude_home, "skills", "skillopt-sleep-learned", "SKILL.md")
            research_live = os.path.join(
                claude_home, "skills", "research-skill", "SKILL.md")
            programming_live = os.path.join(
                claude_home, "skills", "programming-skill", "SKILL.md")
            managed_marker = "MANAGED_ONLY_MARKER"
            research_marker = "RESEARCH_ONLY_MARKER"
            programming_marker = "PROGRAMMING_ONLY_MARKER"
            _write(managed_live, f"# managed\n{managed_marker}\n")
            _write(research_live, f"# research-skill v1\n{research_marker}\n")
            _write(
                programming_live,
                f"# programming-skill v1\n{programming_marker}\n",
            )
            cfg = load_config(
                invoked_project=proj, projects="invoked", backend="mock",
                claude_home=claude_home,
                managed_skill_name="skillopt-sleep-learned", auto_adopt=False,
                multi_skill_report=True, gate_mode="off",
            )
            outcome = run_sleep_cycle(cfg, seed_tasks=self._hinted_tasks())
            rows = staged_skills(outcome.staging_dir)
            names = [r["skill_name"] for r in rows]
            self.assertIn("research-skill", names)
            self.assertIn("programming-skill", names)
            self.assertTrue(os.path.isfile(os.path.join(
                outcome.staging_dir, "proposed_SKILL.research-skill.md")))
            self.assertTrue(os.path.isfile(os.path.join(
                outcome.staging_dir, "proposed_SKILL.programming-skill.md")))
            research_proposal = _read(os.path.join(
                outcome.staging_dir, "proposed_SKILL.research-skill.md"))
            programming_proposal = _read(os.path.join(
                outcome.staging_dir, "proposed_SKILL.programming-skill.md"))
            self.assertIn(research_marker, research_proposal)
            self.assertNotIn(programming_marker, research_proposal)
            self.assertNotIn(managed_marker, research_proposal)
            self.assertIn(programming_marker, programming_proposal)
            self.assertNotIn(research_marker, programming_proposal)
            self.assertNotIn(managed_marker, programming_proposal)
            row_by_name = {row["skill_name"]: row for row in rows}
            self.assertEqual(
                row_by_name["research-skill"]["live_sha256"],
                _sha(f"# research-skill v1\n{research_marker}\n"),
            )
            self.assertEqual(
                row_by_name["programming-skill"]["live_sha256"],
                _sha(f"# programming-skill v1\n{programming_marker}\n"),
            )
            self.assertEqual(
                row_by_name["research-skill"]["live_realpath"],
                os.path.realpath(research_live),
            )
            self.assertEqual(
                _read(research_live), f"# research-skill v1\n{research_marker}\n"
            )
            self.assertEqual(
                _read(programming_live),
                f"# programming-skill v1\n{programming_marker}\n",
            )

            receipts = adopt_skills(outcome.staging_dir, ["research-skill"])
            self.assertEqual([r.skill_name for r in receipts], ["research-skill"])
            self.assertEqual(_read(research_live), research_proposal)
            self.assertEqual(
                _read(programming_live),
                f"# programming-skill v1\n{programming_marker}\n",
            )


class TestAdoptSkillCli(unittest.TestCase):
    def _cli(self, argv):
        import contextlib
        import io

        from skillopt_sleep.__main__ import main

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = main(argv)
        return rc, stdout.getvalue()

    def test_human_error_text_removes_terminal_controls(self):
        from skillopt_sleep.__main__ import _display_error

        rendered = _display_error(
            ValueError("bad\x1b[31m red\x1b[0m\nnext\u202e hidden\a")
        )
        self.assertEqual(rendered, "bad red next hidden")
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\u202e", rendered)

    def test_human_run_report_sanitizes_model_edit_text(self):
        import contextlib
        import io
        from types import SimpleNamespace

        from skillopt_sleep.__main__ import _print_run_report

        hostile = "line\nforged\x1b[31m\u202e api_key=SUPERSECRET123456789"
        report = SleepReport(
            night=1,
            project="/tmp/project",
            edits=[EditRecord("skill", "add", hostile)],
            rejected_edits=[EditRecord("skill", "add", hostile)],
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_run_report(
                SimpleNamespace(
                    report=report,
                    staging_dir="",
                    adopted=False,
                    adopted_paths=[],
                ),
                SimpleNamespace(json=False),
                {},
            )
        rendered = output.getvalue()
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\u202e", rendered)
        self.assertNotIn("SUPERSECRET", rendered)
        self.assertNotIn("\nforged", rendered)

    def test_status_lists_staged_skill_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "status", "--project", tmp, "--claude-home", claude_home, "--json",
            ])
            self.assertEqual(rc, 0)
            payload = json.loads(out)
            self.assertEqual(payload["staged_skills"], ["alpha", "beta"])
            self.assertEqual(payload["adopted_skills"], [])
            self.assertFalse(payload["has_managed_proposal"])
            self.assertEqual(payload["latest_staging"], night.staging)

    def test_status_and_all_skills_operate_on_pending_rows_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            self.assertEqual(
                self._cli([
                    "adopt", "--project", tmp, "--claude-home", claude_home,
                    "--skill", "alpha",
                ])[0],
                0,
            )
            rc, out = self._cli([
                "status", "--project", tmp, "--claude-home", claude_home,
                "--json",
            ])
            self.assertEqual(rc, 0)
            payload = json.loads(out)
            self.assertEqual(payload["staged_skills"], ["beta"])
            self.assertEqual(payload["adopted_skills"], ["alpha"])

            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--all-skills", "--json",
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(
                [row["skill_name"] for row in json.loads(out)["adopted_skills"]],
                ["beta"],
            )
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v2\n")

    def test_status_reports_a_corrupt_latest_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            with open(
                os.path.join(night.staging, "manifest.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("{not json")
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "status", "--project", tmp, "--claude-home", claude_home,
                "--json",
            ])
            self.assertEqual(rc, 1)
            payload = json.loads(out)
            self.assertEqual(payload["latest_staging"], night.staging)
            self.assertTrue(payload["staging_error"])
            self.assertEqual(payload["staged_skills"], [])

    def test_human_status_sanitizes_tampered_report_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            hostile = "api_key=SUPERSECRET123456789\x1b[31m\u202eforged"
            _write(os.path.join(night.staging, "report.md"), hostile + "\n# row\n")
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["skills"][0]["live_skill_path"] = hostile
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "status", "--project", tmp, "--claude-home", claude_home,
            ])
            self.assertEqual(rc, 0, out)
            self.assertNotIn("SUPERSECRET", out)
            self.assertNotIn("\x1b", out)
            self.assertNotIn("\u202e", out)
            self.assertIn("[REDACTED]", out)

    def test_bare_adopt_on_a_multi_skill_night_lists_and_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
            ])
            self.assertEqual(rc, 2)
            self.assertIn("--skill", out)
            self.assertIn("alpha", out)
            self.assertIn("beta", out)
            self.assertEqual(_read(os.path.join(tmp, "live", "alpha", "SKILL.md")),
                             "# alpha v1\n")

    def test_run_guidance_keeps_skill_names_out_of_shell_commands(self):
        import contextlib
        import io
        from types import SimpleNamespace

        from skillopt_sleep.__main__ import _print_run_report

        names = ["red team; $(touch PWN)", "--dangerous"]
        outcome = SimpleNamespace(
            report=SleepReport(
                night=1,
                project="/tmp/project",
                n_sessions=0,
                n_tasks=2,
            ),
            staging_dir="/tmp/staged-night",
            adopted=False,
            adopted_paths=[],
        )
        stdout = io.StringIO()
        with mock.patch(
            "skillopt_sleep.__main__.pending_staged_skills",
            return_value=[{"skill_name": name} for name in names],
        ), mock.patch(
            "skillopt_sleep.__main__.has_pending_staged_managed",
            return_value=False,
        ), contextlib.redirect_stdout(stdout):
            _print_run_report(
                outcome,
                SimpleNamespace(json=False),
                {},
            )
        output = stdout.getvalue()
        command_lines = [
            line for line in output.splitlines()
            if "python -m skillopt_sleep adopt" in line
        ]
        self.assertTrue(command_lines)
        self.assertIn("--skill NAME", output)
        self.assertIn("python -m skillopt_sleep adopt --all-skills", output)
        self.assertNotIn("--legacy", output)
        for name in names:
            self.assertIn(repr(name), output)
            self.assertFalse(any(name in line for line in command_lines))

    def test_adopt_skill_flag_promotes_only_the_named_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--skill", "alpha",
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_adopt_json_returns_machine_readable_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--skill", "alpha", "--json",
            ])
            self.assertEqual(rc, 0, out)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["staging_dir"], night.staging)
            self.assertEqual(
                [row["skill_name"] for row in payload["adopted_skills"]],
                ["alpha"],
            )
            self.assertEqual(payload["updated_paths"], [night.alpha_live])

    def test_all_skills_flag_promotes_every_staged_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--all-skills",
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v2\n")

    def test_repeated_skill_flags_adopt_the_named_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--skill", "alpha", "--skill", "beta",
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v2\n")

    def test_skill_and_all_skills_together_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--skill", "alpha", "--all-skills",
            ])
            self.assertEqual(rc, 2)
            self.assertIn(
                "use exactly one of --skill, --all-skills, or --legacy.",
                out,
            )
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_legacy_night_bare_adopt_still_copies_the_managed_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            live = os.path.join(tmp, "live", "SKILL.md")
            memory = os.path.join(tmp, "live", "CLAUDE.md")
            _write(live, "# live v1\n")
            _write(memory, "# mem v1\n")
            write_staging(
                tmp, report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill="# live v2\n", proposed_memory="# mem v2\n",
                live_skill_path=live, live_memory_path=memory,
                report_md="# report\n",
            )
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(_read(live), "# live v2\n")
            self.assertEqual(_read(memory), "# mem v2\n")

    def test_mixed_night_legacy_and_per_skill_adoptions_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            managed = os.path.join(tmp, "live", "managed", "SKILL.md")
            memory = os.path.join(tmp, "live", "CLAUDE.md")
            alpha = os.path.join(tmp, "live", "alpha", "SKILL.md")
            _write(managed, "# managed v1\n")
            _write(memory, "# memory v1\n")
            _write(alpha, "# alpha v1\n")
            staging = write_staging(
                tmp,
                report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill="# managed v2\n",
                proposed_memory="# memory v2\n",
                live_skill_path=managed,
                live_memory_path=memory,
                report_md="# report\n",
                skill_proposals=[SkillProposal("alpha", "# alpha v2\n", alpha)],
            )
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)

            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--staging", staging, "--legacy", "--json",
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(json.loads(out)["mode"], "legacy")
            self.assertFalse(has_pending_staged_managed(staging))
            self.assertEqual(_read(managed), "# managed v2\n")
            self.assertEqual(_read(memory), "# memory v2\n")
            self.assertEqual(_read(alpha), "# alpha v1\n")

            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--staging", staging, "--skill", "alpha", "--json",
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(
                [row["skill_name"] for row in json.loads(out)["adopted_skills"]],
                ["alpha"],
            )
            self.assertEqual(_read(alpha), "# alpha v2\n")
            self.assertEqual(_read(managed), "# managed v2\n")
            self.assertEqual(_read(memory), "# memory v2\n")

    def test_skill_flag_on_a_legacy_night_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            live = os.path.join(tmp, "live", "SKILL.md")
            _write(live, "# live v1\n")
            write_staging(
                tmp, report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill="# live v2\n", proposed_memory=None,
                live_skill_path=live,
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# report\n",
            )
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--skill", "alpha",
            ])
            self.assertEqual(rc, 2)
            self.assertIn("no per-skill", out)
            self.assertEqual(_read(live), "# live v1\n")

    def test_empty_skill_flag_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--skill", "   ",
            ])
            self.assertEqual(rc, 2)
            self.assertIn("non-empty", out)
            self.assertEqual(_read(os.path.join(tmp, "live", "alpha", "SKILL.md")),
                             "# alpha v1\n")

    def test_skill_flag_strips_surrounding_whitespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--skill", "  alpha  ",
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")


class TestAdoptTimeRevalidationMega(unittest.TestCase):
    def test_casefold_live_path_collision_is_refused_at_adopt(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            live_root = os.path.dirname(os.path.dirname(night.alpha_live))
            manifest["skills"][1]["live_skill_path"] = os.path.join(
                live_root, "ALPHA", "SKILL.md")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_symlink_realpath_collision_is_refused_at_adopt(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            alias = os.path.join(night.live_root, "alias", "SKILL.md")
            os.makedirs(os.path.dirname(alias), exist_ok=True)
            try:
                os.symlink(night.alpha_live, alias)
            except OSError:
                self.skipTest("symlinks unavailable")
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][1]["live_skill_path"] = alias
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_receipt_write_failure_restores_a_previous_receipt(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            adopt_skills(night.staging, ["alpha"])
            receipt_path = os.path.join(night.staging, "adopted_skills.json")
            previous = _read(receipt_path)
            real_write = staging_mod._write_atomic

            def boom(path, text, *, create_parents=True):
                if os.path.basename(path) == "adopted_skills.json":
                    raise OSError("disk full")
                return real_write(path, text, create_parents=create_parents)

            with mock.patch.object(staging_mod, "_write_atomic", side_effect=boom):
                with self.assertRaises(OSError):
                    adopt_skills(night.staging, ["beta"])
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(receipt_path), previous)


def _accepted_group(name, body):
    from skillopt_sleep.consolidate import ConsolidationResult
    from skillopt_sleep.multi_skill import CONSOLIDATED, GroupConsolidation

    result = ConsolidationResult(
        accepted=True, gate_action="accept_new_best",
        baseline_score=0.1, candidate_score=0.2,
        new_skill=body, new_memory="",
        applied_edits=[], rejected_edits=[],
        holdout_baseline=0.1, holdout_candidate=0.2,
    )
    return GroupConsolidation(
        skill_name=name, status=CONSOLIDATED, result=result, n_tasks=2,
    )


class TestSkillProposalsFromGroups(unittest.TestCase):
    def test_skips_managed_catch_all_and_unresolved_names(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import _skill_proposals_from_groups

        with tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            live = os.path.join(claude_home, "skills", "research-skill", "SKILL.md")
            _write(live, "# research v1\n")
            cfg = load_config(
                claude_home=claude_home,
                managed_skill_name="skillopt-sleep-learned",
            )
            proposals, notes = _skill_proposals_from_groups(
                cfg,
                {
                    "skillopt-sleep-learned": _accepted_group(
                        "skillopt-sleep-learned", "# managed v2\n"),
                    "research-skill": _accepted_group(
                        "research-skill", "# research v2\n"),
                    "ghost-skill": _accepted_group(
                        "ghost-skill", "# ghost v2\n"),
                },
                "skillopt-sleep-learned",
            )
            names = [p.skill_name for p in proposals]
            self.assertEqual(names, ["research-skill"])
            self.assertEqual(proposals[0].live_skill_path, os.path.realpath(live))
            self.assertEqual(proposals[0].proposed_skill, "# research v2\n")
            self.assertTrue(any("ghost-skill" in note for note in notes))
            self.assertFalse(any("skillopt-sleep-learned" in note for note in notes))

    def test_skips_a_second_skill_that_resolves_to_the_same_live_file(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import _skill_proposals_from_groups

        with tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            skills = os.path.join(claude_home, "skills")
            research = os.path.join(skills, "research-skill")
            alias = os.path.join(skills, "alias-skill")
            _write(os.path.join(research, "SKILL.md"), "# research v1\n")
            try:
                os.symlink(research, alias)
            except OSError:
                self.skipTest("symlinks unavailable")
            cfg = load_config(claude_home=claude_home)
            proposals, notes = _skill_proposals_from_groups(
                cfg,
                {
                    "research-skill": _accepted_group(
                        "research-skill", "# research v2\n"),
                    "alias-skill": _accepted_group(
                        "alias-skill", "# alias v2\n"),
                },
                "skillopt-sleep-learned",
            )
            self.assertEqual([p.skill_name for p in proposals], ["research-skill"])
            self.assertTrue(any("alias-skill" in note for note in notes))


class TestCycleStagingGaps(unittest.TestCase):
    def _hinted_tasks(self):
        from dataclasses import replace

        from skillopt_sleep.experiments.personas import programmer_persona, researcher_persona
        from skillopt_sleep.mine import assign_splits

        research = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)
        programming = assign_splits(programmer_persona(), holdout_fraction=0.34, seed=1)
        tagged = [replace(t, skill_hint="research-skill") for t in research]
        tagged += [replace(t, id=f"prog-{t.id}", skill_hint="programming-skill")
                   for t in programming]
        return tagged

    def test_missing_live_skill_is_skipped_not_aborted(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import run_sleep_cycle

        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            research_live = os.path.join(
                claude_home, "skills", "research-skill", "SKILL.md")
            _write(research_live, "# research-skill v1\n")
            cfg = load_config(
                invoked_project=proj, projects="invoked", backend="mock",
                claude_home=claude_home,
                managed_skill_name="skillopt-sleep-learned", auto_adopt=False,
                multi_skill_report=True,
            )
            outcome = run_sleep_cycle(cfg, seed_tasks=self._hinted_tasks())
            names = [r["skill_name"] for r in staged_skills(outcome.staging_dir)]
            self.assertEqual(names, ["research-skill"])
            self.assertFalse(os.path.isfile(os.path.join(
                outcome.staging_dir, "proposed_SKILL.programming-skill.md")))
            skip_note = next(
                note for note in outcome.report.notes
                if "programming-skill" in note
            )
            self.assertIn(
                skip_note,
                _read(os.path.join(outcome.staging_dir, "report.md")),
            )
            with open(
                os.path.join(outcome.staging_dir, "report.json"),
                encoding="utf-8",
            ) as handle:
                report_json = json.load(handle)
            self.assertIn(skip_note, report_json["notes"])

    def test_skip_note_cannot_inject_markdown_or_terminal_controls(self):
        import unicodedata

        from skillopt_sleep.cycle import _cycle_skip_note, _render_report_md

        note = _cycle_skip_note(
            "research\n## Forged heading\x1b[31mred\x1b[0m\u202e",
            "missing\r\n- forged item\x07\u2066",
        )
        rendered = _render_report_md(
            SleepReport(night=1, project="/tmp/project", notes=[note]),
            {
                "backend": "mock",
                "replay_mode": "live",
                "gate_no_regression": False,
                "gate_mode": "on",
            },
        )
        self.assertNotIn("\n", note)
        self.assertNotIn("\r", note)
        self.assertNotIn("\x1b", note)
        self.assertNotIn("\x07", note)
        self.assertNotIn("\u202e", note)
        self.assertNotIn("\u2066", note)
        self.assertNotIn("[31m", rendered)
        self.assertNotIn("[0m", rendered)
        self.assertNotIn("\r", rendered)
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\x07", rendered)
        self.assertNotIn("\u202e", rendered)
        self.assertNotIn("\u2066", rendered)
        self.assertFalse(any(
            unicodedata.category(ch) in {"Cc", "Cf"}
            for ch in rendered
            if ch != "\n"
        ))
        self.assertEqual(
            sum(
                line.startswith("- cycle skipped skill")
                for line in rendered.splitlines()
            ),
            1,
        )
        self.assertIn("research", note)
        self.assertIn("Forged heading", note)

    def test_report_off_stages_no_per_skill_proposals(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import run_sleep_cycle

        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            _write(os.path.join(claude_home, "skills", "research-skill", "SKILL.md"),
                   "# research-skill v1\n")
            _write(os.path.join(claude_home, "skills", "programming-skill", "SKILL.md"),
                   "# programming-skill v1\n")
            cfg = load_config(
                invoked_project=proj, projects="invoked", backend="mock",
                claude_home=claude_home,
                managed_skill_name="skillopt-sleep-learned", auto_adopt=False,
                multi_skill_report=False,
            )
            outcome = run_sleep_cycle(cfg, seed_tasks=self._hinted_tasks())
            self.assertEqual(staged_skills(outcome.staging_dir), [])

    def test_evolve_skill_false_disables_per_skill_fanout_too(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import run_sleep_cycle

        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            research_live = os.path.join(
                claude_home, "skills", "research-skill", "SKILL.md"
            )
            programming_live = os.path.join(
                claude_home, "skills", "programming-skill", "SKILL.md"
            )
            _write(research_live, "# research-skill v1\n")
            _write(programming_live, "# programming-skill v1\n")
            cfg = load_config(
                invoked_project=proj,
                projects="invoked",
                backend="mock",
                claude_home=claude_home,
                managed_skill_name="skillopt-sleep-learned",
                auto_adopt=False,
                multi_skill_report=True,
                evolve_skill=False,
                gate_mode="off",
            )
            outcome = run_sleep_cycle(cfg, seed_tasks=self._hinted_tasks())
            self.assertEqual(staged_skills(outcome.staging_dir), [])
            self.assertEqual(_read(research_live), "# research-skill v1\n")
            self.assertEqual(_read(programming_live), "# programming-skill v1\n")

    def test_auto_adopt_does_not_promote_per_skill_live_files(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import run_sleep_cycle

        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            research_live = os.path.join(
                claude_home, "skills", "research-skill", "SKILL.md")
            programming_live = os.path.join(
                claude_home, "skills", "programming-skill", "SKILL.md")
            _write(research_live, "# research-skill v1\n")
            _write(programming_live, "# programming-skill v1\n")
            cfg = load_config(
                invoked_project=proj, projects="invoked", backend="mock",
                claude_home=claude_home,
                managed_skill_name="skillopt-sleep-learned", auto_adopt=True,
                multi_skill_report=True,
            )
            outcome = run_sleep_cycle(cfg, seed_tasks=self._hinted_tasks())
            self.assertEqual(_read(research_live), "# research-skill v1\n")
            self.assertEqual(_read(programming_live), "# programming-skill v1\n")
            names = [r["skill_name"] for r in staged_skills(outcome.staging_dir)]
            self.assertIn("research-skill", names)
            self.assertIn("programming-skill", names)


class TestAdoptHardeningPinsAndLayout(unittest.TestCase):
    def test_tampered_proposal_file_is_refused_by_sha256_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            staged = os.path.join(night.staging, "proposed_SKILL.alpha.md")
            _write(staged, "# alpha tampered\n")
            with self.assertRaisesRegex(StagingError, "does not match its manifest sha256"):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertFalse(
                os.path.exists(os.path.join(night.staging, "adopted_skills.json")))

    def test_missing_sha256_pin_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            del manifest["skills"][0]["sha256"]
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaisesRegex(StagingError, "missing a sha256 pin"):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_empty_staged_proposal_is_refused_even_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            staged = os.path.join(night.staging, "proposed_SKILL.alpha.md")
            _write(staged, "   \n")
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][0]["sha256"] = _sha("   \n")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaisesRegex(StagingError, "is empty"):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_symlink_live_file_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.unlink(night.alpha_live)
            elsewhere = os.path.join(tmp, "elsewhere.md")
            _write(elsewhere, "# elsewhere\n")
            try:
                os.symlink(elsewhere, night.alpha_live)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(
                StagingError, r"symlink|(?:not|must be).*SKILL\.md|canonical"
            ):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(elsewhere), "# elsewhere\n")
            self.assertTrue(os.path.islink(night.alpha_live))

    def test_symlink_parent_directory_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            real_parent = os.path.dirname(night.alpha_live)
            alias_parent = os.path.join(night.live_root, "alias-alpha")
            try:
                os.symlink(real_parent, alias_parent)
            except OSError:
                self.skipTest("symlinks unavailable")
            alias_live = os.path.join(alias_parent, "SKILL.md")
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][0]["skill_name"] = "alias-alpha"
            manifest["skills"][0]["proposed_file"] = "proposed_SKILL.alias-alpha.md"
            manifest["skills"][0]["live_skill_path"] = alias_live
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            os.rename(
                os.path.join(night.staging, "proposed_SKILL.alpha.md"),
                os.path.join(night.staging, "proposed_SKILL.alias-alpha.md"),
            )
            with self.assertRaisesRegex(
                StagingError, r"symlink|(?:not|must be).*SKILL\.md|canonical"
            ):
                adopt_skills(night.staging, ["alias-alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_ancestor_symlink_swap_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            apparent = os.path.join(tmp, "apparent")
            night = TwoSkillNight(apparent)
            real_tree = os.path.join(tmp, "moved-after-staging")
            os.rename(night.live_root, real_tree)
            try:
                os.symlink(real_tree, night.live_root)
            except OSError:
                self.skipTest("symlinks unavailable")

            outside_alpha = os.path.join(real_tree, "alpha", "SKILL.md")
            with self.assertRaisesRegex(
                StagingError, "symlink|canonical target|changed since staging"
            ):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(outside_alpha), "# alpha v1\n")

    def test_hardlinked_live_targets_are_refused_as_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            live_root = os.path.join(tmp, "live")
            alpha = os.path.join(live_root, "alpha", "SKILL.md")
            beta = os.path.join(live_root, "beta", "SKILL.md")
            _write(alpha, "# shared baseline\n")
            os.makedirs(os.path.dirname(beta), exist_ok=True)
            try:
                os.link(alpha, beta)
            except OSError:
                self.skipTest("hard links unavailable")
            staging = write_staging(
                tmp,
                report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill=None,
                proposed_memory=None,
                live_skill_path=alpha,
                live_memory_path=os.path.join(live_root, "CLAUDE.md"),
                report_md="# report\n",
                skill_proposals=[
                    SkillProposal("alpha", "# alpha v2\n", alpha),
                    SkillProposal("beta", "# beta v2\n", beta),
                ],
            )
            with self.assertRaisesRegex(StagingError, "same file|hard link"):
                adopt_skills(staging, ["alpha"])
            self.assertEqual(_read(alpha), "# shared baseline\n")
            self.assertEqual(_read(beta), "# shared baseline\n")

    def test_symlink_staged_proposal_is_refused_even_when_bytes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            staged = os.path.join(night.staging, "proposed_SKILL.alpha.md")
            outside = os.path.join(tmp, "outside-proposal.md")
            _write(outside, "# alpha v2\n")
            os.unlink(staged)
            try:
                os.symlink(outside, staged)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(StagingError, "symlink"):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_invalid_utf8_staged_proposal_is_a_safe_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            staged = os.path.join(night.staging, "proposed_SKILL.alpha.md")
            with open(staged, "wb") as handle:
                handle.write(b"\xff\xfe")
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["skills"][0]["sha256"] = hashlib.sha256(b"\xff\xfe").hexdigest()
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with self.assertRaisesRegex(StagingError, "UTF-8"):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_concurrent_adoption_cleanly_refuses_one_writer(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            entered = threading.Event()
            release = threading.Event()
            first_errors = []
            real_write = staging_mod._write_atomic

            def pause_first_live_write(path, text, *, create_parents=True):
                if path == night.alpha_live and not entered.is_set():
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("test timed out waiting for release")
                return real_write(path, text, create_parents=create_parents)

            def first_adoption():
                try:
                    adopt_skills(night.staging, ["alpha"])
                except BaseException as exc:  # surfaced in the parent thread
                    first_errors.append(exc)

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=pause_first_live_write
            ):
                worker = threading.Thread(target=first_adoption)
                worker.start()
                self.assertTrue(entered.wait(5), "first adoption never reached write")
                try:
                    with self.assertRaisesRegex(StagingError, "in progress|locked"):
                        adopt_skills(night.staging, ["alpha"])
                finally:
                    release.set()
                    worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(first_errors, [])
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")

    def test_separate_nights_share_the_same_live_target_lock(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            live = os.path.join(tmp, "live", "alpha", "SKILL.md")
            _write(live, "# alpha v1\n")

            def stage(proposal):
                return write_staging(
                    tmp,
                    report=SleepReport(night=1, project=tmp, accepted=True),
                    proposed_skill=None,
                    proposed_memory=None,
                    live_skill_path=live,
                    live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                    report_md="# report\n",
                    skill_proposals=[SkillProposal("alpha", proposal, live)],
                )

            first = stage("# alpha from first night\n")
            second = stage("# alpha from second night\n")
            entered = threading.Event()
            release = threading.Event()
            first_errors = []
            real_write = staging_mod._write_atomic

            def pause_first_live_write(path, text, *, create_parents=True):
                if path == live and not entered.is_set():
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("test timed out waiting for release")
                return real_write(path, text, create_parents=create_parents)

            def first_adoption():
                try:
                    adopt_skills(first, ["alpha"])
                except BaseException as exc:
                    first_errors.append(exc)

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=pause_first_live_write
            ):
                worker = threading.Thread(target=first_adoption)
                worker.start()
                self.assertTrue(entered.wait(5), "first adoption never reached write")
                try:
                    with self.assertRaisesRegex(StagingError, "in progress|stale lock"):
                        adopt_skills(second, ["alpha"])
                finally:
                    release.set()
                    worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(first_errors, [])
            self.assertEqual(_read(live), "# alpha from first night\n")

    def test_partial_lock_acquisition_cleans_earlier_locks(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first.lock")
            occupied = os.path.join(tmp, "occupied.lock")
            _write(occupied, "held\n")
            with self.assertRaisesRegex(StagingError, "in progress|stale lock"):
                with staging_mod._exclusive_create_locks([first, occupied]):
                    self.fail("lock acquisition should have refused the occupied lock")
            self.assertFalse(os.path.lexists(first))
            self.assertEqual(_read(occupied), "held\n")

    def test_subset_adopt_refuses_unselected_sibling_realpath_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            alias = os.path.join(night.live_root, "alias", "SKILL.md")
            os.makedirs(os.path.dirname(alias), exist_ok=True)
            try:
                os.symlink(night.alpha_live, alias)
            except OSError:
                self.skipTest("symlinks unavailable")
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][1]["live_skill_path"] = alias
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_live_path_not_named_skill_md_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            wrong = os.path.join(night.live_root, "alpha", "NOTES.md")
            _write(wrong, "# notes\n")
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][0]["live_skill_path"] = wrong
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaisesRegex(StagingError, "must be a SKILL.md file"):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_cycle_skips_empty_proposed_skill_with_a_note(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import _skill_proposals_from_groups

        with tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            live = os.path.join(claude_home, "skills", "research-skill", "SKILL.md")
            _write(live, "# research v1\n")
            cfg = load_config(claude_home=claude_home)
            proposals, notes = _skill_proposals_from_groups(
                cfg,
                {"research-skill": _accepted_group("research-skill", "   \n")},
                "skillopt-sleep-learned",
            )
            self.assertEqual(proposals, [])
            self.assertTrue(any("empty proposed_skill" in note for note in notes))


class TestDurableAdoptionTransaction(unittest.TestCase):
    def _legacy_night(self, tmp):
        skill = os.path.join(tmp, "live", "skill", "SKILL.md")
        memory = os.path.join(tmp, "live", "CLAUDE.md")
        _write(skill, "# skill v1\n")
        _write(memory, "# memory v1\n")
        staging = write_staging(
            tmp,
            report=SleepReport(night=1, project=tmp, accepted=True),
            proposed_skill="# skill v2\n",
            proposed_memory="# memory v2\n",
            live_skill_path=skill,
            live_memory_path=memory,
            report_md="# report\n",
        )
        return staging, skill, memory

    def test_wal_is_durable_before_first_backup_and_removed_at_commit(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            wal_path = os.path.join(night.staging, ".adopt-transaction.json")
            observed = []
            real_write_new = staging_mod._write_new_bytes

            def observe_backup(path, data, *, mode=None):
                if path == wal_path:
                    return real_write_new(path, data, mode=mode)
                with open(wal_path, encoding="utf-8") as handle:
                    wal = json.load(handle)
                observed.append((path, wal["kind"], len(wal["targets"])))
                return real_write_new(path, data, mode=mode)

            with mock.patch.object(
                staging_mod, "_write_new_bytes", side_effect=observe_backup
            ):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0][1:], ("skills", 1))
            self.assertFalse(os.path.lexists(wal_path))

    def test_interrupted_transaction_is_recovered_before_retry(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            wal_path = os.path.join(night.staging, ".adopt-transaction.json")
            real_write = staging_mod._write_atomic

            def commit_then_fail(path, text, *, create_parents=True):
                result = real_write(path, text, create_parents=create_parents)
                if path == night.alpha_live:
                    raise OSError("simulated process interruption")
                return result

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=commit_then_fail
            ), mock.patch.object(
                staging_mod,
                "_recover_transaction_locked",
                return_value=["simulated process terminated before rollback"],
            ):
                with self.assertRaises(StagingRecoveryError):
                    adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertTrue(os.path.isfile(wal_path))

            receipts = adopt_skills(night.staging, ["alpha"])
            self.assertEqual([receipt.skill_name for receipt in receipts], ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertFalse(os.path.lexists(wal_path))
            with open(
                os.path.join(night.staging, "adopted_skills.json"),
                encoding="utf-8",
            ) as handle:
                self.assertEqual(len(json.load(handle)), 1)

    def test_interrupted_transaction_recovers_before_corrupt_manifest_read(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            wal_path = os.path.join(night.staging, ".adopt-transaction.json")
            real_write = staging_mod._write_atomic

            def commit_then_fail(path, text, *, create_parents=True):
                result = real_write(path, text, create_parents=create_parents)
                if path == night.alpha_live:
                    raise OSError("simulated interruption")
                return result

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=commit_then_fail
            ), mock.patch.object(
                staging_mod,
                "_recover_transaction_locked",
                return_value=["process stopped before rollback"],
            ):
                with self.assertRaises(StagingRecoveryError):
                    adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertTrue(os.path.isfile(wal_path))

            _write(os.path.join(night.staging, "manifest.json"), "{broken")
            with self.assertRaisesRegex(StagingError, "manifest"):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertFalse(os.path.lexists(wal_path))
            self.assertFalse(os.path.lexists(os.path.join(
                night.staging, "backup", "skills", "alpha", "SKILL.md"
            )))

    def test_interrupted_relative_staging_recovers_via_absolute_path(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            live = os.path.join(tmp, "live", "alpha", "SKILL.md")
            _write(live, "# alpha v1\n")
            previous_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                relative_staging = write_staging(
                    ".",
                    report=SleepReport(night=1, project=tmp, accepted=True),
                    proposed_skill=None,
                    proposed_memory=None,
                    live_skill_path=live,
                    live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                    report_md="# report\n",
                    skill_proposals=[
                        SkillProposal("alpha", "# alpha v2\n", live),
                    ],
                )
                absolute_staging = os.path.abspath(relative_staging)
                real_write = staging_mod._write_atomic

                def commit_then_fail(path, text, *, create_parents=True):
                    result = real_write(path, text, create_parents=create_parents)
                    if path == live:
                        raise OSError("simulated interruption")
                    return result

                with mock.patch.object(
                    staging_mod, "_write_atomic", side_effect=commit_then_fail
                ), mock.patch.object(
                    staging_mod,
                    "_recover_transaction_locked",
                    return_value=["process stopped before rollback"],
                ):
                    with self.assertRaises(StagingRecoveryError):
                        adopt_skills(relative_staging, ["alpha"])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(_read(live), "# alpha v2\n")
            adopt_skills(absolute_staging, ["alpha"])
            self.assertEqual(_read(live), "# alpha v2\n")
            self.assertFalse(os.path.lexists(os.path.join(
                absolute_staging, ".adopt-transaction.json"
            )))

    def test_restart_cleans_own_hardlink_publication_temp(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            real_write = staging_mod._write_atomic

            def commit_then_fail(path, text, *, create_parents=True):
                result = real_write(path, text, create_parents=create_parents)
                if path == night.alpha_live:
                    raise OSError("simulated interruption")
                return result

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=commit_then_fail
            ), mock.patch.object(
                staging_mod,
                "_recover_transaction_locked",
                return_value=["process stopped before rollback"],
            ):
                with self.assertRaises(StagingRecoveryError):
                    adopt_skills(night.staging, ["alpha"])

            backup = os.path.join(
                night.staging, "backup", "skills", "alpha", "SKILL.md"
            )
            alias = os.path.join(os.path.dirname(backup), ".tmp-new-crash.md")
            try:
                os.link(backup, alias)
            except OSError:
                self.skipTest("hard links unavailable")
            adopt_skills(night.staging, ["alpha"])
            self.assertFalse(os.path.lexists(alias))
            self.assertFalse(os.path.lexists(os.path.join(
                night.staging, ".adopt-transaction.json"
            )))

    def test_rollback_preserves_concurrent_human_edit_and_retains_wal(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            real_write = staging_mod._write_atomic

            def fail_beta_after_human_edit(path, text, *, create_parents=True):
                if path == night.beta_live:
                    _write(night.alpha_live, "# concurrent human edit\n")
                    raise OSError("beta disk failure")
                return real_write(path, text, create_parents=create_parents)

            with mock.patch.object(
                staging_mod,
                "_write_atomic",
                side_effect=fail_beta_after_human_edit,
            ):
                with self.assertRaises(StagingRecoveryError) as caught:
                    adopt_skills(night.staging)
            self.assertIsInstance(caught.exception.primary, OSError)
            self.assertEqual(_read(night.alpha_live), "# concurrent human edit\n")
            self.assertTrue(os.path.isfile(os.path.join(
                night.staging, ".adopt-transaction.json"
            )))
            self.assertTrue(os.path.isfile(os.path.join(
                night.staging, "backup", "skills", "alpha", "SKILL.md"
            )))

    def test_edit_during_receipt_publication_never_commits_a_false_receipt(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            receipt_path = os.path.join(night.staging, "adopted_skills.json")
            real_write = staging_mod._write_atomic

            def edit_live_before_receipt(path, text, *, create_parents=True):
                if path == receipt_path:
                    _write(night.alpha_live, "# concurrent human edit\n")
                return real_write(path, text, create_parents=create_parents)

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=edit_live_before_receipt
            ):
                with self.assertRaises(StagingRecoveryError) as caught:
                    adopt_skills(night.staging, ["alpha"])
            self.assertIn("changed after publication", str(caught.exception.primary))
            self.assertEqual(_read(night.alpha_live), "# concurrent human edit\n")
            self.assertFalse(os.path.lexists(receipt_path))
            self.assertTrue(os.path.isfile(os.path.join(
                night.staging, ".adopt-transaction.json"
            )))

    def test_prior_backup_must_still_match_immutable_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            receipt = adopt_skills(night.staging, ["alpha"])[0]
            _write(receipt.backup_path, "# tampered backup\n")
            with self.assertRaisesRegex(StagingError, "immutable backup.*changed"):
                adopt_skills(night.staging, ["beta"])
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_prior_backup_cannot_be_reached_through_symlinked_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            receipt = adopt_skills(night.staging, ["alpha"])[0]
            backup_parent = os.path.dirname(receipt.backup_path)
            outside_parent = os.path.join(tmp, "outside-backup")
            os.rename(backup_parent, outside_parent)
            try:
                os.symlink(outside_parent, backup_parent)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(StagingError, "immutable backup.*missing"):
                adopt_skills(night.staging, ["beta"])
            self.assertEqual(_read(os.path.join(outside_parent, "SKILL.md")), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_existing_receipt_requires_the_exact_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            adopt_skills(night.staging, ["alpha"])
            receipt_path = os.path.join(night.staging, "adopted_skills.json")
            with open(receipt_path, encoding="utf-8") as handle:
                receipt = json.load(handle)
            receipt[0]["unexpected"] = True
            with open(receipt_path, "w", encoding="utf-8") as handle:
                json.dump(receipt, handle)
            with self.assertRaisesRegex(StagingError, "invalid schema"):
                adopt_skills(night.staging, ["beta"])
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_case_only_live_retarget_is_not_the_pinned_posix_identity(self):
        if os.path.normcase("a") == os.path.normcase("A"):
            self.skipTest("case-insensitive platform path identity")
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            alternate = os.path.join(tmp, "live", "Alpha", "SKILL.md")
            _write(alternate, "# alpha v1\n")
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["skills"][0]["live_skill_path"] = alternate
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with self.assertRaisesRegex(StagingError, "canonical target"):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(alternate), "# alpha v1\n")
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_unicode_equivalent_skill_directory_adopts(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = "caf\u00e9"
            on_disk_name = "cafe\u0301"
            live = os.path.join(tmp, "live", on_disk_name, "SKILL.md")
            _write(live, "# cafe v1\n")
            staging = write_staging(
                tmp,
                report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill=None,
                proposed_memory=None,
                live_skill_path=live,
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# report\n",
                skill_proposals=[SkillProposal(name, "# cafe v2\n", live)],
            )
            receipts = adopt_skills(staging, [name])
            self.assertEqual([row.skill_name for row in receipts], [name])
            self.assertEqual(_read(live), "# cafe v2\n")

    def test_legacy_manifest_is_pinned_and_adoption_has_a_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, skill, memory = self._legacy_night(tmp)
            with open(os.path.join(staging, "manifest.json"), encoding="utf-8") as handle:
                legacy = json.load(handle)["legacy"]
            self.assertEqual(legacy["skill"]["live_sha256"], _sha("# skill v1\n"))
            self.assertEqual(legacy["memory"]["live_sha256"], _sha("# memory v1\n"))
            self.assertEqual(adopt(staging), [skill, memory])
            self.assertEqual(_read(skill), "# skill v2\n")
            self.assertEqual(_read(memory), "# memory v2\n")
            with open(
                os.path.join(staging, "adopted_legacy.json"), encoding="utf-8"
            ) as handle:
                self.assertEqual(
                    [row["target"] for row in json.load(handle)],
                    ["skill", "memory"],
                )

    def test_legacy_missing_targets_can_share_one_new_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            live_root = os.path.join(tmp, "new-live")
            skill = os.path.join(live_root, "SKILL.md")
            memory = os.path.join(live_root, "CLAUDE.md")
            staging = write_staging(
                tmp,
                report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill="# skill v2\n",
                proposed_memory="# memory v2\n",
                live_skill_path=skill,
                live_memory_path=memory,
                report_md="# report\n",
            )
            self.assertEqual(adopt(staging), [skill, memory])
            self.assertEqual(_read(skill), "# skill v2\n")
            self.assertEqual(_read(memory), "# memory v2\n")

    def test_failed_legacy_adoption_removes_its_exact_new_directory_tree(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            live_root = os.path.join(tmp, "new", "nested", "live")
            skill = os.path.join(live_root, "SKILL.md")
            memory = os.path.join(live_root, "CLAUDE.md")
            staging = write_staging(
                tmp,
                report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill="# skill v2\n",
                proposed_memory="# memory v2\n",
                live_skill_path=skill,
                live_memory_path=memory,
                report_md="# report\n",
            )
            receipt = os.path.join(staging, "adopted_legacy.json")
            real_write = staging_mod._write_atomic

            def fail_receipt(path, text, *, create_parents=True):
                if path == receipt:
                    raise OSError("receipt device full")
                return real_write(path, text, create_parents=create_parents)

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=fail_receipt
            ), self.assertRaisesRegex(OSError, "receipt device full"):
                adopt(staging)
            self.assertFalse(os.path.lexists(os.path.join(tmp, "new")))
            self.assertFalse(os.path.lexists(os.path.join(
                staging, ".adopt-transaction.json"
            )))

    def test_recovery_never_removes_a_replaced_created_directory(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            live_root = os.path.join(tmp, "new-live")
            skill = os.path.join(live_root, "SKILL.md")
            memory = os.path.join(live_root, "CLAUDE.md")
            staging = write_staging(
                tmp,
                report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill="# skill v2\n",
                proposed_memory="# memory v2\n",
                live_skill_path=skill,
                live_memory_path=memory,
                report_md="# report\n",
            )
            receipt = os.path.join(staging, "adopted_legacy.json")
            moved_original = os.path.join(tmp, "transaction-owned-directory")
            real_write = staging_mod._write_atomic

            def fail_receipt(path, text, *, create_parents=True):
                if path == receipt:
                    raise OSError("receipt device full")
                return real_write(path, text, create_parents=create_parents)

            def replace_before_directory_cleanup(targets, staging_dir):
                os.rename(live_root, moved_original)
                os.mkdir(live_root)
                return []

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=fail_receipt
            ), mock.patch.object(
                staging_mod,
                "_cleanup_transaction_backups",
                side_effect=replace_before_directory_cleanup,
            ):
                with self.assertRaises(StagingRecoveryError):
                    adopt(staging)
            self.assertTrue(os.path.isdir(live_root))
            self.assertTrue(os.path.isdir(moved_original))
            self.assertTrue(os.path.isfile(os.path.join(
                staging, ".adopt-transaction.json"
            )))

    def test_restart_recovery_removes_journaled_created_directories(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            live_root = os.path.join(tmp, "restart", "live")
            skill = os.path.join(live_root, "SKILL.md")
            memory = os.path.join(live_root, "CLAUDE.md")
            staging = write_staging(
                tmp,
                report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill="# skill v2\n",
                proposed_memory="# memory v2\n",
                live_skill_path=skill,
                live_memory_path=memory,
                report_md="# report\n",
            )
            receipt = os.path.join(staging, "adopted_legacy.json")
            real_write = staging_mod._write_atomic

            def fail_receipt(path, text, *, create_parents=True):
                if path == receipt:
                    raise OSError("simulated interruption")
                return real_write(path, text, create_parents=create_parents)

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=fail_receipt
            ), mock.patch.object(
                staging_mod,
                "_recover_transaction_locked",
                return_value=["process stopped before rollback"],
            ), self.assertRaises(StagingRecoveryError):
                adopt(staging)
            self.assertTrue(os.path.isdir(live_root))

            _write(os.path.join(staging, "manifest.json"), "{broken")
            with self.assertRaisesRegex(StagingError, "manifest"):
                adopt(staging)
            self.assertFalse(os.path.lexists(os.path.join(tmp, "restart")))
            self.assertFalse(os.path.lexists(os.path.join(
                staging, ".adopt-transaction.json"
            )))

    def test_legacy_unpinned_manifest_refuses_and_requires_restage(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, skill, _memory = self._legacy_night(tmp)
            manifest_path = os.path.join(staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            del manifest["legacy"]
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with self.assertRaisesRegex(StagingError, "discard and restage"):
                adopt(staging)
            self.assertEqual(_read(skill), "# skill v1\n")

    def test_legacy_symlink_swap_never_writes_through_to_outside(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, skill, _memory = self._legacy_night(tmp)
            outside = os.path.join(tmp, "outside.md")
            _write(outside, "# outside\n")
            os.unlink(skill)
            try:
                os.symlink(outside, skill)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(StagingError, "symlink"):
                adopt(staging)
            self.assertEqual(_read(outside), "# outside\n")

    def test_legacy_second_target_failure_rolls_back_first(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            staging, skill, memory = self._legacy_night(tmp)
            real_write = staging_mod._write_atomic

            def fail_memory(path, text, *, create_parents=True):
                if path == memory:
                    raise OSError("memory disk failure")
                return real_write(path, text, create_parents=create_parents)

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=fail_memory
            ):
                with self.assertRaisesRegex(OSError, "memory disk failure"):
                    adopt(staging)
            self.assertEqual(_read(skill), "# skill v1\n")
            self.assertEqual(_read(memory), "# memory v1\n")
            self.assertFalse(os.path.lexists(os.path.join(
                staging, ".adopt-transaction.json"
            )))

    def test_insecure_existing_lock_root_is_rejected(self):
        from skillopt_sleep import staging as staging_mod

        if not hasattr(os, "getuid"):
            self.skipTest("POSIX ownership and mode check")
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, f"skillopt-sleep-adopt-{os.getuid()}")
            os.mkdir(root)
            os.chmod(root, 0o777)
            with mock.patch.object(
                staging_mod.tempfile, "gettempdir", return_value=tmp
            ):
                with self.assertRaisesRegex(StagingError, "permissions are unsafe"):
                    staging_mod._target_lock_paths([os.path.join(tmp, "SKILL.md")])


if __name__ == "__main__":
    unittest.main()

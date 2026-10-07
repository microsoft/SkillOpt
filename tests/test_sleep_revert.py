"""Tests for undoing an adoption: `skillopt-sleep revert`."""
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

from skillopt_sleep.__main__ import main
from skillopt_sleep.staging import (
    SkillProposal,
    StagingError,
    _recover_revert_transaction_locked,
    _revert_transaction_wal,
    _revert_wal_path,
    _RevertTarget,
    _write_revert_wal,
    adopt,
    adopt_skills,
    adopted_skill_names,
    has_adopted_legacy,
    latest_adopted_staging,
    revert,
    revert_skills,
    write_staging,
)
from skillopt_sleep.types import SleepReport

ORIGINAL = "# original\n"
PROPOSED = "# proposed\n"
MEMORY_ORIGINAL = "# memory original\n"
MEMORY_PROPOSED = "# memory proposed\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TestSleepRevert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.project = os.path.join(self.tmp, "project")
        os.makedirs(self.project, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_revert_skills(self):
        skill_live = os.path.join(self.project, "skills", "hello", "SKILL.md")
        _write(skill_live, ORIGINAL)

        proposal = SkillProposal("hello", PROPOSED, skill_live)
        report = SleepReport(night=1, project=self.project)
        staging = write_staging(
            self.project,
            report=report,
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[proposal],
            report_md="",
        )

        receipts = adopt_skills(staging)
        self.assertEqual(len(receipts), 1)
        self.assertEqual(_read(skill_live), PROPOSED)

        reverted = revert_skills(staging)
        self.assertEqual(len(reverted), 1)
        self.assertEqual(_read(skill_live), ORIGINAL)

    def test_revert_legacy(self):
        skill_live = os.path.join(self.project, "SKILL.md")
        _write(skill_live, ORIGINAL)

        report = SleepReport(night=1, project=self.project, accepted=True)
        staging = write_staging(
            self.project,
            report=report,
            proposed_skill=PROPOSED,
            proposed_memory=None,
            live_skill_path=skill_live,
            live_memory_path=None,
            skill_proposals=[],
            report_md="",
        )

        updated = adopt(staging)
        self.assertEqual(len(updated), 1)
        self.assertEqual(_read(skill_live), PROPOSED)

        reverted = revert(staging)
        self.assertEqual(len(reverted), 1)
        self.assertEqual(_read(skill_live), ORIGINAL)

    def test_removes_a_file_adoption_created(self):
        skill_live = os.path.join(self.project, "skills", "brand_new", "SKILL.md")
        os.makedirs(os.path.dirname(skill_live), exist_ok=True)
        self.assertFalse(os.path.exists(skill_live))

        proposal = SkillProposal("brand_new", PROPOSED, skill_live)
        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[proposal],
            report_md="",
        )

        adopt_skills(staging)
        self.assertTrue(os.path.exists(skill_live))
        self.assertEqual(_read(skill_live), PROPOSED)

        reverted = revert_skills(staging)
        self.assertEqual(len(reverted), 1)
        self.assertFalse(os.path.exists(skill_live))

    def test_restores_skill_and_memory_together(self):
        skill_live = os.path.join(self.project, "SKILL.md")
        memory_live = os.path.join(self.project, "CLAUDE.md")
        _write(skill_live, ORIGINAL)
        _write(memory_live, MEMORY_ORIGINAL)

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project, accepted=True),
            proposed_skill=PROPOSED,
            proposed_memory=MEMORY_PROPOSED,
            live_skill_path=skill_live,
            live_memory_path=memory_live,
            skill_proposals=[],
            report_md="",
        )

        adopt(staging)
        self.assertEqual(_read(skill_live), PROPOSED)
        self.assertEqual(_read(memory_live), MEMORY_PROPOSED)

        revert(staging)
        self.assertEqual(_read(skill_live), ORIGINAL)
        self.assertEqual(_read(memory_live), MEMORY_ORIGINAL)

    def test_adopt_revert_adopt_roundtrip(self):
        """Regression test for Yif-Yang review: adopt -> revert -> adopt must succeed."""
        skill_live = os.path.join(self.project, "skills", "cycle", "SKILL.md")
        _write(skill_live, ORIGINAL)

        proposal = SkillProposal("cycle", PROPOSED, skill_live)
        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[proposal],
            report_md="",
        )

        adopt_skills(staging)
        self.assertEqual(_read(skill_live), PROPOSED)

        revert_skills(staging)
        self.assertEqual(_read(skill_live), ORIGINAL)

        # Adopting the same staging again must succeed without "immutable backup already exists" error
        adopt_skills(staging)
        self.assertEqual(_read(skill_live), PROPOSED)

        revert_skills(staging)
        self.assertEqual(_read(skill_live), ORIGINAL)

    def test_revert_converges_when_run_twice(self):
        skill_live = os.path.join(self.project, "skills", "idempotent", "SKILL.md")
        _write(skill_live, ORIGINAL)

        proposal = SkillProposal("idempotent", PROPOSED, skill_live)
        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[proposal],
            report_md="",
        )

        adopt_skills(staging)
        reverted1 = revert_skills(staging)
        self.assertEqual(len(reverted1), 1)

        # Second revert converges cleanly and reports nothing left to revert
        reverted2 = revert_skills(staging)
        self.assertEqual(len(reverted2), 0)
        self.assertEqual(_read(skill_live), ORIGINAL)

    def test_never_adopted_is_a_no_op(self):
        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[],
            report_md="",
        )
        self.assertEqual(revert_skills(staging), [])
        self.assertEqual(revert(staging), [])

    def test_refuses_when_live_file_edited_after_adoption(self):
        skill_live = os.path.join(self.project, "skills", "edited", "SKILL.md")
        _write(skill_live, ORIGINAL)

        proposal = SkillProposal("edited", PROPOSED, skill_live)
        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[proposal],
            report_md="",
        )

        adopt_skills(staging)
        _write(skill_live, "# locally modified by developer\n")

        with self.assertRaises(StagingError) as ctx:
            revert_skills(staging)
        self.assertIn("changed since adoption", str(ctx.exception))
        self.assertEqual(_read(skill_live), "# locally modified by developer\n")

    def test_refuses_an_unknown_skill_name(self):
        skill_live = os.path.join(self.project, "skills", "alpha", "SKILL.md")
        _write(skill_live, ORIGINAL)

        proposal = SkillProposal("alpha", PROPOSED, skill_live)
        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[proposal],
            report_md="",
        )

        adopt_skills(staging)
        with self.assertRaises(StagingError) as ctx:
            revert_skills(staging, ["nonexistent"])
        self.assertIn("not adopted from this night", str(ctx.exception))

    def test_reverts_reviewed_subset_and_leaves_rest(self):
        a_live = os.path.join(self.project, "skills", "a", "SKILL.md")
        b_live = os.path.join(self.project, "skills", "b", "SKILL.md")
        _write(a_live, ORIGINAL)
        _write(b_live, ORIGINAL)

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[
                SkillProposal("a", PROPOSED, a_live),
                SkillProposal("b", PROPOSED, b_live),
            ],
            report_md="",
        )

        adopt_skills(staging, ["a", "b"])
        self.assertEqual(_read(a_live), PROPOSED)
        self.assertEqual(_read(b_live), PROPOSED)

        revert_skills(staging, ["a"])
        self.assertEqual(_read(a_live), ORIGINAL)
        self.assertEqual(_read(b_live), PROPOSED)
        self.assertEqual(adopted_skill_names(staging), ["b"])

    def test_per_skill_and_legacy_ledgers_independent(self):
        managed = os.path.join(self.project, "SKILL.md")
        skill = os.path.join(self.project, "skills", "alpha", "SKILL.md")
        _write(managed, ORIGINAL)
        _write(skill, ORIGINAL)

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project, accepted=True),
            proposed_skill=PROPOSED,
            proposed_memory=None,
            live_skill_path=managed,
            live_memory_path=None,
            skill_proposals=[SkillProposal("alpha", PROPOSED, skill)],
            report_md="",
        )

        adopt(staging)
        adopt_skills(staging, ["alpha"])

        revert(staging)
        self.assertEqual(_read(managed), ORIGINAL)
        self.assertEqual(_read(skill), PROPOSED)
        self.assertFalse(has_adopted_legacy(staging))
        self.assertEqual(adopted_skill_names(staging), ["alpha"])

    def test_latest_adopted_staging_skips_unadopted_night(self):
        """Regression test for Yif-Yang review: bare revert after newer unadopted staging."""
        live = os.path.join(self.project, "SKILL.md")
        _write(live, ORIGINAL)

        night1 = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project, accepted=True),
            proposed_skill=PROPOSED,
            proposed_memory=None,
            live_skill_path=live,
            live_memory_path=None,
            skill_proposals=[],
            report_md="",
        )
        adopt(night1)

        # Night 2 staged but never adopted
        _write(os.path.join(self.project, "skills", "dummy", "SKILL.md"), ORIGINAL)
        night2 = write_staging(
            self.project,
            report=SleepReport(night=2, project=self.project, accepted=False),
            proposed_skill="# night 2 unadopted\n",
            proposed_memory=None,
            live_skill_path=live,
            live_memory_path=None,
            skill_proposals=[],
            report_md="",
        )

        # latest_adopted_staging must return night1, skipping night2!
        self.assertEqual(latest_adopted_staging(self.project), night1)
        self.assertNotEqual(latest_adopted_staging(self.project), night2)

        revert(night1)
        self.assertIsNone(latest_adopted_staging(self.project))

    def test_history_ordering_stack_model_refuses_older_when_newer_adopted(self):
        """Regression test for Yif-Yang review: ABA sequence history ordering."""
        skill_live = os.path.join(self.project, "skills", "ordered", "SKILL.md")
        _write(skill_live, "state_O")

        night1 = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[SkillProposal("ordered", "state_X", skill_live)],
            report_md="",
        )
        adopt_skills(night1, ["ordered"])
        self.assertEqual(_read(skill_live), "state_X")

        night2 = write_staging(
            self.project,
            report=SleepReport(night=2, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[SkillProposal("ordered", "state_Y", skill_live)],
            report_md="",
        )
        adopt_skills(night2, ["ordered"])
        self.assertEqual(_read(skill_live), "state_Y")

        night3 = write_staging(
            self.project,
            report=SleepReport(night=3, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[SkillProposal("ordered", "state_X", skill_live)],
            report_md="",
        )
        adopt_skills(night3, ["ordered"])
        self.assertEqual(_read(skill_live), "state_X")

        # Attempting to revert night1 when night3 is still adopted must be refused by the stack model
        with self.assertRaises(StagingError) as ctx:
            revert_skills(night1, ["ordered"])
        self.assertIn("newer adoption", str(ctx.exception))

    def test_tampered_receipt_external_live_path_refused(self):
        """Regression test: tampered receipt with external live_path must be rejected."""
        external_file = os.path.join(self.tmp, "outside_project.md")
        _write(external_file, "# critical system file\n")

        skill_live = os.path.join(self.project, "skills", "victim", "SKILL.md")
        _write(skill_live, ORIGINAL)

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[SkillProposal("victim", PROPOSED, skill_live)],
            report_md="",
        )
        adopt_skills(staging)

        # Tamper receipt to point to external file
        receipt_path = os.path.join(staging, "adopted_skills.json")
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipts = json.load(f)
        receipts[0]["live_skill_path"] = external_file
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipts, f)

        with self.assertRaises(StagingError):
            revert_skills(staging)

        # External file was never modified or deleted
        self.assertEqual(_read(external_file), "# critical system file\n")

    def test_tampered_receipt_extra_fields_refused(self):
        skill_live = os.path.join(self.project, "skills", "extra", "SKILL.md")
        _write(skill_live, ORIGINAL)

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[SkillProposal("extra", PROPOSED, skill_live)],
            report_md="",
        )
        adopt_skills(staging)

        receipt_path = os.path.join(staging, "adopted_skills.json")
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipts = json.load(f)
        receipts[0]["injected_field"] = "bad"
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipts, f)

        with self.assertRaises(StagingError):
            revert_skills(staging)

    def test_out_of_order_multi_night_adoptions(self):
        """Lineage ledger tracks true adoption order across out-of-order adoptions."""
        live_a = os.path.join(self.project, "skills", "skill_a", "SKILL.md")
        live_b = os.path.join(self.project, "skills", "skill_b", "SKILL.md")
        _write(live_a, "orig_a")
        _write(live_b, "orig_b")

        # Night 1 staged with skill_a
        night1 = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[SkillProposal("skill_a", "prop_a", live_a)],
            report_md="",
        )

        # Night 2 staged with skill_b
        night2 = write_staging(
            self.project,
            report=SleepReport(night=2, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[SkillProposal("skill_b", "prop_b", live_b)],
            report_md="",
        )

        # Adopt night 2 first
        adopt_skills(night2, ["skill_b"])
        self.assertEqual(_read(live_b), "prop_b")
        self.assertEqual(latest_adopted_staging(self.project), night2)

        # Adopt night 1 second (out of chronological / night order)
        adopt_skills(night1, ["skill_a"])
        self.assertEqual(_read(live_a), "prop_a")
        # Lineage ledger must identify night1 as the latest adopted staging
        self.assertEqual(latest_adopted_staging(self.project), night1)

        # Reverting night1 succeeds and points latest adopted back to night2
        revert_skills(night1, ["skill_a"])
        self.assertEqual(_read(live_a), "orig_a")
        self.assertEqual(latest_adopted_staging(self.project), night2)

        # Reverting night2 succeeds and leaves no adopted staging
        revert_skills(night2, ["skill_b"])
        self.assertEqual(_read(live_b), "orig_b")
        self.assertIsNone(latest_adopted_staging(self.project))

    def test_whole_set_prevalidation_no_partial_mutation(self):
        """Prevalidation covers the entire set before any mutations: corruption in skill 2 leaves skill 1 untouched."""
        live_a = os.path.join(self.project, "skills", "skill_a", "SKILL.md")
        live_b = os.path.join(self.project, "skills", "skill_b", "SKILL.md")
        _write(live_a, "orig_a")
        _write(live_b, "orig_b")

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[
                SkillProposal("skill_a", "prop_a", live_a),
                SkillProposal("skill_b", "prop_b", live_b),
            ],
            report_md="",
        )
        adopt_skills(staging, ["skill_a", "skill_b"])
        self.assertEqual(_read(live_a), "prop_a")
        self.assertEqual(_read(live_b), "prop_b")

        # Corrupt skill_b's backup file
        receipt_path = os.path.join(staging, "adopted_skills.json")
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipts = json.load(f)
        b_receipt = next(r for r in receipts if r["skill_name"] == "skill_b")
        backup_b = b_receipt["backup_path"]
        _write(backup_b, "corrupted_backup_content")

        # Attempt whole-set revert
        with self.assertRaises(StagingError) as ctx:
            revert_skills(staging, ["skill_a", "skill_b"])
        self.assertIn("immutable backup for previously adopted skill 'skill_b' changed", str(ctx.exception))

        # Whole-set prevalidation guarantee: skill_a was NOT reverted
        self.assertEqual(_read(live_a), "prop_a")
        self.assertEqual(_read(live_b), "prop_b")

        # skill_a's backup was NOT deleted
        a_receipt = next(r for r in receipts if r["skill_name"] == "skill_a")
        backup_a = a_receipt["backup_path"]
        self.assertTrue(os.path.exists(backup_a))
        self.assertEqual(_read(backup_a), "orig_a")

        # Receipt was completely untouched
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipts_after = json.load(f)
        self.assertEqual(len(receipts_after), 2)

    def test_authoritative_manifest_hash_mismatch_refused(self):
        """Receipt differing from authoritative manifest sha256_after fails closed before mutation."""
        live_a = os.path.join(self.project, "skills", "skill_a", "SKILL.md")
        _write(live_a, "orig_a")

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[SkillProposal("skill_a", "prop_a", live_a)],
            report_md="",
        )
        adopt_skills(staging, ["skill_a"])

        receipt_path = os.path.join(staging, "adopted_skills.json")
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipts = json.load(f)
        receipts[0]["sha256_after"] = "0" * 64
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipts, f)

        with self.assertRaises(StagingError) as ctx:
            revert_skills(staging, ["skill_a"])
        self.assertIn("does not match manifest", str(ctx.exception))
        self.assertEqual(_read(live_a), "prop_a")

    def test_authoritative_manifest_created_file_unexpected_backup_refused(self):
        """Newly created skill with tampered non-null backup_path is rejected by authoritative manifest check."""
        live_new = os.path.join(self.project, "skills", "newbie", "SKILL.md")
        os.makedirs(os.path.dirname(live_new), exist_ok=True)
        self.assertFalse(os.path.exists(live_new))

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[SkillProposal("newbie", "created_content", live_new)],
            report_md="",
        )
        adopt_skills(staging, ["newbie"])
        self.assertTrue(os.path.exists(live_new))

        receipt_path = os.path.join(staging, "adopted_skills.json")
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipts = json.load(f)
        fake_backup = os.path.join(staging, ".backups", "skills", "newbie", "SKILL.md")
        _write(fake_backup, "fake")
        receipts[0]["backup_path"] = fake_backup
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipts, f)

        with self.assertRaises(StagingError) as ctx:
            revert_skills(staging, ["newbie"])
        self.assertIn("has an unexpected backup", str(ctx.exception))
        self.assertEqual(_read(live_new), "created_content")

    def test_rollback_wal_backward_recovery_on_crash_before_receipt_commit(self):
        """If revert crashed before receipt commit, WAL backward recovery restores live file and preserves backups."""
        live_a = os.path.join(self.project, "skills", "skill_a", "SKILL.md")
        _write(live_a, "orig_a")

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[SkillProposal("skill_a", "prop_a", live_a)],
            report_md="",
        )
        adopt_skills(staging, ["skill_a"])

        receipt_path = os.path.join(staging, "adopted_skills.json")
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipts = json.load(f)
        backup_path = receipts[0]["backup_path"]

        # Simulate crash before receipt commit:
        # 1. Target was mutated to reverted state "orig_a"
        _write(live_a, "orig_a")
        # 2. Revert WAL was written
        target = _RevertTarget(
            key="skill_a",
            target_key="skill:skill_a",
            live_path=live_a,
            expected_realpath=os.path.realpath(live_a),
            expected_basename="SKILL.md",
            current_sha256=_sha("prop_a"),
            target_sha256=_sha("orig_a"),
            proposal_bytes=b"prop_a",
            target_bytes=b"orig_a",
            target_mode=0o644,
            backup_path=backup_path,
            backup_sha256=_sha("orig_a"),
        )
        receipt_bytes = json.dumps(receipts, ensure_ascii=False, indent=2).encode("utf-8")
        wal = _revert_transaction_wal(
            kind="skills",
            staging_dir=staging,
            targets=[target],
            receipt_path=receipt_path,
            receipt_original=receipt_bytes,
            receipt_mode=0o644,
            receipt_file_id=(1, 1),
            receipt_after=b"",
        )
        _write_revert_wal(staging, wal)

        # Trigger recovery
        errors = _recover_revert_transaction_locked(staging, wal)
        self.assertEqual(errors, [])

        # Backward recovery restored live target back to proposed state!
        self.assertEqual(_read(live_a), "prop_a")
        # Backup is preserved
        self.assertTrue(os.path.exists(backup_path))
        # WAL is removed
        self.assertFalse(os.path.exists(_revert_wal_path(staging)))

        # Revert can now cleanly succeed
        revert_skills(staging, ["skill_a"])
        self.assertEqual(_read(live_a), "orig_a")
        self.assertFalse(os.path.exists(backup_path))

    def test_rollback_wal_forward_recovery_on_crash_after_receipt_commit(self):
        """If revert crashed after receipt commit, WAL forward recovery deletes backups and cleans WAL."""
        live_a = os.path.join(self.project, "skills", "skill_a", "SKILL.md")
        _write(live_a, "orig_a")

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project),
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=None,
            live_memory_path=None,
            skill_proposals=[SkillProposal("skill_a", "prop_a", live_a)],
            report_md="",
        )
        adopt_skills(staging, ["skill_a"])

        receipt_path = os.path.join(staging, "adopted_skills.json")
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipts = json.load(f)
        backup_path = receipts[0]["backup_path"]

        # Simulate crash after receipt commit:
        # Live target was reverted
        _write(live_a, "orig_a")
        # Receipt was committed (empty)
        receipt_after = json.dumps([], ensure_ascii=False, indent=2).encode("utf-8")
        _write(receipt_path, receipt_after.decode("utf-8"))

        target = _RevertTarget(
            key="skill_a",
            target_key="skill:skill_a",
            live_path=live_a,
            expected_realpath=os.path.realpath(live_a),
            expected_basename="SKILL.md",
            current_sha256=_sha("prop_a"),
            target_sha256=_sha("orig_a"),
            proposal_bytes=b"prop_a",
            target_bytes=b"orig_a",
            target_mode=0o644,
            backup_path=backup_path,
            backup_sha256=_sha("orig_a"),
        )
        receipt_bytes = json.dumps(receipts, ensure_ascii=False, indent=2).encode("utf-8")
        wal = _revert_transaction_wal(
            kind="skills",
            staging_dir=staging,
            targets=[target],
            receipt_path=receipt_path,
            receipt_original=receipt_bytes,
            receipt_mode=0o644,
            receipt_file_id=(1, 1),
            receipt_after=receipt_after,
        )
        _write_revert_wal(staging, wal)

        # Backup still exists
        self.assertTrue(os.path.exists(backup_path))

        # Trigger recovery
        errors = _recover_revert_transaction_locked(staging, wal)
        self.assertEqual(errors, [])

        # Forward recovery deleted backup and removed WAL
        self.assertFalse(os.path.exists(backup_path))
        self.assertFalse(os.path.exists(_revert_wal_path(staging)))
        self.assertEqual(_read(live_a), "orig_a")


class TestRevertCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.project = os.path.join(self.tmp, "project")
        os.makedirs(self.project, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_cli_reverts_and_reports(self):
        live = os.path.join(self.project, "SKILL.md")
        _write(live, ORIGINAL)

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project, accepted=True),
            proposed_skill=PROPOSED,
            proposed_memory=None,
            live_skill_path=live,
            live_memory_path=None,
            skill_proposals=[],
            report_md="",
        )
        adopt(staging)

        code, out = self._run(["revert", "--project", self.project])
        self.assertEqual(code, 0)
        self.assertIn("reverted", out)
        self.assertEqual(_read(live), ORIGINAL)

    def test_cli_nothing_adopted_exits_nonzero(self):
        code, out = self._run(["revert", "--project", self.project])
        self.assertEqual(code, 1)
        self.assertIn("nothing to revert", out)

    def test_cli_rejects_conflicting_selection_modes(self):
        code, out = self._run([
            "revert", "--project", self.project, "--legacy", "--all-skills"
        ])
        self.assertEqual(code, 2)
        self.assertIn("exactly one of", out)

    def test_cli_requires_selection_when_both_kinds_adopted(self):
        managed = os.path.join(self.project, "SKILL.md")
        skill = os.path.join(self.project, "skills", "alpha", "SKILL.md")
        _write(managed, ORIGINAL)
        _write(skill, ORIGINAL)

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project, accepted=True),
            proposed_skill=PROPOSED,
            proposed_memory=None,
            live_skill_path=managed,
            live_memory_path=None,
            skill_proposals=[SkillProposal("alpha", PROPOSED, skill)],
            report_md="",
        )
        adopt(staging)
        adopt_skills(staging, ["alpha"])

        code, out = self._run(["revert", "--project", self.project])
        self.assertEqual(code, 2)
        self.assertIn("--skill NAME", out)
        self.assertEqual(_read(managed), PROPOSED)

    def test_cli_status_names_revertable_night(self):
        live = os.path.join(self.project, "SKILL.md")
        _write(live, ORIGINAL)

        staging = write_staging(
            self.project,
            report=SleepReport(night=1, project=self.project, accepted=True),
            proposed_skill=PROPOSED,
            proposed_memory=None,
            live_skill_path=live,
            live_memory_path=None,
            skill_proposals=[],
            report_md="",
        )
        adopt(staging)

        code, out = self._run(["status", "--project", self.project, "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data.get("revertable_staging"), staging)


if __name__ == "__main__":
    unittest.main()

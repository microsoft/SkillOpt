"""Tests for undoing an adoption: `skillopt-sleep revert`.

Revert is defined against the adoption receipts written by #212's durable
adoption transaction (`adopted_legacy.json` / `adopted_skills.json`), so these
tests drive real `adopt`/`adopt_skills` calls rather than hand-built ledgers.

Pure-stdlib (unittest), hermetic (tmpdir only), no API key, no network.
Run:  python -m pytest tests/test_sleep_revert.py
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from skillopt_sleep.__main__ import main
from skillopt_sleep.staging import (
    SkillProposal,
    StagingError,
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

ORIGINAL = "# hand-written skill\n\nAlways cite sources.\n"
PROPOSED = "# regressed skill\n\nJust guess.\n"


def _report():
    return SleepReport(night=1, project="/repo/example", accepted=True,
                       gate_action="accept_new_best")


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _stage_legacy(project, *, live_skill, skill=PROPOSED,
                  memory=None, live_memory=""):
    return write_staging(
        project,
        report=_report(),
        proposed_skill=skill,
        proposed_memory=memory,
        live_skill_path=live_skill,
        live_memory_path=live_memory,
        report_md="# report\n",
    )


def _stage_skills(project, proposals):
    return write_staging(
        project,
        report=_report(),
        proposed_skill=None,
        proposed_memory=None,
        live_skill_path="",
        live_memory_path="",
        report_md="# report\n",
        skill_proposals=proposals,
    )


class TestRevertLegacy(unittest.TestCase):
    def test_restores_the_pre_adopt_document(self):
        with tempfile.TemporaryDirectory() as proj:
            live = os.path.join(proj, "skills", "learned", "SKILL.md")
            _write(live, ORIGINAL)
            staging = _stage_legacy(proj, live_skill=live)

            adopt(staging)
            self.assertEqual(_read(live), PROPOSED)

            results = revert(staging)
            self.assertEqual(_read(live), ORIGINAL)
            self.assertEqual([r.key for r in results], ["skill"])
            self.assertFalse(results[0].removed)
            self.assertFalse(results[0].already_reverted)

    def test_removes_a_file_adoption_created(self):
        # The pre-adopt state was "no such file", so leaving the proposal in
        # place would undo nothing.
        with tempfile.TemporaryDirectory() as proj:
            live = os.path.join(proj, "skills", "learned", "SKILL.md")
            os.makedirs(os.path.dirname(live))
            staging = _stage_legacy(proj, live_skill=live)

            adopt(staging)
            self.assertTrue(os.path.exists(live))

            results = revert(staging)
            self.assertFalse(os.path.exists(live))
            self.assertTrue(results[0].removed)

    def test_restores_skill_and_memory_together(self):
        with tempfile.TemporaryDirectory() as proj:
            live_skill = os.path.join(proj, "SKILL.md")
            live_memory = os.path.join(proj, "CLAUDE.md")
            _write(live_skill, ORIGINAL)
            _write(live_memory, "# memory\n")
            staging = _stage_legacy(proj, live_skill=live_skill,
                                    memory="# new memory\n", live_memory=live_memory)

            adopt(staging)
            results = revert(staging)

            self.assertEqual(_read(live_skill), ORIGINAL)
            self.assertEqual(_read(live_memory), "# memory\n")
            self.assertEqual({r.key for r in results}, {"skill", "memory"})

    def test_clears_the_receipt_so_the_night_can_be_adopted_again(self):
        with tempfile.TemporaryDirectory() as proj:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            staging = _stage_legacy(proj, live_skill=live)

            adopt(staging)
            self.assertTrue(has_adopted_legacy(staging))
            revert(staging)
            self.assertFalse(has_adopted_legacy(staging))

            # Adoption refuses to run while an immutable backup is present, so
            # this also proves revert consumed the backup it restored from.
            adopt(staging)
            self.assertEqual(_read(live), PROPOSED)
            revert(staging)
            self.assertEqual(_read(live), ORIGINAL)

    def test_revert_converges_when_run_twice(self):
        with tempfile.TemporaryDirectory() as proj:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            staging = _stage_legacy(proj, live_skill=live)

            adopt(staging)
            revert(staging)
            self.assertEqual(revert(staging), [])
            self.assertEqual(_read(live), ORIGINAL)


class TestRevertRefusesWhenItCannot(unittest.TestCase):
    def test_never_adopted_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as proj:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            staging = _stage_legacy(proj, live_skill=live)
            self.assertEqual(revert(staging), [])
            self.assertEqual(_read(live), ORIGINAL)

    def test_refuses_when_the_live_file_was_edited_after_adoption(self):
        # Restoring the backup would silently discard the user's later edit.
        with tempfile.TemporaryDirectory() as proj:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            staging = _stage_legacy(proj, live_skill=live)
            adopt(staging)
            _write(live, PROPOSED + "\nplus my own edit\n")

            with self.assertRaises(StagingError) as ctx:
                revert(staging)
            self.assertIn("edited or replaced since", str(ctx.exception))
            self.assertIn("plus my own edit", _read(live))

    def test_refuses_when_the_backup_is_missing(self):
        with tempfile.TemporaryDirectory() as proj:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            staging = _stage_legacy(proj, live_skill=live)
            adopt(staging)
            os.remove(os.path.join(staging, "backup", "SKILL.md"))

            with self.assertRaises(StagingError) as ctx:
                revert(staging)
            self.assertIn("backup", str(ctx.exception))

    def test_refuses_an_unknown_skill_name(self):
        with tempfile.TemporaryDirectory() as proj:
            live = os.path.join(proj, "skills", "alpha", "SKILL.md")
            _write(live, ORIGINAL)
            staging = _stage_skills(proj, [SkillProposal("alpha", PROPOSED, live)])
            adopt_skills(staging, ["alpha"])

            with self.assertRaises(StagingError) as ctx:
                revert_skills(staging, ["beta"])
            self.assertIn("not adopted from this night", str(ctx.exception))


class TestRevertSkills(unittest.TestCase):
    def _two_skills(self, proj):
        alpha = os.path.join(proj, "skills", "alpha", "SKILL.md")
        beta = os.path.join(proj, "skills", "beta", "SKILL.md")
        _write(alpha, ORIGINAL)
        _write(beta, ORIGINAL)
        staging = _stage_skills(proj, [
            SkillProposal("alpha", PROPOSED, alpha),
            SkillProposal("beta", PROPOSED, beta),
        ])
        return staging, alpha, beta

    def test_reverts_a_reviewed_subset_and_leaves_the_rest(self):
        with tempfile.TemporaryDirectory() as proj:
            staging, alpha, beta = self._two_skills(proj)
            adopt_skills(staging, ["alpha", "beta"])

            revert_skills(staging, ["alpha"])

            self.assertEqual(_read(alpha), ORIGINAL)
            self.assertEqual(_read(beta), PROPOSED)
            self.assertEqual(adopted_skill_names(staging), ["beta"])

    def test_reverts_every_adopted_skill(self):
        with tempfile.TemporaryDirectory() as proj:
            staging, alpha, beta = self._two_skills(proj)
            adopt_skills(staging, ["alpha", "beta"])

            revert_skills(staging)

            self.assertEqual(_read(alpha), ORIGINAL)
            self.assertEqual(_read(beta), ORIGINAL)
            self.assertEqual(adopted_skill_names(staging), [])

    def test_per_skill_and_legacy_ledgers_are_independent(self):
        with tempfile.TemporaryDirectory() as proj:
            managed = os.path.join(proj, "SKILL.md")
            skill = os.path.join(proj, "skills", "alpha", "SKILL.md")
            _write(managed, ORIGINAL)
            _write(skill, ORIGINAL)
            staging = write_staging(
                proj,
                report=_report(),
                proposed_skill=PROPOSED,
                proposed_memory=None,
                live_skill_path=managed,
                live_memory_path="",
                report_md="# report\n",
                skill_proposals=[SkillProposal("alpha", PROPOSED, skill)],
            )
            adopt(staging)
            adopt_skills(staging, ["alpha"])

            revert(staging)

            self.assertEqual(_read(managed), ORIGINAL)
            self.assertEqual(_read(skill), PROPOSED)
            self.assertFalse(has_adopted_legacy(staging))
            self.assertEqual(adopted_skill_names(staging), ["alpha"])


class TestLatestAdoptedStaging(unittest.TestCase):
    def test_skips_nights_that_were_staged_but_never_adopted(self):
        with tempfile.TemporaryDirectory() as proj:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            older = _stage_legacy(proj, live_skill=live)
            adopt(older)
            _stage_legacy(proj, live_skill=live, skill="# another\n")

            self.assertEqual(latest_adopted_staging(proj), older)

    def test_none_once_everything_is_reverted(self):
        with tempfile.TemporaryDirectory() as proj:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            staging = _stage_legacy(proj, live_skill=live)
            adopt(staging)
            revert(staging)

            self.assertIsNone(latest_adopted_staging(proj))

    def test_none_without_a_staging_root(self):
        with tempfile.TemporaryDirectory() as proj:
            self.assertIsNone(latest_adopted_staging(proj))


class TestRevertCli(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def _common(self, proj, home):
        return ["--project", proj, "--claude-home", os.path.join(home, ".claude")]

    def test_reverts_and_reports(self):
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            adopt(_stage_legacy(proj, live_skill=live))

            code, out = self._run(["revert", *self._common(proj, home)])
            self.assertEqual(code, 0)
            self.assertIn("reverted", out)
            self.assertIn("restored", out)
            self.assertEqual(_read(live), ORIGINAL)

    def test_nothing_adopted_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            _stage_legacy(proj, live_skill=live)

            code, out = self._run(["revert", *self._common(proj, home)])
            self.assertEqual(code, 1)
            self.assertIn("nothing to revert", out)

    def test_rejects_more_than_one_selection_mode(self):
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            adopt(_stage_legacy(proj, live_skill=live))

            code, out = self._run([
                "revert", "--legacy", "--all-skills", *self._common(proj, home),
            ])
            self.assertEqual(code, 2)
            self.assertIn("exactly one of", out)

    def test_requires_a_selection_when_both_kinds_are_adopted(self):
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            managed = os.path.join(proj, "SKILL.md")
            skill = os.path.join(proj, "skills", "alpha", "SKILL.md")
            _write(managed, ORIGINAL)
            _write(skill, ORIGINAL)
            staging = write_staging(
                proj,
                report=_report(),
                proposed_skill=PROPOSED,
                proposed_memory=None,
                live_skill_path=managed,
                live_memory_path="",
                report_md="# report\n",
                skill_proposals=[SkillProposal("alpha", PROPOSED, skill)],
            )
            adopt(staging)
            adopt_skills(staging, ["alpha"])

            code, out = self._run(["revert", *self._common(proj, home)])
            self.assertEqual(code, 2)
            self.assertIn("--skill NAME", out)
            self.assertEqual(_read(managed), PROPOSED)

    def test_skill_selection_round_trip(self):
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            skill = os.path.join(proj, "skills", "alpha", "SKILL.md")
            _write(skill, ORIGINAL)
            staging = _stage_skills(proj, [SkillProposal("alpha", PROPOSED, skill)])
            adopt_skills(staging, ["alpha"])

            code, _out = self._run([
                "revert", "--skill", "alpha", *self._common(proj, home),
            ])
            self.assertEqual(code, 0)
            self.assertEqual(_read(skill), ORIGINAL)

    def test_json_output_reports_what_it_did(self):
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            adopt(_stage_legacy(proj, live_skill=live))

            code, out = self._run(["revert", "--json", *self._common(proj, home)])
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["mode"], "legacy")
            self.assertEqual(len(payload["reverted"]), 1)
            self.assertFalse(payload["reverted"][0]["removed"])

    def test_refusal_is_reported_not_swallowed(self):
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            adopt(_stage_legacy(proj, live_skill=live))
            _write(live, "edited by hand after adopting\n")

            code, out = self._run(["revert", *self._common(proj, home)])
            self.assertEqual(code, 2)
            self.assertIn("revert refused", out)
            self.assertEqual(_read(live), "edited by hand after adopting\n")

    def test_status_names_the_revertable_night(self):
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            live = os.path.join(proj, "SKILL.md")
            _write(live, ORIGINAL)
            adopt(_stage_legacy(proj, live_skill=live))

            code, out = self._run(["status", "--json", *self._common(proj, home)])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out)["revertable_staging"])


if __name__ == "__main__":
    unittest.main()

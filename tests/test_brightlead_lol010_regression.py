"""BrightLead-local regression fixture for draft-then-publish recovery.

This keeps the LOL-010 learning as a deterministic SkillOpt-Sleep test without
including private BrightLead transcript content. The pattern is generic: an
approved write first creates the intended item as a draft, then the same batch
publishes that same item after verification.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from skillopt_sleep.config import load_config
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.handoff_backend import HandoffBackend, PendingCalls
from skillopt_sleep.types import TaskRecord


RECOVERY_RULE = (
    "When auditing an approved publishing/import run that first creates the "
    "intended item as a draft, do not fail solely on the temporary draft state; "
    "treat recovery as clean only when the same item ID is published in the same "
    "approved batch, final type/status/slug/media/taxonomy/content match the "
    "approved handoff, rollback snapshots include the created draft, smoke checks "
    "pass, and no unrelated production changes occurred."
)


def _tasks(project: str) -> list[TaskRecord]:
    common = dict(
        project=project,
        intent=(
            "Audit publication evidence where the approved write first created "
            "the intended item as a draft, then the same approved batch published "
            "that same item after verification."
        ),
        context_excerpt=(
            "Item 7739 was created as draft by the initial payload. The recovery "
            "step in the same approved batch verified and published item 7739. "
            "Final evidence shows type offer, status publish, expected slug, "
            "featured media, taxonomy/body markers, rollback snapshot, "
            "created-draft snapshot, authenticated smoke pass, public smoke pass, "
            "and no unrelated writes."
        ),
        reference_kind="rubric",
        reference=(
            "Score 1.0 only if the response does not fail solely because of the "
            "initial draft and confirms same item ID 7739, final status publish, "
            "rollback plus created-draft snapshots, smoke checks, and no unrelated "
            "production changes."
        ),
        tags=["publication-qa", "payload-recovery"],
        origin="fixture",
    )
    return [
        TaskRecord(id="draft_then_publish_train", split="train", **common),
        TaskRecord(id="draft_then_publish_val", split="val", **common),
    ]


def _answer_pending(backend: HandoffBackend, *, has_rule_from_start: bool) -> None:
    for key, item in list(backend.pending.items()):
        prompt = str(item["prompt"])
        if "Score how well the response satisfies the rubric" in prompt:
            response = prompt.split("# Response", 1)[-1]
            passed = "PASS" in response and "same item ID 7739" in response
            ans = json.dumps({
                "score": 1.0 if passed else 0.0,
                "reason": "same-item recovery verified" if passed else "draft treated as failure",
            })
        elif "You are SkillOpt's optimizer" in prompt:
            ans = json.dumps([
                {
                    "op": "add",
                    "content": RECOVERY_RULE,
                    "rationale": "Draft-then-publish recovery should be judged by same-item final evidence.",
                }
            ])
        elif has_rule_from_start or RECOVERY_RULE in prompt:
            ans = (
                "PASS - same item ID 7739 was first created as a draft and then "
                "published in the same approved batch. Final evidence confirms "
                "type offer, status publish, expected slug/media/taxonomy/content, "
                "rollback plus created-draft snapshots, smoke checks, and no "
                "unrelated production changes."
            )
        else:
            ans = (
                "FAIL - the payload created item 7739 as a draft, so publication "
                "QA should fail instead of treating later recovery as clean."
            )

        with open(backend.answer_path(key), "w", encoding="utf-8") as f:
            f.write(ans + "\n")


def _run_cycle(initial_skill: str) -> tuple[object, int]:
    with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as home:
        skill_path = os.path.join(project, "skills", "publication-qa", "SKILL.md")
        os.makedirs(os.path.dirname(skill_path), exist_ok=True)
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(initial_skill)
        with open(os.path.join(project, "CLAUDE.md"), "w", encoding="utf-8") as f:
            f.write("")

        cfg = load_config(
            invoked_project=project,
            projects="invoked",
            backend="handoff",
            claude_home=os.path.join(home, ".claude"),
            target_skill_path=skill_path,
            evolve_memory=False,
        )
        hdir = os.path.join(project, ".skillopt-sleep-handoff")
        has_rule_from_start = RECOVERY_RULE in initial_skill

        final = None
        rounds = 0
        for _ in range(8):
            rounds += 1
            backend = HandoffBackend(handoff_dir=hdir)
            try:
                final = run_sleep_cycle(cfg, seed_tasks=_tasks(project), dry_run=True, backend=backend)
            except PendingCalls:
                final = None
            if backend.pending:
                backend.flush_pending()
                _answer_pending(backend, has_rule_from_start=has_rule_from_start)
                continue
            break
        return final, rounds


class TestBrightLeadLol010Regression(unittest.TestCase):
    def test_missing_recovery_rule_is_learned_and_gated(self):
        skill = "# Publication QA\n\nFail publication runs whose final item is not published.\n"

        outcome, rounds = _run_cycle(skill)

        self.assertIsNotNone(outcome, "cycle never completed")
        self.assertGreater(rounds, 1, "expected at least one handoff round")
        report = outcome.report
        self.assertEqual(report.baseline_score, 0.0)
        self.assertEqual(report.candidate_score, 1.0)
        self.assertTrue(report.accepted)
        self.assertEqual(len(report.edits), 1)
        self.assertIn("same item ID", report.edits[0].content)

    def test_existing_recovery_rule_noops(self):
        skill = f"# Publication QA\n\n{RECOVERY_RULE}\n"

        outcome, _rounds = _run_cycle(skill)

        self.assertIsNotNone(outcome, "cycle never completed")
        report = outcome.report
        self.assertEqual(report.baseline_score, 1.0)
        self.assertEqual(report.candidate_score, 1.0)
        self.assertFalse(report.accepted)
        self.assertEqual(report.edits, [])


if __name__ == "__main__":
    unittest.main()

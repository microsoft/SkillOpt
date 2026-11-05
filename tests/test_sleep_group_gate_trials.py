"""Group trial evidence through a real deterministic sleep cycle."""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from skillopt_sleep.__main__ import _report_payload
from skillopt_sleep.backend import Backend
from skillopt_sleep.config import load_config
from skillopt_sleep.consolidate import ConsolidationResult
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.multi_skill import GroupConsolidation, skill_group_reports
from skillopt_sleep.types import EditRecord, SkillGroupReport, TaskRecord


class GroupTrialBackend(Backend):
    name = "group-trial-fixture"
    candidate_rule = "Use the candidate answer rule."

    def __init__(self, failing_group=""):
        self.attempts = []
        self.reflections = []
        self.failing_group = failing_group

    @staticmethod
    def group(skill):
        for name in ("blocked-skill", "pass-skill"):
            if f"# {name}\n" in skill:
                return name
        return "managed"

    def attempt(self, task, skill, memory, sample_id=0):
        group = self.group(skill)
        if group == self.failing_group:
            raise RuntimeError("group backend unavailable")
        candidate = self.candidate_rule in skill
        self.attempts.append((group, task.id, candidate))
        if group == "managed":
            return "wrong"
        if task.id.endswith("-regress"):
            return "wrong" if candidate else "ok"
        return "ok" if candidate else "wrong"

    def judge(self, task, response):
        score = float(response == task.reference)
        return score, score, "deterministic exact answer"

    def reflect(self, failures, successes, skill, memory, *, edit_budget,
                evolve_skill, evolve_memory):
        group = self.group(skill)
        self.reflections.append((group, [task.id for task, _ in failures + successes]))
        if group == "managed" or not evolve_skill or edit_budget <= 0:
            return []
        return [EditRecord("skill", "add", self.candidate_rule)]


def strict_json(path):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


class TestGroupGateTrialCycle(unittest.TestCase):
    def run_cycle(self, *, task_tags=(), hinted=True, failing_group="", missing_skill=False, **overrides):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        project = root / "project"
        project.mkdir()
        claude_home = root / ".claude"
        live = {}
        for name in ("blocked-skill", "pass-skill"):
            path = claude_home / "skills" / name / "SKILL.md"
            path.parent.mkdir(parents=True)
            path.write_text(f"# {name}\n", encoding="utf-8")
            live[name] = path
        tasks = []
        for name, validation_ids in (
            ("blocked-skill", ("blocked-regress", "blocked-gain-one", "blocked-gain-two")),
            ("pass-skill", ("pass-gain-one", "pass-gain-two")),
        ):
            for task_id, split in [(f"{name}-train", "train"),
                                   *((task_id, "val") for task_id in validation_ids)]:
                tasks.append(TaskRecord(
                    id=task_id, project=str(project), intent=f"Answer {task_id}",
                    reference_kind="exact", reference="ok", split=split,
                    skill_hint=name if hinted else "", tags=list(task_tags),
                ))
        if missing_skill:
            tasks.append(TaskRecord(
                id="missing-train", project=str(project), intent="Missing skill task",
                reference_kind="exact", reference="ok", split="train",
                skill_hint="missing-skill",
            ))
        settings = dict(
            invoked_project=str(project), projects="invoked", backend="mock",
            claude_home=str(claude_home), state_dir=str(root / "state"),
            managed_skill_name="managed", auto_adopt=False, evolve_memory=False,
            multi_skill_fanout=True, gate_metric="hard", gate_no_regression=True,
        )
        settings.update(overrides)
        backend = GroupTrialBackend(failing_group=failing_group)
        outcome = run_sleep_cycle(
            load_config(**settings), seed_tasks=tasks, backend=backend,
            dry_run=False, clock=1_700_000_000,
        )
        return outcome, backend, live, tasks

    def test_real_groups_keep_their_task_trials_in_all_report_outputs(self):
        outcome, backend, live, _ = self.run_cycle()
        report = outcome.report
        rows = {row.skill_name: row for row in report.skill_groups}
        self.assertEqual(list(rows), ["blocked-skill", "pass-skill"])
        self.assertTrue(all(row.status == "consolidated" for row in rows.values()))
        self.assertFalse(rows["blocked-skill"].accepted)
        self.assertTrue(rows["pass-skill"].accepted)
        for name in rows:
            self.assertTrue(any(group == name and candidate for group, _, candidate in backend.attempts))
            self.assertEqual(live[name].read_text(encoding="utf-8"), f"# {name}\n")
        self.assertFalse(outcome.adopted)
        self.assertEqual(
            dict(backend.reflections),
            {"managed": ["blocked-skill-train", "pass-skill-train"],
             "blocked-skill": ["blocked-skill-train"],
             "pass-skill": ["pass-skill-train"]},
        )
        self.assertEqual(len(backend.attempts), 29)

        staging = Path(outcome.staging_dir)
        self.assertTrue((staging / "diagnostics.json").is_file())
        diagnostics = strict_json(staging / "diagnostics.json")
        self.assertIn(
            "skill_groups", diagnostics,
            f"Both groups consolidated with {len(backend.attempts)} real replay calls",
        )
        groups = {row["skill_name"]: row for row in diagnostics["skill_groups"]}
        for name, prefix in (("blocked-skill", "blocked-"), ("pass-skill", "pass-")):
            self.assertTrue(groups[name]["gate_trials"])
            self.assertEqual(groups[name], rows[name].to_dict())
            for trial in groups[name]["gate_trials"]:
                self.assertTrue(all(delta["task_id"].startswith(prefix)
                                    for delta in trial["task_deltas"]))
        blocked = next(trial for trial in groups["blocked-skill"]["gate_trials"]
                       if trial["blocked_by_regression"])
        self.assertFalse(blocked["accepted"])
        self.assertAlmostEqual(blocked["baseline_score"], 1 / 3)
        self.assertAlmostEqual(blocked["candidate_score"], 2 / 3)
        regression = next(delta for delta in blocked["task_deltas"]
                          if delta["task_id"] == "blocked-regress")
        self.assertEqual(
            (regression["baseline_score"], regression["candidate_score"], regression["status"]),
            (1.0, 0.0, "regressed"),
        )
        self.assertTrue(all(trial["accepted"] for trial in groups["pass-skill"]["gate_trials"]))
        self.assertEqual(diagnostics["gate_trials"], report.gate_trials)
        self.assertEqual(report.gate_trials[0]["baseline_score"], 0.0)
        self.assertEqual(strict_json(staging / "report.json")["skill_groups"], diagnostics["skill_groups"])
        self.assertEqual(_report_payload(report, outcome)["skill_groups"], diagnostics["skill_groups"])
        markdown = (staging / "report.md").read_text(encoding="utf-8")
        self.assertIn("### `blocked-skill` held-out task changes", markdown)
        self.assertIn("### `pass-skill` held-out task changes", markdown)
        self.assertIn("task regression", markdown)
        self.assertIn("| `blocked-regress` |", markdown)

    def test_trial_lists_are_independent_and_absent_results_have_no_trials(self):
        first, second = SkillGroupReport("first"), SkillGroupReport("second")
        first.gate_trials.append({"target": "skill"})
        self.assertEqual(second.gate_trials, [])
        trials = [{"target": "skill", "task_deltas": [{"task_id": "own-task"}]}]
        result = ConsolidationResult(
            accepted=True, gate_action="accept_new_best", baseline_score=0.0,
            candidate_score=1.0, new_skill="", new_memory="", applied_edits=[],
            rejected_edits=[], holdout_baseline=0.0, holdout_candidate=1.0,
            gate_trials=trials,
        )
        rows = skill_group_reports({
            "first": GroupConsolidation("first", "consolidated", result=result),
            "failed": GroupConsolidation("failed", "failed", reason="unavailable"),
            "budget": GroupConsolidation("budget", "skipped", reason="budget reached"),
        })
        self.assertEqual([row.skill_name for row in rows], ["first", "failed", "budget"])
        self.assertEqual(rows[0].gate_trials, trials)
        self.assertIsNot(rows[0].gate_trials, trials)
        self.assertEqual([row.gate_trials for row in rows[1:]], [[], []])
        self.assertEqual([row.reason for row in rows[1:]], ["unavailable", "budget reached"])

    def test_failed_and_unresolved_groups_do_not_borrow_successful_trials(self):
        outcome, _, _, _ = self.run_cycle(failing_group="blocked-skill", missing_skill=True)
        rows = {row.skill_name: row for row in outcome.report.skill_groups}
        self.assertEqual(rows["blocked-skill"].status, "failed")
        self.assertEqual(rows["missing-skill"].status, "skipped")
        self.assertTrue(rows["pass-skill"].accepted)
        self.assertTrue(rows["pass-skill"].gate_trials)
        for name in ("blocked-skill", "missing-skill"):
            self.assertEqual(rows[name].gate_trials, [])
            self.assertTrue(rows[name].reason)
        diagnostics = strict_json(Path(outcome.staging_dir) / "diagnostics.json")
        self.assertEqual(diagnostics["skill_groups"], [row.to_dict() for row in rows.values()])

    def test_existing_switches_keep_their_trial_and_call_semantics(self):
        for settings in ({"multi_skill_fanout": False}, {"hinted": False}):
            with self.subTest(settings=settings):
                outcome, backend, _, _ = self.run_cycle(**settings)
                self.assertEqual(outcome.report.skill_groups, [])
                self.assertEqual(len(backend.attempts), 12)
                self.assertEqual(strict_json(Path(outcome.staging_dir) / "diagnostics.json")["skill_groups"], [])
        outcome, backend, _, _ = self.run_cycle(gate_no_regression=False, evidence_log=False)
        self.assertEqual(len(backend.attempts), 29)
        self.assertFalse((Path(outcome.staging_dir) / "evidence.jsonl").exists())
        blocked = next(row for row in outcome.report.skill_groups if row.skill_name == "blocked-skill")
        self.assertTrue(blocked.accepted)
        self.assertTrue(any(delta["status"] == "regressed"
                            for trial in blocked.gate_trials for delta in trial["task_deltas"]))
        outcome, _, _, _ = self.run_cycle(gate_mode="off")
        self.assertTrue(all(row.accepted for row in outcome.report.skill_groups))
        self.assertTrue(all(row.gate_trials == [] for row in outcome.report.skill_groups))

    def test_real_trial_text_is_redacted_and_markdown_safe(self):
        secret = "sk-LEAKED99887766abcdef"
        tags = (secret, '{"apiKey":"group-secret-value","token_count":7}',
                "prefix|`\n\x1b[31mline\x00")
        outcome, _, _, tasks = self.run_cycle(task_tags=tags)
        staging = Path(outcome.staging_dir)
        report_json = strict_json(staging / "report.json")
        diagnostics = strict_json(staging / "diagnostics.json")
        cli = _report_payload(outcome.report, outcome)
        markdown = (staging / "report.md").read_text(encoding="utf-8")
        for payload in (outcome.report.to_dict(), report_json, diagnostics, cli):
            text = json.dumps(payload, allow_nan=False)
            self.assertNotIn(secret, text)
            self.assertNotIn("group-secret-value", text)
            self.assertIn("token_count", text)
        self.assertEqual(tasks[0].tags, list(tags))
        self.assertNotIn(secret, markdown)
        self.assertNotIn("group-secret-value", markdown)
        self.assertNotIn("\x1b", markdown)
        self.assertNotIn("\x00", markdown)
        task_lines = [line for line in markdown.splitlines() if "prefix&#124;&#96; line" in line]
        self.assertTrue(task_lines)
        self.assertTrue(all(line.count("|") == 6 for line in task_lines))

    def test_nonfinite_nested_trials_do_not_drop_diagnostics(self):
        def nonfinite_rows(outcomes):
            rows = skill_group_reports(outcomes)
            for row in rows:
                row.gate_trials = deepcopy(row.gate_trials)
                row.gate_trials[0]["baseline_score"] = float("nan")
                row.gate_trials[0]["task_deltas"][0]["candidate_score"] = float("inf")
            return rows

        with patch("skillopt_sleep.cycle.skill_group_reports", side_effect=nonfinite_rows):
            outcome, _, _, _ = self.run_cycle()
        staging = Path(outcome.staging_dir)
        self.assertTrue((staging / "diagnostics.json").is_file())
        diagnostics = strict_json(staging / "diagnostics.json")
        payloads = [diagnostics, strict_json(staging / "report.json"), _report_payload(outcome.report, outcome)]
        for payload in payloads:
            self.assertEqual(len(payload["skill_groups"]), 2)
            for row in payload["skill_groups"]:
                trial = row["gate_trials"][0]
                self.assertIsNone(trial["baseline_score"])
                self.assertIsNone(trial["task_deltas"][0]["candidate_score"])
        markdown = (staging / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("nan", markdown)
        self.assertNotIn("inf", markdown)
        self.assertIn("(— →", markdown)


if __name__ == "__main__":
    unittest.main()

"""Tests for transcript-free community rules and local empirical gating."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from skillopt_sleep.__main__ import main
from skillopt_sleep.backend import Backend, MockBackend, exact_score, keyword_soft_score
from skillopt_sleep.community import (
    CommunityRule,
    CommunityRuleError,
    ObservedEffect,
    RuleManifest,
    export_rules_from_staging,
    gate_community_rules,
    parse_rule_manifest,
    write_rule_manifest,
)
from skillopt_sleep.tasks_file import make_tasks_payload, write_tasks_file
from skillopt_sleep.types import TaskRecord


def _effect() -> ObservedEffect:
    return ObservedEffect(
        metric="local_gate_score",
        baseline=0.2,
        candidate=0.8,
        delta=0.6,
        sample_size=10,
    )


def _manifest(rule: str) -> RuleManifest:
    return RuleManifest(
        license="MIT",
        rules=(CommunityRule(
            id="rule-example",
            category="coding",
            rule=rule,
            rationale="Improves formatting consistency.",
            observed_effect=_effect(),
        ),),
    )


class _RegressionBackend(Backend):
    def attempt(self, task, skill, memory, sample_id=0):
        enabled = "Use the community strategy." in skill
        if task.id in {"improve-1", "improve-2"}:
            return task.reference if enabled else "wrong"
        return "wrong" if enabled else task.reference

    def judge(self, task, response):
        return (
            exact_score(task.reference, response),
            keyword_soft_score(task.reference, response),
            "",
        )

    def reflect(self, *args, **kwargs):
        raise AssertionError("community imports must not invoke reflection")


class TestCommunityManifest(unittest.TestCase):
    def test_manifest_rejects_transcript_fields(self):
        payload = _manifest("Use a stable formatter.").to_dict()
        payload["transcripts"] = ["private session"]
        with self.assertRaisesRegex(CommunityRuleError, "unknown fields: transcripts"):
            parse_rule_manifest(payload)

    def test_manifest_rejects_inconsistent_effect_delta(self):
        payload = _manifest("Use a stable formatter.").to_dict()
        payload["rules"][0]["observed_effect"]["delta"] = 99
        with self.assertRaisesRegex(CommunityRuleError, "delta must equal"):
            parse_rule_manifest(payload)

    def test_export_contains_only_accepted_skill_additions(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = {
                "accepted": True,
                "baseline_score": 0.25,
                "candidate_score": 0.75,
                "n_tasks": 12,
                "edits": [
                    {
                        "target": "skill",
                        "op": "add",
                        "content": "Verify generated files before reporting success.",
                        "anchor": "",
                        "rationale": "Prevents false completion reports.",
                    },
                    {
                        "target": "memory",
                        "op": "add",
                        "content": "Private preference",
                        "anchor": "",
                        "rationale": "Local only.",
                    },
                ],
            }
            with open(os.path.join(tmp, "report.json"), "w", encoding="utf-8") as handle:
                json.dump(report, handle)
            output = os.path.join(tmp, "community-rules.json")

            _, manifest = export_rules_from_staging(
                tmp,
                output,
                category="coding",
                license_id="MIT",
            )
            with open(output, encoding="utf-8") as handle:
                raw = json.load(handle)

        self.assertEqual(len(manifest.rules), 1)
        self.assertEqual(raw["rules"][0]["observed_effect"]["scope"], "candidate_set")
        self.assertNotIn("Private preference", json.dumps(raw))
        self.assertNotIn("transcript", json.dumps(raw).lower())


class TestCommunityGate(unittest.TestCase):
    def test_rule_must_improve_local_tasks(self):
        task = TaskRecord(
            id="commit",
            project="/repo",
            intent="write a commit subject",
            reference_kind="exact",
            reference="feat: add parser",
            tags=["rule:commit-imperative"],
        )
        rule = MockBackend.RULE_TEXT["commit-imperative"]

        result = gate_community_rules(MockBackend(), [task], "# Skill\n", "", _manifest(rule))

        self.assertTrue(result.accepted)
        self.assertGreater(result.candidate_score, result.baseline_score)
        self.assertIn(rule, result.new_skill)
        self.assertEqual(len(result.accepted_edits), 1)

    def test_publisher_effect_does_not_override_local_rejection(self):
        task = TaskRecord(
            id="commit",
            project="/repo",
            intent="write a commit subject",
            reference_kind="exact",
            reference="feat: add parser",
            tags=["rule:commit-imperative"],
        )

        result = gate_community_rules(
            MockBackend(),
            [task],
            "# Skill\n",
            "",
            _manifest("An unrelated rule with a claimed large effect."),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(len(result.rejected_edits), 1)
        self.assertEqual(result.trials[0]["reason"], "no_strict_lift_or_regression")

    def test_any_local_regression_blocks_aggregate_lift(self):
        tasks = [
            TaskRecord(id="improve-1", project="/repo", intent="one", reference="answer one"),
            TaskRecord(id="improve-2", project="/repo", intent="two", reference="answer two"),
            TaskRecord(id="regress", project="/repo", intent="three", reference="answer three"),
        ]

        result = gate_community_rules(
            _RegressionBackend(),
            tasks,
            "# Skill\n",
            "",
            _manifest("Use the community strategy."),
            gate_metric="hard",
        )

        self.assertFalse(result.accepted)
        self.assertGreater(result.trials[0]["candidate_score"], result.baseline_score)
        self.assertTrue(any(
            row["status"] == "regressed" for row in result.trials[0]["task_deltas"]
        ))


class TestCommunityCli(unittest.TestCase):
    def test_import_requires_explicit_rule_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = write_rule_manifest(
                os.path.join(tmp, "rules.json"),
                _manifest("Use a stable formatter."),
            )
            out = StringIO()
            with redirect_stdout(out):
                rc = main(["import-rules", "--manifest", manifest_path, "--json"])

        payload = json.loads(out.getvalue())
        self.assertEqual(rc, 2)
        self.assertTrue(payload["review_required"])
        self.assertEqual(payload["rules"][0]["id"], "rule-example")

    def test_reviewed_import_stages_only_a_locally_accepted_rule(self):
        with tempfile.TemporaryDirectory() as project:
            target = os.path.join(project, ".agents", "skills", "demo", "SKILL.md")
            os.makedirs(os.path.dirname(target))
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("# Demo\n")
            rule = MockBackend.RULE_TEXT["commit-imperative"]
            manifest_path = write_rule_manifest(
                os.path.join(project, "rules.json"), _manifest(rule)
            )
            tasks_path = os.path.join(project, "tasks.json")
            tasks_payload = make_tasks_payload(
                [TaskRecord(
                    id="commit",
                    project=project,
                    intent="write a commit subject",
                    reference_kind="exact",
                    reference="feat: add parser",
                    tags=["rule:commit-imperative"],
                    split="val",
                )],
                project=project,
                target_skill_path=target,
            )
            tasks_payload["reviewed"] = True
            write_tasks_file(tasks_path, tasks_payload)
            out = StringIO()
            with redirect_stdout(out):
                rc = main([
                    "import-rules",
                    "--manifest", manifest_path,
                    "--reviewed",
                    "--project", project,
                    "--backend", "mock",
                    "--tasks-file", tasks_path,
                    "--json",
                ])
            payload = json.loads(out.getvalue())
            with open(
                os.path.join(payload["staging_dir"], "proposed_SKILL.md"),
                encoding="utf-8",
            ) as handle:
                proposed = handle.read()

        self.assertEqual(rc, 0)
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["accepted_rules"], 1)
        self.assertIn(rule, proposed)

    def test_import_gate_uses_val_split_only(self):
        with tempfile.TemporaryDirectory() as project:
            target = os.path.join(project, ".agents", "skills", "demo", "SKILL.md")
            os.makedirs(os.path.dirname(target))
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("# Demo\n")
            manifest_path = write_rule_manifest(
                os.path.join(project, "rules.json"),
                _manifest(MockBackend.RULE_TEXT["commit-imperative"]),
            )
            tasks_path = os.path.join(project, "tasks.json")
            tasks_payload = make_tasks_payload(
                [
                    TaskRecord(
                        id="train-commit",
                        project=project,
                        intent="write a commit subject",
                        reference_kind="exact",
                        reference="feat: add parser",
                        tags=["rule:commit-imperative"],
                        split="train",
                    ),
                    TaskRecord(
                        id="val-units",
                        project=project,
                        intent="report a measurement",
                        reference_kind="exact",
                        reference="10 kg",
                        tags=["rule:units-si"],
                        split="val",
                    ),
                    TaskRecord(
                        id="test-json",
                        project=project,
                        intent="return JSON",
                        reference_kind="exact",
                        reference='{"ok": true}',
                        tags=["rule:json-only"],
                        split="test",
                    ),
                ],
                project=project,
                target_skill_path=target,
            )
            tasks_payload["reviewed"] = True
            write_tasks_file(tasks_path, tasks_payload)
            out = StringIO()
            with redirect_stdout(out):
                rc = main([
                    "import-rules",
                    "--manifest", manifest_path,
                    "--reviewed",
                    "--project", project,
                    "--backend", "mock",
                    "--tasks-file", tasks_path,
                    "--json",
                ])
            payload = json.loads(out.getvalue())

        self.assertEqual(rc, 0)
        self.assertFalse(payload["accepted"])
        self.assertEqual(payload["trials"][0]["task_deltas"][0]["task_id"], "val-units")


if __name__ == "__main__":
    unittest.main()

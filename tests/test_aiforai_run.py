from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from skillopt_sleep.aiforai.cli import main
from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.harvesters import Harvester
from skillopt_sleep.aiforai.mine import mine_tasks
from skillopt_sleep.aiforai.regression_suite import curated_regression_tasks
from skillopt_sleep.aiforai.replay import (
    AiforaiScoreSummary,
    evaluate_tasks,
    gate_candidate,
    propose_mock_rules,
)
from skillopt_sleep.aiforai.run import run_audit
from skillopt_sleep.aiforai.types import AiforaiSessionDigest, AiforaiTaskRecord


def _session(
    source: str,
    session_id: str,
    prompt: str,
    final: str = "",
) -> AiforaiSessionDigest:
    return AiforaiSessionDigest(
        source_agent=source,  # type: ignore[arg-type]
        session_id=session_id,
        raw_path=f"/tmp/{session_id}.jsonl",
        cwd="/repo",
        user_prompts=[prompt],
        assistant_finals=[final] if final else [],
        skill_mentions=["ai-model-rd-protocol"],
    )


def _task(
    task_id: str,
    source: str,
    family: str,
    *required: str,
) -> AiforaiTaskRecord:
    return AiforaiTaskRecord(
        id=task_id,
        source_agent=source,
        source_sessions=[f"{task_id}-session"],
        project="AIForAI",
        intent=f"{family} intent",
        context_excerpt=f"{family} context",
        task_family=family,
        outcome="unknown",
        split="val",
        reference_kind="rule",
        judge={
            "kind": "rule",
            "checks": [{"op": "contains", "arg": item} for item in required],
        },
        origin="curated",
    )


class StaticHarvester(Harvester):
    def __init__(self, source_agent: str, sessions: list[AiforaiSessionDigest]) -> None:
        self.source_agent = source_agent
        self._sessions = sessions

    def harvest(self, cfg: AiforaiConfig) -> list[AiforaiSessionDigest]:
        return list(self._sessions)


class FailingHarvester(Harvester):
    def __init__(self, source_agent: str, message: str) -> None:
        self.source_agent = source_agent
        self._message = message

    def harvest(self, cfg: AiforaiConfig) -> list[AiforaiSessionDigest]:
        raise RuntimeError(self._message)


class AiforaiRunTests(unittest.TestCase):
    def _assert_cli_error(self, argv: list[str], expected_text: str) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                main(argv)
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn(expected_text, stderr.getvalue())

    def test_cli_rejects_unsupported_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._assert_cli_error(
                [
                    "audit",
                    "--target-skill-repo",
                    tmp,
                    "--sources",
                    "codex,deepseek",
                ],
                "unsupported source",
            )

    def test_cli_rejects_empty_source_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._assert_cli_error(
                [
                    "audit",
                    "--target-skill-repo",
                    tmp,
                    "--sources",
                    ", ,",
                ],
                "at least one source",
            )

    def test_cli_rejects_nonpositive_numeric_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._assert_cli_error(
                [
                    "audit",
                    "--target-skill-repo",
                    tmp,
                    "--lookback-days",
                    "0",
                ],
                "lookback-days must be > 0",
            )
            self._assert_cli_error(
                [
                    "audit",
                    "--target-skill-repo",
                    tmp,
                    "--max-tasks-per-source",
                    "-1",
                ],
                "max-tasks-per-source must be > 0",
            )

    def test_run_audit_stages_report_and_manifest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_repo = Path(tmp) / "AIForAI"
            target_repo.mkdir()
            cfg = AiforaiConfig(
                target_skill_repo=str(target_repo),
                sources=("codex", "claude"),
                max_tasks_per_source=5,
                val_fraction=0.5,
                test_fraction=0.0,
                seed=7,
            )
            harvesters = [
                StaticHarvester(
                    "codex",
                    [
                        _session(
                            "codex",
                            "codex-1",
                            "start a training run",
                            "Need a training contract before launch.",
                        )
                    ],
                ),
                StaticHarvester(
                    "claude",
                    [
                        _session(
                            "claude",
                            "claude-1",
                            "download full dataset locally",
                            "Use shared storage instead of local download.",
                        ),
                        _session("claude", "claude-2", "write a poem"),
                    ],
                ),
            ]

            result = run_audit(cfg, harvesters=harvesters)

            self.assertEqual(result.mode, "audit")
            self.assertEqual(result.sessions_by_source["codex"], 1)
            self.assertEqual(result.checkable_tasks, 2)
            self.assertEqual(result.uncheckable_candidates, 1)

            out_dir = Path(result.staging_dir)
            self.assertTrue((out_dir / "audit_report.md").exists())
            self.assertTrue((out_dir / "report.json").exists())
            self.assertTrue((out_dir / "sessions.jsonl").exists())
            self.assertTrue((out_dir / "task_manifest.jsonl").exists())
            self.assertTrue((out_dir / "uncheckable_candidates.jsonl").exists())

            report_json = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report_json["result"]["mode"], "audit")
            self.assertEqual(report_json["source_coverage"]["sessions_by_source"]["codex"], 1)

            session_rows = [
                json.loads(line)
                for line in (out_dir / "sessions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(session_rows), 3)
            self.assertEqual(session_rows[0]["session_id"], "codex-1")

            manifest_rows = [
                json.loads(line)
                for line in (out_dir / "task_manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(manifest_rows), 2)
            self.assertIn(manifest_rows[0]["source_agent"], {"codex", "claude"})

            report_text = (out_dir / "audit_report.md").read_text(encoding="utf-8")
            self.assertIn("Source Coverage", report_text)
            self.assertIn("Task Families", report_text)
            self.assertIn("Checkability", report_text)
            self.assertIn("Audit boundary: read-only.", report_text)
            self.assertIn("No live AIForAI skill files were modified.", report_text)

    def test_run_audit_raises_when_no_supported_harvesters_are_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_repo = Path(tmp) / "AIForAI"
            target_repo.mkdir()
            cfg = AiforaiConfig(
                target_skill_repo=str(target_repo),
                sources=("deepseek",),
            )

            with self.assertRaisesRegex(ValueError, "No supported AIForAI harvesters selected"):
                run_audit(cfg)

            self.assertFalse(Path(cfg.staging_root).exists())

    def test_run_audit_raises_when_all_selected_harvesters_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_repo = Path(tmp) / "AIForAI"
            target_repo.mkdir()
            cfg = AiforaiConfig(
                target_skill_repo=str(target_repo),
                sources=("codex", "claude"),
            )

            with self.assertRaisesRegex(ValueError, "No AIForAI sessions were harvested"):
                run_audit(
                    cfg,
                    harvesters=[
                        FailingHarvester("codex", "codex boom"),
                        FailingHarvester("claude", "claude boom"),
                    ],
                )

            self.assertFalse(Path(cfg.staging_root).exists())

    def test_run_audit_raises_when_selected_harvesters_return_no_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_repo = Path(tmp) / "AIForAI"
            target_repo.mkdir()
            cfg = AiforaiConfig(
                target_skill_repo=str(target_repo),
                sources=("codex",),
            )

            with self.assertRaisesRegex(ValueError, "No AIForAI sessions were harvested"):
                run_audit(cfg, harvesters=[StaticHarvester("codex", [])])

            self.assertFalse(Path(cfg.staging_root).exists())

    def test_run_audit_counts_sources_from_session_digests_and_notes_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_repo = Path(tmp) / "AIForAI"
            target_repo.mkdir()
            cfg = AiforaiConfig(
                target_skill_repo=str(target_repo),
                sources=("codex",),
                max_tasks_per_source=5,
            )

            result = run_audit(
                cfg,
                harvesters=[
                    StaticHarvester(
                        "codex",
                        [
                            _session(
                                "claude",
                                "mismatch-1",
                                "start a training run",
                                "Need a training contract before launch.",
                            )
                        ],
                    )
                ],
            )

            self.assertEqual(result.sessions_by_source["codex"], 0)
            self.assertEqual(result.sessions_by_source["claude"], 1)
            self.assertEqual(result.tasks_by_source["codex"], 0)
            self.assertEqual(result.tasks_by_source["claude"], 1)
            self.assertTrue(any("mismatch" in note for note in result.notes))


class AiforaiReplayGateTests(unittest.TestCase):
    def test_curated_regression_suite_has_required_families(self) -> None:
        tasks = curated_regression_tasks()
        families = {task.task_family for task in tasks}
        self.assertEqual(
            families,
            {
                "training_contract",
                "data_acquisition",
                "dirty_worktree_gate",
                "claim_integrity",
                "rag_agent_diagnosis",
                "cluster_preflight",
            },
        )

    def test_mock_replay_improves_after_rule_added(self) -> None:
        tasks, _ = mine_tasks([
            AiforaiSessionDigest(
                source_agent="codex",
                session_id="s1",
                raw_path="/tmp/raw",
                cwd="/repo",
                user_prompts=["start a training run"],
            )
        ])

        baseline = evaluate_tasks(tasks, "")
        rules = propose_mock_rules(tasks, "")
        candidate = "\n".join(rules)
        candidate_score = evaluate_tasks(tasks, candidate)

        self.assertLess(baseline.aggregate_hard, candidate_score.aggregate_hard)
        self.assertTrue(gate_candidate(baseline, candidate_score).accepted)

    def test_boundary_sensitive_contains_keeps_missing_requirements_and_rules(self) -> None:
        tasks = [
            _task("dirty-1", "codex", "dirty_worktree_gate", "formal"),
            _task("rag-1", "claude", "rag_agent_diagnosis", "tool"),
        ]

        score = evaluate_tasks(tasks, "Use informal notes and tooling summaries.")
        rules = propose_mock_rules(tasks, "Use informal notes and tooling summaries.")

        self.assertEqual(score.aggregate_hard, 0.0)
        self.assertIn("missing: formal", score.results[0].fail_reason)
        self.assertIn("missing: tool", score.results[1].fail_reason)
        self.assertEqual(
            rules,
            [
                "For dirty_worktree_gate tasks, explicitly include: formal.",
                "For rag_agent_diagnosis tasks, explicitly include: tool.",
            ],
        )

    def test_replay_summaries_report_expected_source_and_family_means(self) -> None:
        tasks = [
            _task("codex-train", "codex", "training_contract", "training contract"),
            _task("codex-rag", "codex", "rag_agent_diagnosis", "trajectory"),
            _task("claude-train", "claude", "training_contract", "evaluation contract"),
            _task("claude-cluster", "claude", "cluster_preflight", "artifact"),
        ]

        score = evaluate_tasks(
            tasks,
            "training contract\nevaluation contract\nartifact",
        )

        self.assertEqual(score.aggregate_hard, 0.75)
        self.assertEqual(score.by_source, {"codex": 0.5, "claude": 1.0})
        self.assertEqual(
            score.by_family,
            {
                "training_contract": 1.0,
                "rag_agent_diagnosis": 0.0,
                "cluster_preflight": 1.0,
            },
        )

    def test_propose_mock_rules_merges_same_family_missing_requirements(self) -> None:
        tasks = [
            _task(
                "train-1",
                "codex",
                "training_contract",
                "training contract",
                "evaluation contract",
            ),
            _task(
                "train-2",
                "claude",
                "training_contract",
                "evaluation contract",
                "artifact paths",
            ),
        ]

        rules = propose_mock_rules(tasks, "Keep a training contract in the plan.")

        self.assertEqual(
            rules,
            [
                "For training_contract tasks, explicitly include: artifact paths, evaluation contract."
            ],
        )

    def test_propose_mock_rules_dedupes_overlapping_family_requirements(self) -> None:
        tasks = [
            _task("train-1", "codex", "training_contract", "contract"),
            _task("train-2", "claude", "training_contract", "training contract"),
        ]

        rules = propose_mock_rules(tasks, "")

        self.assertEqual(
            rules,
            [
                "For training_contract tasks, explicitly include: training contract."
            ],
        )

    def test_propose_mock_rules_preserves_boundary_distinct_requirements(self) -> None:
        tasks = [
            _task("rag-1", "codex", "rag_agent_diagnosis", "tool"),
            _task("rag-2", "claude", "rag_agent_diagnosis", "tooling"),
        ]

        rules = propose_mock_rules(tasks, "")

        self.assertEqual(
            rules,
            [
                "For rag_agent_diagnosis tasks, explicitly include: tool, tooling."
            ],
        )

    def test_gate_rejects_tie_without_aggregate_improvement(self) -> None:
        baseline = AiforaiScoreSummary(
            aggregate_hard=0.5,
            aggregate_soft=0.5,
            by_source={"codex": 0.5},
            by_family={"training_contract": 0.5},
        )
        candidate = AiforaiScoreSummary(
            aggregate_hard=0.5,
            aggregate_soft=0.8,
            by_source={"codex": 0.5},
            by_family={"training_contract": 0.5},
        )

        decision = gate_candidate(baseline, candidate)

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.action, "reject")
        self.assertIn("did not improve", decision.reason)

    def test_gate_rejects_source_slice_regression_despite_aggregate_gain(self) -> None:
        baseline = AiforaiScoreSummary(
            aggregate_hard=0.4,
            aggregate_soft=0.4,
            by_source={"codex": 0.2, "claude": 0.6},
            by_family={"training_contract": 0.4},
        )
        candidate = AiforaiScoreSummary(
            aggregate_hard=0.6,
            aggregate_soft=0.6,
            by_source={"codex": 0.8, "claude": 0.4},
            by_family={"training_contract": 0.6},
        )

        decision = gate_candidate(baseline, candidate)

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.action, "reject")
        self.assertIn("source", decision.reason)
        self.assertIn("claude", decision.reason)

    def test_gate_rejects_family_slice_regression_despite_aggregate_gain(self) -> None:
        baseline = AiforaiScoreSummary(
            aggregate_hard=0.4,
            aggregate_soft=0.4,
            by_source={"codex": 0.4},
            by_family={"training_contract": 0.6, "cluster_preflight": 0.2},
        )
        candidate = AiforaiScoreSummary(
            aggregate_hard=0.6,
            aggregate_soft=0.6,
            by_source={"codex": 0.6},
            by_family={"training_contract": 0.4, "cluster_preflight": 0.8},
        )

        decision = gate_candidate(baseline, candidate)

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.action, "reject")
        self.assertIn("family", decision.reason)
        self.assertIn("training_contract", decision.reason)


if __name__ == "__main__":
    unittest.main()

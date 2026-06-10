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
from skillopt_sleep.aiforai.run import run_audit
from skillopt_sleep.aiforai.types import AiforaiSessionDigest


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


if __name__ == "__main__":
    unittest.main()

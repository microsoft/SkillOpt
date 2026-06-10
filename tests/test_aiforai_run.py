from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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


class AiforaiRunTests(unittest.TestCase):
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
            self.assertTrue((out_dir / "task_manifest.jsonl").exists())
            self.assertTrue((out_dir / "uncheckable_candidates.jsonl").exists())

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


if __name__ == "__main__":
    unittest.main()

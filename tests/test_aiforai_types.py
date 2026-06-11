from __future__ import annotations

import unittest

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.types import (
    AiforaiRunResult,
    AiforaiSessionDigest,
    AiforaiTaskRecord,
)


class AiforaiTypesTests(unittest.TestCase):
    def test_session_round_trip_preserves_source(self) -> None:
        session = AiforaiSessionDigest(
            source_agent="codex",
            session_id="s1",
            raw_path="/tmp/session.jsonl",
            cwd="/repo",
            user_prompts=["train a model"],
            assistant_finals=["done"],
            tools_used=["exec"],
            skill_mentions=["ai-model-rd-protocol"],
            parse_warnings=["ignored record"],
        )

        restored = AiforaiSessionDigest.from_dict(session.to_dict())

        self.assertEqual(restored.source_agent, "codex")
        self.assertEqual(restored.session_id, "s1")
        self.assertEqual(restored.user_prompts, ["train a model"])
        self.assertEqual(restored.parse_warnings, ["ignored record"])

    def test_task_round_trip_preserves_judge(self) -> None:
        task = AiforaiTaskRecord(
            id="t1",
            source_agent="claude",
            source_sessions=["s1"],
            project="/repo",
            intent="prepare training contract",
            context_excerpt="ctx",
            task_family="training_contract",
            outcome="fail",
            split="val",
            judge={
                "kind": "rule",
                "checks": [{"op": "contains", "arg": "training contract"}],
            },
        )

        restored = AiforaiTaskRecord.from_dict(task.to_dict())

        self.assertEqual(restored.source_agent, "claude")
        self.assertEqual(restored.split, "val")
        self.assertEqual(restored.judge["checks"][0]["arg"], "training contract")

    def test_config_defaults_include_three_sources(self) -> None:
        cfg = AiforaiConfig(target_skill_repo="/tmp/AIForAI")

        self.assertEqual(cfg.sources, ("codex", "claude", "codewhale"))
        self.assertEqual(cfg.skill_rel_path, "ai-model-rd-protocol/SKILL.md")
        self.assertFalse(cfg.auto_adopt)

    def test_run_result_serializes_counts(self) -> None:
        result = AiforaiRunResult(
            mode="audit",
            staging_dir="/tmp/staging",
            sessions_by_source={"codex": 2, "claude": 1, "codewhale": 3},
            tasks_by_source={"codex": 1},
            accepted=False,
            notes=["audit only"],
        )

        data = result.to_dict()

        self.assertEqual(data["mode"], "audit")
        self.assertEqual(data["sessions_by_source"]["codewhale"], 3)
        self.assertEqual(data["notes"], ["audit only"])


if __name__ == "__main__":
    unittest.main()

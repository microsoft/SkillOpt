from __future__ import annotations

import unittest

from skillopt_sleep.aiforai.mine import mine_tasks, split_tasks
from skillopt_sleep.aiforai.types import AiforaiSessionDigest, AiforaiTaskRecord


def _session(source: str, prompt: str, final: str = "") -> AiforaiSessionDigest:
    return AiforaiSessionDigest(
        source_agent=source,  # type: ignore[arg-type]
        session_id=f"{source}-1",
        raw_path=f"/tmp/{source}.jsonl",
        cwd="/repo",
        user_prompts=[prompt],
        assistant_finals=[final],
        skill_mentions=["ai-model-rd-protocol"],
    )


class AiforaiMineTests(unittest.TestCase):
    def test_mines_training_contract_task(self) -> None:
        tasks, uncheckable = mine_tasks([
            _session("codex", "Start a training run without a contract", "Need a training contract")
        ])

        self.assertEqual(uncheckable, [])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_family, "training_contract")
        self.assertEqual(tasks[0].judge["checks"][0]["op"], "contains")

    def test_mines_data_acquisition_task(self) -> None:
        tasks, _ = mine_tasks([
            _session("claude", "Download the full dataset locally before training")
        ])

        self.assertEqual(tasks[0].task_family, "data_acquisition")

    def test_unrelated_prompt_is_uncheckable(self) -> None:
        tasks, uncheckable = mine_tasks([
            _session("codewhale", "Write a poem about rain")
        ])

        self.assertEqual(tasks, [])
        self.assertEqual(len(uncheckable), 1)

    def test_split_tasks_keeps_sources_represented(self) -> None:
        tasks = [
            AiforaiTaskRecord(
                id=f"{source}-{idx}",
                source_agent=source,
                source_sessions=[f"{source}-{idx}"],
                project="/repo",
                intent="prepare training contract",
                context_excerpt="",
                task_family="training_contract",
                outcome="fail",
                judge={"kind": "rule", "checks": [{"op": "contains", "arg": "training contract"}]},
            )
            for source in ("codex", "claude", "codewhale")
            for idx in range(6)
        ]

        split = split_tasks(tasks, val_fraction=0.25, test_fraction=0.0, seed=7)
        val_sources = {task.source_agent for task in split if task.split == "val"}

        self.assertEqual(val_sources, {"codex", "claude", "codewhale"})
        self.assertTrue(any(task.split == "train" for task in split))


if __name__ == "__main__":
    unittest.main()

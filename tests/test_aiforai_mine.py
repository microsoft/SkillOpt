from __future__ import annotations

import unittest

from skillopt_sleep.aiforai.mine import mine_tasks, split_tasks
from skillopt_sleep.aiforai.types import AiforaiSessionDigest, AiforaiTaskRecord


def _session(
    source: str,
    prompt: str,
    final: str = "",
    *,
    session_id: str | None = None,
    feedback_signals: list[str] | None = None,
) -> AiforaiSessionDigest:
    return AiforaiSessionDigest(
        source_agent=source,  # type: ignore[arg-type]
        session_id=session_id or f"{source}-1",
        raw_path=f"/tmp/{source}.jsonl",
        cwd="/repo",
        user_prompts=[prompt],
        assistant_finals=[final],
        feedback_signals=feedback_signals or [],
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

    def test_ambiguous_prompt_is_uncheckable(self) -> None:
        tasks, uncheckable = mine_tasks([
            _session("codex", "Train the model and download the dataset locally")
        ])

        self.assertEqual(tasks, [])
        self.assertEqual(len(uncheckable), 1)
        self.assertEqual(
            uncheckable[0]["reason"],
            "no_checkable_aiforai_family",
        )

    def test_duplicate_sessions_do_not_consume_source_cap_before_dedup(self) -> None:
        tasks, uncheckable = mine_tasks(
            [
                _session(
                    "codex",
                    "Start a training run without a contract",
                    session_id="codex-1",
                ),
                _session(
                    "codex",
                    "Start a training run without a contract",
                    session_id="codex-2",
                ),
                _session(
                    "codex",
                    "Download the full dataset locally before training",
                    session_id="codex-3",
                ),
            ],
            max_tasks_per_source=2,
        )

        self.assertEqual(uncheckable, [])
        self.assertEqual(len(tasks), 2)
        self.assertEqual(
            {task.task_family for task in tasks},
            {"training_contract", "data_acquisition"},
        )
        training_task = next(task for task in tasks if task.task_family == "training_contract")
        self.assertEqual(training_task.source_sessions, ["codex-1", "codex-2"])

    def test_duplicate_conflicting_feedback_merges_outcome_to_mixed(self) -> None:
        tasks, uncheckable = mine_tasks(
            [
                _session(
                    "claude",
                    "Start a training run without a contract",
                    session_id="claude-1",
                    feedback_signals=["pos:looks good"],
                ),
                _session(
                    "claude",
                    "Start a training run without a contract",
                    session_id="claude-2",
                    feedback_signals=["neg:still broken"],
                ),
            ]
        )

        self.assertEqual(uncheckable, [])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].outcome, "mixed")
        self.assertEqual(tasks[0].source_sessions, ["claude-1", "claude-2"])

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

    def test_split_tasks_allocates_test_when_fraction_is_nonzero_and_possible(self) -> None:
        tasks = [
            AiforaiTaskRecord(
                id=f"codex-{idx}",
                source_agent="codex",
                source_sessions=[f"codex-{idx}"],
                project="/repo",
                intent="prepare training contract",
                context_excerpt="",
                task_family="training_contract",
                outcome="fail",
                judge={"kind": "rule", "checks": [{"op": "contains", "arg": "training contract"}]},
            )
            for idx in range(4)
        ]

        split = split_tasks(tasks, val_fraction=0.25, test_fraction=0.1, seed=11)

        self.assertTrue(any(task.split == "test" for task in split))
        self.assertTrue(any(task.split == "train" for task in split))


if __name__ == "__main__":
    unittest.main()

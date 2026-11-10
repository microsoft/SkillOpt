"""Tests for judge-side observability in replay, consolidate, and cycle reports (#283).

Pure-stdlib (unittest), deterministic, no API key, no external dependencies.
"""
from __future__ import annotations

import tempfile
import time
import unittest
from typing import Tuple

from skillopt_sleep.backend import DualBackend, MockBackend
from skillopt_sleep.config import load_config
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.cycle import _render_report_md, run_sleep_cycle
from skillopt_sleep.replay import replay_one
from skillopt_sleep.types import SleepReport, TaskRecord


class InstrumentedJudgeBackend(MockBackend):
    """Mock backend that tracks judge token counts and latency."""

    def __init__(self, model_name: str = "mock-judge-v1"):
        super().__init__()
        self.name = model_name
        self._tokens = 0

    def tokens_used(self) -> int:
        return self._tokens

    @property
    def judge_model_name(self) -> str:
        return self.name

    def judge(self, task: TaskRecord, response: str) -> Tuple[float, float, str]:
        # Simulate LLM judge token usage and small latency
        self._tokens += 42
        time.sleep(0.005)
        return super().judge(task, response)


class TestJudgeObservability(unittest.TestCase):
    def test_replay_one_rule_judge_records_rule_model_and_zero_tokens(self):
        backend = MockBackend()
        task = TaskRecord(
            id="t1",
            project="test",
            intent="write summary",
            reference_kind="rule",
            judge={"operator": "section_present", "target": "Summary"},
        )
        res = replay_one(backend, task, skill="# Summary\nhello", memory="")
        self.assertEqual(res.judge_model, "rule")
        self.assertEqual(res.judge_tokens, 0)
        self.assertGreaterEqual(res.judge_latency_ms, 0.0)

    def test_replay_one_rubric_judge_measures_latency_and_tokens(self):
        backend = InstrumentedJudgeBackend(model_name="rubric-eval-llm")
        task = TaskRecord(
            id="t2",
            project="test",
            intent="implement feature",
            reference_kind="rubric",
            reference="clean architecture",
        )
        res = replay_one(backend, task, skill="", memory="")
        self.assertEqual(res.judge_model, "rubric-eval-llm")
        self.assertEqual(res.judge_tokens, 42)
        self.assertGreater(res.judge_latency_ms, 0.0)

    def test_dual_backend_judge_model_attribution(self):
        target = MockBackend()
        target.name = "llama-3-8b"
        optimizer = MockBackend()
        optimizer.name = "gpt-4o-judge"

        dual = DualBackend(target, optimizer)
        self.assertEqual(dual.judge_model_name, "gpt-4o-judge")

    def test_consolidate_aggregates_judge_metrics(self):
        backend = InstrumentedJudgeBackend(model_name="judge-agg-model")
        tasks = [
            TaskRecord(
                id="t1",
                project="test",
                intent="p1",
                reference_kind="rubric",
                reference="ref1",
                split="val",
            ),
            TaskRecord(
                id="t2",
                project="test",
                intent="p2",
                reference_kind="rubric",
                reference="ref2",
                split="val",
            ),
        ]
        result = consolidate(
            backend,
            tasks,
            skill="",
            memory="",
            gate_mode="on",
        )
        self.assertGreaterEqual(result.judge_requests, 2)
        self.assertGreaterEqual(result.judge_tokens, 84)
        self.assertGreater(result.judge_latency_ms, 0.0)
        self.assertEqual(result.judge_model, "judge-agg-model")

    def test_render_report_md_includes_judge_observability(self):
        report = SleepReport(
            night=1,
            project="test",
            tokens_used=1500,
            judge_requests=4,
            judge_latency_ms=312.5,
            judge_tokens=850,
            judge_model="gpt-4o",
        )
        rendered = _render_report_md(report, {})
        self.assertIn("- judge: 4 request(s), 312.5ms, 850 tokens (model: `gpt-4o`)", rendered)

    def test_run_sleep_cycle_records_judge_metrics(self):
        backend = InstrumentedJudgeBackend(model_name="cycle-judge-model")
        tasks = [
            TaskRecord(
                id="t1",
                project="test",
                intent="prompt 1",
                reference_kind="rubric",
                reference="expected 1",
                split="train",
            ),
            TaskRecord(
                id="t2",
                project="test",
                intent="prompt 2",
                reference_kind="rubric",
                reference="expected 2",
                split="test",
            ),
        ]
        with tempfile.TemporaryDirectory() as td:
            cfg = load_config(project=td, gate_mode="off")
            outcome = run_sleep_cycle(
                cfg,
                seed_tasks=tasks,
                backend=backend,
                dry_run=True,
            )
            report = outcome.report
            self.assertGreaterEqual(report.judge_requests, 1)
            self.assertGreater(report.judge_latency_ms, 0.0)
            self.assertGreater(report.judge_tokens, 0)
            self.assertEqual(report.judge_model, "cycle-judge-model")


if __name__ == "__main__":
    unittest.main()

"""max_tokens_per_night stops a night from scheduling more work (issue #313).

Pure-stdlib (unittest), deterministic, no API key, no network.
Run:  python -m pytest tests/test_sleep_token_budget.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

from skillopt_sleep.backend import ClaudeCliBackend, MockBackend
from skillopt_sleep.config import load_config
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.evidence import read_events
from skillopt_sleep.experiments.personas import programmer_persona, researcher_persona
from skillopt_sleep.handoff_backend import HandoffBackend, PendingCalls
from skillopt_sleep.mine import assign_splits
from skillopt_sleep.multi_skill import (
    CONSOLIDATED,
    SKIPPED,
    SkillGroup,
    consolidate_groups,
)
from skillopt_sleep.types import TaskRecord

COST = 100  # tokens billed per uncached model call
_UNSET = object()


class MeteredBackend(MockBackend):
    """MockBackend that reports token usage like the CLI backends do.

    Every uncached attempt and every reflect call bills ``COST`` tokens. As in
    ``CliBackend``, re-running the same task with the same skill and memory is
    served from cache and costs nothing, and the mock judge is local.
    """

    name = "metered"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.attempt_skills: list[str] = []
        self._seen: set = set()
        self._tokens = 0

    def _bill(self, kind: str) -> None:
        self.calls.append(kind)
        self._tokens += COST

    def attempt(self, task, skill, memory, sample_id=0):
        key = (task.id, skill, memory, sample_id)
        if key not in self._seen:
            self._seen.add(key)
            self.attempt_skills.append(skill)
            self._bill("attempt")
        return super().attempt(task, skill, memory, sample_id)

    def reflect(self, failures, successes, skill, memory, **kwargs):
        if failures:
            self._bill("reflect")
        return super().reflect(failures, successes, skill, memory, **kwargs)

    def tokens_used(self) -> int:
        return self._tokens


class CountedBeforeConsolidationBackend(MeteredBackend):
    """Usage already recorded before consolidation starts.

    Mirrors a backend that counts LLM mining usage: the counter is 0 when the
    night starts and over the default budget at every later reading.
    """

    def __init__(self) -> None:
        super().__init__()
        self._reads = 0

    def tokens_used(self) -> int:
        self._reads += 1
        return 0 if self._reads == 1 else 500_000


def _tasks(**kwargs):
    kwargs.setdefault("val_fraction", 0.34)
    return assign_splits(researcher_persona(), seed=42, **kwargs)


def _replay_cost(tasks) -> int:
    """Tokens for the baseline val replay plus the train replay."""
    return COST * sum(1 for t in tasks if t.split in ("train", "val"))


def _budget_notes(report) -> list:
    return [n for n in report.notes if "max_tokens_per_night" in n]


def _arithmetic_tasks():
    tasks = [
        TaskRecord(
            id=f"t{i}", project="/p",
            intent=f"What is {i} + {i}?",
            reference_kind="exact", reference=str(i + i),
        )
        for i in range(1, 7)
    ]
    return assign_splits(tasks, holdout_fraction=0.34, seed=42)


def _tool_tasks():
    """Tasks whose rule judge needs a real tool call (the tool replay route)."""
    tasks = [
        TaskRecord(
            id=f"look{i}", project="/p",
            intent=f"Find note {i} and summarize it.",
            reference_kind="rule",
            judge={"kind": "rule", "checks": [{"op": "tool_called", "arg": "search"}]},
        )
        for i in range(1, 7)
    ]
    return assign_splits(tasks, val_fraction=0.34, seed=42)


_REAL_RUN = subprocess.run
_SEARCH_RULE = "Always run ./search before answering."


class SyntheticClaudeProcess:
    """Stands in for the external ``claude`` process; no model is called.

    Records every process the backend spawns as ``(phase, route)``, where the
    route is ``tools`` for ``attempt_with_tools`` and ``text`` for ``_call``.
    Replies with a fixed answer, or raises ``TimeoutExpired`` like a process
    that ran past the backend's timeout. A reflect call (text route during the
    train phase) proposes ``_SEARCH_RULE``; a tool replay whose prompt carries
    that rule logs a call to the ``search`` shim, so its rule judge passes.
    """

    def __init__(self, backend: ClaudeCliBackend, *, timeout: bool = False) -> None:
        self.backend = backend
        self.timeout = timeout
        self.calls: list[tuple[str, str]] = []

    def __call__(self, cmd, **kwargs):
        if not cmd or cmd[0] != self.backend.claude_path:
            return _REAL_RUN(cmd, **kwargs)
        route = "tools" if "--allowedTools" in cmd else "text"
        phase = getattr(self.backend, "evidence_phase", "")
        self.calls.append((phase, route))
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        prompt = kwargs.get("input") or ""
        if route == "text" and phase == "train":
            reply = json.dumps([{"op": "add", "content": _SEARCH_RULE}])
        else:
            reply = "Here is the summary."
        if route == "tools" and _SEARCH_RULE in prompt:
            with open(os.path.join(kwargs["cwd"], "_tool_calls.log"), "a") as log:
                log.write("search\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=reply, stderr="")

    def phases(self) -> list[str]:
        return [phase for phase, _route in self.calls]


def _synthetic_claude(timeout: bool = False):
    backend = ClaudeCliBackend(claude_path="synthetic-claude-for-tests")
    process = SyntheticClaudeProcess(backend, timeout=timeout)
    return backend, process


class TokenBudgetCycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._proj = tempfile.TemporaryDirectory()
        self._home = tempfile.TemporaryDirectory()
        self.addCleanup(self._proj.cleanup)
        self.addCleanup(self._home.cleanup)

    def _cfg(self, budget=_UNSET, **overrides):
        cfg = load_config(
            invoked_project=self._proj.name,
            projects="invoked",
            backend="mock",
            claude_home=os.path.join(self._home.name, ".claude"),
            state_dir=os.path.join(self._home.name, "state"),
            auto_adopt=False,
            **overrides,
        )
        if budget is not _UNSET:
            # Set directly: load_config drops None overrides, but a config
            # file can still carry an explicit null.
            cfg.data["max_tokens_per_night"] = budget
        return cfg

    def _run(self, tasks, backend=None, budget=_UNSET, dry_run=False, **overrides):
        cfg = self._cfg(budget, **overrides)
        backend = backend or MeteredBackend()
        outcome = run_sleep_cycle(
            cfg, seed_tasks=tasks, backend=backend, dry_run=dry_run
        )
        return backend, outcome

    def _report_md(self, outcome) -> str:
        with open(
            os.path.join(outcome.staging_dir, "report.md"), encoding="utf-8"
        ) as handle:
            return handle.read()

    def _state_history(self) -> list:
        with open(self._cfg().state_path, encoding="utf-8") as handle:
            return json.load(handle)["history"]

    def test_night_stops_scheduling_work_once_the_budget_is_spent(self):
        tasks = _tasks()
        limit = _replay_cost(tasks)

        backend, outcome = self._run(tasks, budget=limit)

        self.assertLessEqual(backend.tokens_used(), limit)
        self.assertNotIn("reflect", backend.calls)
        report = outcome.report
        self.assertFalse(report.accepted)
        self.assertEqual(report.edits, [])
        notes = _budget_notes(report)
        self.assertEqual(len(notes), 1)
        self.assertIn(f"{limit} of {limit}", notes[0])
        self.assertIn("skill edits", notes[0])
        self.assertIn("memory edits", notes[0])
        self.assertIn("max\\_tokens\\_per\\_night", self._report_md(outcome))
        self.assertFalse(
            os.path.exists(os.path.join(outcome.staging_dir, "proposed_SKILL.md"))
        )
        events = read_events(os.path.join(outcome.staging_dir, "evidence.jsonl"))
        end = [e for e in events if (e["stage"], e["event"]) == ("cycle", "end")]
        self.assertEqual(end[-1]["max_tokens_per_night"], limit)
        self.assertEqual(
            end[-1]["budget_skipped"],
            ["skill edits", "memory edits", "final validation replay"],
        )

    def test_one_token_of_headroom_lets_the_next_step_run(self):
        tasks = _tasks()
        limit = _replay_cost(tasks) + 1

        backend, outcome = self._run(tasks, budget=limit)

        # The skill step started under budget and was allowed to finish, so
        # its validated edit is kept; the memory step was never started.
        self.assertEqual(backend.calls.count("reflect"), 1)
        report = outcome.report
        self.assertTrue(report.accepted)
        self.assertTrue(report.edits)
        notes = _budget_notes(report)
        self.assertEqual(len(notes), 1)
        self.assertNotIn("skill edits", notes[0])
        self.assertIn("memory edits", notes[0])
        self.assertTrue(
            os.path.exists(os.path.join(outcome.staging_dir, "proposed_SKILL.md"))
        )

    def test_dry_run_stops_on_the_budget_without_persisting(self):
        tasks = _tasks()
        limit = _replay_cost(tasks)

        backend, outcome = self._run(tasks, budget=limit, dry_run=True)

        self.assertNotIn("reflect", backend.calls)
        self.assertEqual(len(_budget_notes(outcome.report)), 1)
        self.assertEqual(outcome.staging_dir, "")
        self.assertFalse(os.path.exists(self._cfg().state_path))

    def test_spent_before_consolidation_reports_no_scores(self):
        backend = CountedBeforeConsolidationBackend()

        _, outcome = self._run(_tasks(), backend=backend)

        self.assertEqual(backend.calls, [])
        report = outcome.report
        self.assertEqual(report.gate_action, "budget_exhausted")
        self.assertFalse(report.accepted)
        self.assertEqual(report.n_replayed, 0)
        notes = _budget_notes(report)
        self.assertEqual(len(notes), 1)
        self.assertIn("skipped: consolidation", notes[0])
        held_out = [
            line for line in self._report_md(outcome).splitlines()
            if line.startswith("- held-out score:")
        ]
        self.assertEqual(len(held_out), 1)
        self.assertNotIn("0.000", held_out[0])
        (night,) = self._state_history()
        self.assertNotIn("baseline", night)
        self.assertNotIn("candidate", night)
        with open(
            os.path.join(outcome.staging_dir, "diagnostics.json"), encoding="utf-8"
        ) as handle:
            diagnostics = json.load(handle)
        self.assertIsNone(diagnostics["baseline_score"])
        self.assertIsNone(diagnostics["candidate_score"])

    def test_default_budget_leaves_an_ordinary_night_unchanged(self):
        backend, outcome = self._run(_tasks())
        self.assertGreater(backend.tokens_used(), _replay_cost(_tasks()))
        self.assertIn("reflect", backend.calls)
        self.assertTrue(outcome.report.accepted)
        self.assertFalse(_budget_notes(outcome.report))
        (night,) = self._state_history()
        self.assertIn("baseline", night)
        self.assertIn("candidate", night)

    def test_zero_and_null_mean_unlimited(self):
        reference, _ = self._run(_tasks())
        for value in (None, 0, "0", 0.0):
            with self.subTest(max_tokens_per_night=value):
                backend, outcome = self._run(_tasks(), budget=value)
                self.assertEqual(backend.calls, reference.calls)
                self.assertTrue(outcome.report.accepted)
                self.assertFalse(_budget_notes(outcome.report))

    def test_budget_counts_only_tonight_spend_on_a_reused_backend(self):
        tasks = _tasks()
        backend = MeteredBackend()
        backend._tokens = 10 * _replay_cost(tasks)  # spent on an earlier night

        _, outcome = self._run(
            tasks, backend=backend, budget=_replay_cost(tasks) + 1
        )

        self.assertIn("reflect", backend.calls)
        self.assertTrue(outcome.report.accepted)

    def test_held_out_test_replay_is_skipped_when_the_budget_is_spent(self):
        tasks = _tasks(test_fraction=0.2)
        n_test = sum(1 for t in tasks if t.split == "test")
        self.assertGreater(n_test, 0)
        limit = _replay_cost(tasks)

        backend, outcome = self._run(tasks, budget=limit)

        self.assertLessEqual(backend.tokens_used(), limit)
        notes = _budget_notes(outcome.report)
        self.assertEqual(len(notes), 1)
        self.assertIn("held-out test replay", notes[0])
        events = read_events(os.path.join(outcome.staging_dir, "evidence.jsonl"))
        skips = [e["step"] for e in events if (e["stage"], e["event"]) == ("budget", "skip")]
        self.assertIn("held-out test replay", skips)
        self.assertFalse(
            [e for e in events if (e["stage"], e["event"]) == ("test", "held_out_score")]
        )

    def test_fanout_groups_do_not_start_once_the_budget_is_spent(self):
        claude_home = os.path.join(self._home.name, ".claude")
        group_docs = {}
        for name in ("research-skill", "programming-skill"):
            path = os.path.join(claude_home, "skills", name, "SKILL.md")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            group_docs[name] = f"# {name}\n"
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(group_docs[name])
        tasks = [replace(t, skill_hint="research-skill") for t in _tasks()]
        tasks += [
            replace(t, id=f"prog-{t.id}", skill_hint="programming-skill")
            for t in assign_splits(programmer_persona(), val_fraction=0.34, seed=1)
        ]

        backend, outcome = self._run(tasks, budget=1, multi_skill_fanout=True)

        # The main replay had already started, so it finished; no reflect ran
        # and no group replayed anything against its own skill document.
        self.assertNotIn("reflect", backend.calls)
        self.assertTrue(backend.attempt_skills)
        for skill in backend.attempt_skills:
            for doc in group_docs.values():
                self.assertNotIn(doc.strip(), skill)
        rows = {row.skill_name: row for row in outcome.report.skill_groups}
        self.assertEqual(set(rows), set(group_docs))
        for row in rows.values():
            self.assertEqual(row.status, SKIPPED)
            self.assertIn("token budget", row.reason)
        notes = _budget_notes(outcome.report)
        self.assertEqual(len(notes), 1)
        self.assertIn("2 per-skill group(s)", notes[0])
        events = read_events(os.path.join(outcome.staging_dir, "evidence.jsonl"))
        skipped_groups = {
            e.get("skill") for e in events
            if (e["stage"], e["event"]) == ("budget", "skip")
            and e["step"] == "per-skill group"
        }
        self.assertEqual(skipped_groups, set(group_docs))

    def test_invalid_budget_is_rejected_before_any_model_call(self):
        for value in (-1, "lots", 0.5, True, float("inf")):
            with self.subTest(max_tokens_per_night=value):
                backend = MeteredBackend()
                with self.assertRaisesRegex(ValueError, "max_tokens_per_night"):
                    self._run(_tasks(), backend=backend, budget=value)
                self.assertEqual(backend.calls, [])

    def test_infinity_from_a_json_config_is_rejected(self):
        value = json.loads('{"max_tokens_per_night": Infinity}')["max_tokens_per_night"]
        with self.assertRaisesRegex(ValueError, "max_tokens_per_night"):
            self._run(_tasks(), budget=value)


class TokenBudgetHandoffTest(unittest.TestCase):
    """A handoff round never stops on a count that includes unanswered prompts."""

    def test_cli_round_with_unanswered_prompts_pauses_and_persists_nothing(self):
        from skillopt_sleep.__main__ import main
        from skillopt_sleep.tasks_file import make_tasks_payload, write_tasks_file

        with tempfile.TemporaryDirectory() as proj, \
                tempfile.TemporaryDirectory() as home:
            config_path = os.path.join(home, "config.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump({"max_tokens_per_night": 1}, handle)
            tasks_path = os.path.join(proj, "tasks.json")
            payload = make_tasks_payload(_arithmetic_tasks(), project=proj)
            payload["reviewed"] = True
            write_tasks_file(tasks_path, payload)
            claude_home = os.path.join(home, ".claude")

            with mock.patch(
                "skillopt_sleep.config._user_config_path", return_value=config_path
            ):
                rc = main([
                    "run", "--backend", "handoff",
                    "--project", proj,
                    "--claude-home", claude_home,
                    "--tasks-file", tasks_path,
                ])
                state_path = load_config(
                    invoked_project=proj, claude_home=claude_home
                ).state_path

            self.assertEqual(rc, 3)
            hdir = os.path.join(proj, ".skillopt-sleep-handoff")
            self.assertTrue(os.path.exists(os.path.join(hdir, "PROMPTS.md")))
            self.assertFalse(os.path.exists(state_path))
            staged = glob.glob(
                os.path.join(proj, ".skillopt-sleep", "staging", "*", "report.md")
            )
            self.assertEqual(staged, [])

    def test_budget_is_applied_once_the_round_is_answered(self):
        with tempfile.TemporaryDirectory() as proj, \
                tempfile.TemporaryDirectory() as home:
            hdir = os.path.join(proj, ".skillopt-sleep-handoff")
            cfg = load_config(
                invoked_project=proj, projects="invoked", backend="handoff",
                claude_home=os.path.join(home, ".claude"),
                state_dir=os.path.join(home, "state"), auto_adopt=False,
            )
            cfg.data["max_tokens_per_night"] = 1

            first = HandoffBackend(handoff_dir=hdir)
            with self.assertRaises(PendingCalls):
                run_sleep_cycle(cfg, seed_tasks=_arithmetic_tasks(), backend=first)
            self.assertTrue(first.pending)
            self.assertFalse(os.path.exists(cfg.state_path))
            for key in first.pending:
                with open(first.answer_path(key), "w", encoding="utf-8") as handle:
                    handle.write("cannot say")

            second = HandoffBackend(handoff_dir=hdir)
            outcome = run_sleep_cycle(
                cfg, seed_tasks=_arithmetic_tasks(), backend=second
            )

            self.assertEqual(second.pending, {})
            self.assertEqual(outcome.report.night, 1)
            notes = _budget_notes(outcome.report)
            self.assertEqual(len(notes), 1)
            self.assertIn("skill edits", notes[0])
            self.assertTrue(os.path.exists(cfg.state_path))


class TokenBudgetConsolidateTest(unittest.TestCase):
    def test_spent_budget_skips_the_whole_consolidation(self):
        backend = MeteredBackend()
        result = consolidate(
            backend, _tasks(), "", "", budget_exhausted=lambda: True
        )
        self.assertEqual(backend.calls, [])
        self.assertFalse(result.accepted)
        self.assertEqual(result.gate_action, "budget_exhausted")
        self.assertEqual(result.budget_skipped, ["consolidation"])
        self.assertEqual(result.applied_edits, [])

    def test_no_budget_keeps_the_previous_behavior(self):
        plain = MeteredBackend()
        budgeted = MeteredBackend()
        before = consolidate(plain, _tasks(), "", "")
        after = consolidate(budgeted, _tasks(), "", "", budget_exhausted=lambda: False)
        self.assertEqual(plain.calls, budgeted.calls)
        self.assertEqual(before.gate_action, after.gate_action)
        self.assertEqual(before.new_skill, after.new_skill)
        self.assertEqual(after.budget_skipped, [])


class TokenBudgetGroupsTest(unittest.TestCase):
    def test_groups_not_started_are_skipped_once_the_budget_is_spent(self):
        backend = MeteredBackend()
        tasks = _tasks()
        groups = [
            SkillGroup("first-skill", "", list(tasks)),
            SkillGroup("second-skill", "", list(tasks)),
        ]

        outcomes = consolidate_groups(
            backend,
            groups,
            budget_exhausted=lambda: backend.tokens_used() > 0,
        )

        first = outcomes["first-skill"]
        self.assertEqual(first.status, CONSOLIDATED)
        self.assertIn("skipped skill edits", first.reason)
        second = outcomes["second-skill"]
        self.assertEqual(second.status, SKIPPED)
        self.assertIn("token budget", second.reason)
        self.assertEqual(second.n_tasks, len(tasks))
        self.assertIsNone(second.result)


class TokenBudgetProviderCallTest(unittest.TestCase):
    """Count the provider processes a budget-stopped night really starts.

    These drive ``ClaudeCliBackend`` through its own replay routes, with
    ``subprocess.run`` answered by ``SyntheticClaudeProcess``. Replies the
    response cache does not serve (tool replay, a timed out call) are visible
    as new processes, which a caching mock cannot show.
    """

    def setUp(self) -> None:
        self._proj = tempfile.TemporaryDirectory()
        self._home = tempfile.TemporaryDirectory()
        self.addCleanup(self._proj.cleanup)
        self.addCleanup(self._home.cleanup)
        patcher = mock.patch.dict(os.environ, {"SKILLOPT_SLEEP_WORKERS": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _night(self, tasks, *, timeout: bool):
        cfg = load_config(
            invoked_project=self._proj.name,
            projects="invoked",
            backend="claude",
            claude_home=os.path.join(self._home.name, ".claude"),
            state_dir=os.path.join(self._home.name, "state"),
            auto_adopt=False,
        )
        # Any first call spends this, so only the initial replay may run.
        cfg.data["max_tokens_per_night"] = 1
        backend, process = _synthetic_claude(timeout)
        with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=process):
            outcome = run_sleep_cycle(cfg, seed_tasks=tasks, backend=backend)
        return process, outcome

    def _assert_only_the_initial_replay_ran(self, tasks, process, outcome) -> None:
        n_val = sum(1 for t in tasks if t.split == "val")
        n_train = sum(1 for t in tasks if t.split == "train")
        self.assertGreater(n_val, 0)
        phases = process.phases()
        self.assertEqual(phases.count("final_val"), 0)
        self.assertEqual(
            phases, ["baseline_val"] * n_val + ["train"] * n_train
        )
        report = outcome.report
        self.assertFalse(report.accepted)
        notes = _budget_notes(report)
        self.assertEqual(len(notes), 1)
        self.assertIn(
            "skipped: skill edits, memory edits, final validation replay", notes[0]
        )
        events = read_events(os.path.join(outcome.staging_dir, "evidence.jsonl"))
        self.assertFalse([
            e for e in events
            if (e["stage"], e["event"]) == ("replay", "result")
            and e.get("phase") == "final_val"
        ])
        skips = [e["step"] for e in events if (e["stage"], e["event"]) == ("budget", "skip")]
        self.assertIn("final validation replay", skips)
        decision = [e for e in events if (e["stage"], e["event"]) == ("gate", "decision")]
        self.assertEqual(len(decision), 1)
        self.assertEqual(
            decision[0]["candidate_score"], decision[0]["baseline_score"]
        )

    def test_tool_route_answers_make_no_final_validation_call(self):
        tasks = _tool_tasks()
        process, outcome = self._night(tasks, timeout=False)
        self.assertEqual({route for _phase, route in process.calls}, {"tools"})
        self._assert_only_the_initial_replay_ran(tasks, process, outcome)

    def test_tool_route_timeouts_make_no_final_validation_call(self):
        tasks = _tool_tasks()
        process, outcome = self._night(tasks, timeout=True)
        self.assertEqual({route for _phase, route in process.calls}, {"tools"})
        self._assert_only_the_initial_replay_ran(tasks, process, outcome)

    def test_text_route_timeouts_make_no_final_validation_call(self):
        # An empty reply from a failed call is not kept in the response cache,
        # so re-scoring the same documents would start the process again.
        tasks = _arithmetic_tasks()
        process, outcome = self._night(tasks, timeout=True)
        self.assertEqual({route for _phase, route in process.calls}, {"text"})
        self._assert_only_the_initial_replay_ran(tasks, process, outcome)

    def test_text_route_answers_stay_free_cache_hits(self):
        # Control: here the final replay would have been served from cache.
        tasks = _arithmetic_tasks()
        process, outcome = self._night(tasks, timeout=False)
        self._assert_only_the_initial_replay_ran(tasks, process, outcome)


class TokenBudgetFinalScoringTest(unittest.TestCase):
    """With the budget spent, final scoring reuses the replay of the kept edit."""

    def test_kept_skill_edit_is_scored_from_its_own_validation_replay(self):
        tasks = _tool_tasks()
        n_val = sum(1 for t in tasks if t.split == "val")
        n_train = sum(1 for t in tasks if t.split == "train")
        backend, process = _synthetic_claude()

        def spent() -> bool:
            # The budget runs out once the skill step's reflect call started.
            return ("train", "text") in process.calls

        with mock.patch.dict(os.environ, {"SKILLOPT_SLEEP_WORKERS": "1"}), \
                mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=process):
            result = consolidate(
                backend, tasks, "# skill\n", "", budget_exhausted=spent
            )

        self.assertEqual(
            process.calls,
            [("baseline_val", "tools")] * n_val
            + [("train", "tools")] * n_train
            + [("train", "text")]
            + [("gate_trial:skill", "tools")] * n_val,
        )
        self.assertEqual(result.budget_skipped, ["memory edits", "final validation replay"])
        self.assertTrue(result.accepted)
        self.assertIn(_SEARCH_RULE, result.new_skill)
        self.assertEqual(result.candidate_score, 1.0)
        trial, final = result.gate_trials
        self.assertEqual(trial["target"], "skill")
        self.assertEqual(final["target"], "final")
        self.assertEqual(final["candidate_score"], trial["candidate_score"])


if __name__ == "__main__":
    unittest.main()

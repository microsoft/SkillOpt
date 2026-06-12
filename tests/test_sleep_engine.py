"""Tests for the SkillOpt-Sleep engine.

Pure-stdlib (unittest), deterministic, no API key, no third-party deps.
Run:  python3.12 -m pytest tests/test_sleep_engine.py
  or: python3.12 -m unittest skillopt_sleep ... (see bottom)
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from skillopt_sleep.backend import MockBackend, exact_score, keyword_soft_score
from skillopt_sleep.config import load_config
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.experiments.personas import researcher_persona, programmer_persona
from skillopt_sleep.harvest import digest_transcript, _detect_feedback, _is_meta_prompt
from skillopt_sleep.memory import apply_edits, current_learned_lines, extract_learned, set_learned
from skillopt_sleep.mine import assign_splits, heuristic_mine, dedup_tasks
from skillopt_sleep.staging import adopt, latest_staging
from skillopt_sleep.types import EditRecord, SessionDigest, TaskRecord


class TestScoring(unittest.TestCase):
    def test_exact_score(self):
        self.assertEqual(exact_score("arXiv:1706.03762", "the id is arXiv:1706.03762 ok"), 1.0)
        self.assertEqual(exact_score("arXiv:1706.03762", "approximately arXiv:1706.037"), 0.0)

    def test_keyword_soft(self):
        self.assertGreater(keyword_soft_score("add login form", "please add the login form"), 0.5)


class TestMemoryEdits(unittest.TestCase):
    def test_add_and_dedup(self):
        doc = set_learned("# skill\n", [])
        doc2, applied = apply_edits(doc, [EditRecord("skill", "add", "Rule A"),
                                          EditRecord("skill", "add", "Rule A")])
        self.assertEqual(len(applied), 1)
        self.assertIn("Rule A", extract_learned(doc2))

    def test_protected_region_roundtrip(self):
        base = "# My hand-written skill\nkeep me\n"
        doc = set_learned(base, ["Rule X"])
        self.assertIn("keep me", doc)
        self.assertEqual(current_learned_lines(doc), ["Rule X"])
        # replacing learned region must preserve hand-written content
        doc2 = set_learned(doc, ["Rule Y"])
        self.assertIn("keep me", doc2)
        self.assertEqual(current_learned_lines(doc2), ["Rule Y"])

    def test_replace_and_delete(self):
        doc = set_learned("", ["old rule about commits"])
        doc, _ = apply_edits(doc, [EditRecord("skill", "replace", "new rule", anchor="old rule")])
        self.assertIn("new rule", extract_learned(doc))
        doc, _ = apply_edits(doc, [EditRecord("skill", "delete", "", anchor="new rule")])
        self.assertEqual(current_learned_lines(doc), [])


class TestHarvest(unittest.TestCase):
    def test_feedback_detection(self):
        self.assertTrue(any(s.startswith("neg:") for s in _detect_feedback("this is still broken")))
        self.assertTrue(any(s.startswith("pos:") for s in _detect_feedback("perfect, thanks")))

    def test_meta_prompt_filter(self):
        self.assertTrue(_is_meta_prompt("/clear"))
        self.assertTrue(_is_meta_prompt("<system-reminder>x</system-reminder>"))
        self.assertFalse(_is_meta_prompt("please refactor the auth module"))

    def test_digest_real_transcript_if_present(self):
        # uses the live machine's transcripts when available; skips otherwise
        base = os.path.expanduser("~/.claude/projects")
        if not os.path.isdir(base):
            self.skipTest("no ~/.claude/projects on this machine")
        found = None
        for root, _d, files in os.walk(base):
            for fn in files:
                if fn.endswith(".jsonl"):
                    found = os.path.join(root, fn)
                    break
            if found:
                break
        if not found:
            self.skipTest("no transcripts")
        d = digest_transcript(found)
        # may be None for empty transcripts; if not, it must have core fields
        if d is not None:
            self.assertIsInstance(d.session_id, str)
            self.assertGreaterEqual(d.n_user_turns + d.n_assistant_turns, 0)


class TestMine(unittest.TestCase):
    def _digest(self, prompts, feedback):
        return SessionDigest(
            session_id="s1", project="/p", user_prompts=prompts,
            assistant_finals=["did stuff"], feedback_signals=feedback,
            n_user_turns=len(prompts), n_assistant_turns=1,
        )

    def test_outcome_inference(self):
        fail = heuristic_mine([self._digest(["fix the parser bug please"], ["neg:still broken"])])
        self.assertEqual(fail[0].outcome, "fail")
        ok = heuristic_mine([self._digest(["format the output"], ["pos:perfect"])])
        self.assertEqual(ok[0].outcome, "success")

    def test_split_stable_and_nonempty(self):
        tasks = assign_splits(researcher_persona(), val_fraction=0.34, seed=42)
        splits = {t.split for t in tasks}
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        # stable across calls
        again = assign_splits(researcher_persona(), val_fraction=0.34, seed=42)
        self.assertEqual([t.split for t in tasks], [t.split for t in again])

    def test_dream_never_in_val_or_test(self):
        # the anti-overfitting guarantee: origin='dream' tasks only ever land in train
        from skillopt_sleep.types import TaskRecord
        real = researcher_persona()
        dream = [TaskRecord(id=f"d{i}", project="/p", intent=f"dream {i}",
                            origin="dream", derived_from="r0") for i in range(5)]
        tasks = assign_splits(real + dream, val_fraction=0.3, test_fraction=0.3, seed=7)
        for t in tasks:
            if t.origin == "dream":
                self.assertEqual(t.split, "train")
        # val and test contain ONLY real tasks
        for t in tasks:
            if t.split in ("val", "test"):
                self.assertEqual(t.origin, "real")
        # and val/test are disjoint (a task is in exactly one split)
        self.assertTrue(any(t.split == "val" for t in tasks))


class TestConsolidateGate(unittest.TestCase):
    def test_accepts_helpful_rejects_harmful(self):
        be = MockBackend()
        tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)
        res = consolidate(be, tasks, set_learned("", []), "", edit_budget=4,
                          gate_metric="mixed", night=1)
        self.assertTrue(res.accepted)
        self.assertGreater(res.candidate_score, res.baseline_score)

    def test_no_op_when_already_optimal(self):
        be = MockBackend()
        tasks = assign_splits(programmer_persona(), holdout_fraction=0.34, seed=1)
        # first night learns the rule
        r1 = consolidate(be, tasks, set_learned("", []), "", edit_budget=4, night=1)
        # second night on the learned skill should find nothing to add
        r2 = consolidate(be, tasks, r1.new_skill, r1.new_memory, edit_budget=4, night=2)
        self.assertEqual(len(r2.applied_edits), 0)


class TestRuleJudge(unittest.TestCase):
    def test_section_and_regex(self):
        from skillopt_sleep.judges import score_rule_judge
        j = {"kind": "rule", "checks": [
            {"op": "section_present", "arg": "Key Risks"},
            {"op": "regex", "arg": r"[Cc]onfidence\s*[:=]"},
        ]}
        ok = "# Brief\n## Key Risks\nstuff\nConfidence: High"
        self.assertEqual(score_rule_judge(j, ok)[0], 1.0)
        self.assertEqual(score_rule_judge(j, "just an answer")[0], 0.0)

    def test_max_chars(self):
        from skillopt_sleep.judges import score_rule_judge
        j = {"checks": [{"op": "max_chars", "arg": 50}]}
        self.assertEqual(score_rule_judge(j, "x" * 10)[0], 1.0)
        self.assertEqual(score_rule_judge(j, "x" * 100)[0], 0.0)

    def test_partial_soft_score(self):
        from skillopt_sleep.judges import score_rule_judge
        j = {"checks": [
            {"op": "contains", "arg": "alpha"},
            {"op": "contains", "arg": "beta"},
        ]}
        h, s, _ = score_rule_judge(j, "only alpha here")
        self.assertEqual(h, 0.0)
        self.assertAlmostEqual(s, 0.5)


class TestGbrainLoader(unittest.TestCase):
    def test_loads_when_present(self):
        from skillopt_sleep.experiments.gbrain_bench import find_data_root, load_seed
        root = find_data_root()
        if not root:
            self.skipTest("gbrain-evals data not present")
        skill, tasks = load_seed(root, "brief-writer")
        self.assertTrue(skill)
        # gbrain held-out maps to our 'test'; benchmark pool to train/val
        self.assertTrue(any(t.split == "test" for t in tasks))
        self.assertTrue(any(t.split == "val" for t in tasks))
        self.assertTrue(all(t.reference_kind == "rule" for t in tasks))
        # the deficient skill must FAIL its own held-out (test) checks (baseline 0)
        from skillopt_sleep.judges import score_rule_judge
        ho = [t for t in tasks if t.split == "test"][0]
        self.assertEqual(score_rule_judge(ho.judge, skill)[0], 0.0)


class TestLlmMiner(unittest.TestCase):
    def test_miner_emits_checkable_tasks(self):
        # a stub backend whose _call returns canned miner JSON => deterministic
        from skillopt_sleep.backend import Backend
        from skillopt_sleep.llm_miner import make_llm_miner

        class StubBackend(Backend):
            name = "stub"
            def _call(self, prompt, *, max_tokens=1024):
                return ('[{"intent":"write a research brief",'
                        '"checks":[{"op":"section_present","arg":"Key Risks"}],'
                        '"rubric":"has a risks section","satisfied":false}]')

        digest = SessionDigest(session_id="s1", project="/p",
                               user_prompts=["write a brief on X"],
                               assistant_finals=["a brief"], n_user_turns=1)
        miner = make_llm_miner(StubBackend())
        tasks = miner([digest])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].reference_kind, "rule")
        self.assertEqual(tasks[0].judge["checks"][0]["op"], "section_present")

    def test_miner_drops_uncheckable(self):
        from skillopt_sleep.backend import Backend
        from skillopt_sleep.llm_miner import make_llm_miner

        class EmptyBackend(Backend):
            name = "stub"
            def _call(self, prompt, *, max_tokens=1024):
                return "[]"

        digest = SessionDigest(session_id="s1", project="/p",
                               user_prompts=["chat"], n_user_turns=1)
        self.assertEqual(make_llm_miner(EmptyBackend())([digest]), [])


class TestMultiObjectiveAndPrefs(unittest.TestCase):
    def test_multi_objective_reward(self):
        from skillopt_sleep.replay import multi_objective_reward
        from skillopt_sleep.types import ReplayResult, TaskRecord
        t = TaskRecord(id="t", project="/p", intent="x")
        expensive = [(t, ReplayResult(id="t", hard=1.0, tokens=4000, latency_ms=20000))]
        cheap = [(t, ReplayResult(id="t", hard=1.0, tokens=200, latency_ms=1000))]
        self.assertEqual(
            multi_objective_reward(expensive, w_acc=1, w_tokens=0, w_latency=0),
            multi_objective_reward(cheap, w_acc=1, w_tokens=0, w_latency=0),
        )
        re = multi_objective_reward(expensive, w_acc=1, w_tokens=1, w_latency=1)
        rc = multi_objective_reward(cheap, w_acc=1, w_tokens=1, w_latency=1)
        self.assertGreater(rc, re)

    def test_preferences_injected_into_reflect(self):
        from skillopt_sleep.backend import CliBackend
        from skillopt_sleep.types import TaskRecord, ReplayResult
        captured = {}

        class CapBackend(CliBackend):
            name = "cap"
            def _call(self, prompt, *, max_tokens=1024):
                captured["prompt"] = prompt
                return "[]"

        be = CapBackend()
        be.preferences = "Prefer concise British English."
        t = TaskRecord(id="t", project="/p", intent="x", reference_kind="rule",
                       judge={"checks": [{"op": "contains", "arg": "z"}]})
        be.reflect([(t, ReplayResult(id="t", hard=0.0, fail_reason="failed: contains=z"))],
                   [], "skill", "", edit_budget=2, evolve_skill=True, evolve_memory=False)
        self.assertIn("British English", captured["prompt"])

    def test_replay_records_cost(self):
        from skillopt_sleep.backend import MockBackend
        from skillopt_sleep.replay import replay_one
        from skillopt_sleep.types import TaskRecord
        t = TaskRecord(id="t", project="/p", intent="hello world",
                       reference_kind="exact", reference="hi")
        r = replay_one(MockBackend(), t, "some skill text", "")
        self.assertGreater(r.tokens, 0)
        self.assertGreaterEqual(r.latency_ms, 0.0)


class TestMultiRolloutAndBudget(unittest.TestCase):
    def test_rolloutset_stats(self):
        from skillopt_sleep.rollout import RolloutSet
        from skillopt_sleep.types import ReplayResult, TaskRecord
        rs = RolloutSet(task=TaskRecord(id="t", project="/p", intent="x"),
                        attempts=[ReplayResult(id="t", hard=1.0),
                                  ReplayResult(id="t", hard=0.0),
                                  ReplayResult(id="t", hard=1.0)])
        self.assertEqual(rs.best.hard, 1.0)
        self.assertEqual(rs.worst.hard, 0.0)
        self.assertEqual(rs.spread, 1.0)
        self.assertAlmostEqual(rs.pass_rate, 2 / 3)

    def test_budget_exhaustion_and_plan(self):
        from skillopt_sleep.budget import Budget, plan_depth
        clock = [0.0]
        b = Budget(max_tokens=1000)
        b.start(lambda: clock[0], tokens_now=0)
        self.assertFalse(b.exhausted(tokens_now=500, clock_fn=lambda: clock[0]))
        self.assertTrue(b.exhausted(tokens_now=1000, clock_fn=lambda: clock[0]))
        self.assertEqual(plan_depth(Budget(), n_tasks=5, default_nights=2, default_k=1), (2, 1))
        nights, k = plan_depth(Budget(max_tokens=100_000), n_tasks=5)
        self.assertGreaterEqual(nights, 1)
        self.assertGreaterEqual(k, 1)

    def test_contrastive_reflect_with_stub(self):
        from skillopt_sleep.backend import Backend
        from skillopt_sleep.rollout import RolloutSet, contrastive_reflect
        from skillopt_sleep.types import ReplayResult, TaskRecord

        class StubBackend(Backend):
            name = "stub"
            def _call(self, prompt, *, max_tokens=1024):
                return '[{"op":"add","content":"always do the good thing","rationale":"good passed"}]'

        rs = RolloutSet(task=TaskRecord(id="t", project="/p", intent="x"),
                        attempts=[ReplayResult(id="t", hard=1.0, response="good"),
                                  ReplayResult(id="t", hard=0.0, response="bad")])
        edits = contrastive_reflect(StubBackend(), [rs], "skill", "")
        self.assertEqual(len(edits), 1)
        self.assertIn("good thing", edits[0].content)


class TestSlowUpdate(unittest.TestCase):
    def test_protected_field_roundtrip(self):
        from skillopt_sleep.slow_update import (
            replace_slow_field, extract_slow_field, has_slow_field,
            SLOW_UPDATE_START, SLOW_UPDATE_END,
        )
        base = "# skill\nkeep me\n"
        doc = replace_slow_field(base, "durable lesson A")
        self.assertTrue(has_slow_field(doc))
        self.assertIn("keep me", doc)
        self.assertEqual(extract_slow_field(doc), "durable lesson A")
        # replacing keeps exactly one block and preserves hand-written text
        doc2 = replace_slow_field(doc, "durable lesson B")
        self.assertEqual(doc2.count(SLOW_UPDATE_START), 1)
        self.assertEqual(doc2.count(SLOW_UPDATE_END), 1)
        self.assertEqual(extract_slow_field(doc2), "durable lesson B")
        self.assertIn("keep me", doc2)

    def test_run_slow_update_with_stub_backend(self):
        from skillopt_sleep.backend import Backend
        from skillopt_sleep.slow_update import run_slow_update
        from skillopt_sleep.types import TaskRecord, ReplayResult

        class StubBackend(Backend):
            name = "stub"
            def _call(self, prompt, *, max_tokens=1024):
                return '{"guidance": "- keep doing X\\n- avoid regression Y"}'

        t = TaskRecord(id="t1", project="/p", intent="do thing")
        prev = [(t, ReplayResult(id="t1", hard=0.0))]  # was failing
        curr = [(t, ReplayResult(id="t1", hard=1.0))]  # now passing (improved)
        out = run_slow_update(StubBackend(), prev_skill="s0", curr_skill="s1",
                              prev_pairs=prev, curr_pairs=curr)
        # improvements alone with no regression/persistent-fail and no prior text -> None
        self.assertIsNone(out)
        # a regression triggers guidance
        prev2 = [(t, ReplayResult(id="t1", hard=1.0))]
        curr2 = [(t, ReplayResult(id="t1", hard=0.0))]
        out2 = run_slow_update(StubBackend(), prev_skill="s0", curr_skill="s1",
                               prev_pairs=prev2, curr_pairs=curr2)
        self.assertIn("keep doing X", out2)


class TestToolLoop(unittest.TestCase):
    def test_tool_called_judge_via_replay(self):
        from skillopt_sleep.backend import MockBackend
        from skillopt_sleep.replay import replay_one, _required_tools
        from skillopt_sleep.memory import set_learned
        from skillopt_sleep.types import TaskRecord

        task = TaskRecord(
            id="qa1", project="/p", intent="answer the question",
            reference_kind="rule",
            judge={"kind": "rule", "checks": [{"op": "tool_called", "arg": "search"}]},
        )
        self.assertEqual(_required_tools(task), ["search"])
        be = MockBackend()
        # deficient skill: no instruction to search -> tool not called -> hard 0
        deficient = "Answer from memory. Do NOT use tools."
        r0 = replay_one(be, task, deficient, "")
        self.assertEqual(r0.hard, 0.0)
        self.assertEqual(r0.tools_called, [])
        # learned rule to use ./search -> tool called -> hard 1
        learned = set_learned(deficient, ["Before answering you MUST run ./search first."])
        r1 = replay_one(be, task, learned, "")
        self.assertEqual(r1.hard, 1.0)
        self.assertEqual(r1.tools_called, ["search"])


class TestFullCycleAndAdopt(unittest.TestCase):
    def test_cycle_stage_then_adopt_with_backup(self):
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            cfg = load_config(
                invoked_project=proj, projects="invoked", backend="mock",
                claude_home=os.path.join(home, ".claude"),
                managed_skill_name="skillopt-sleep-learned",
                auto_adopt=False,
            )
            # seed a known persona so we don't depend on ~/.claude
            tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)

            outcome = run_sleep_cycle(cfg, seed_tasks=tasks)
            self.assertTrue(outcome.report.accepted)
            self.assertTrue(os.path.isdir(outcome.staging_dir))
            self.assertTrue(os.path.exists(os.path.join(outcome.staging_dir, "report.md")))

            # nothing live touched yet
            live_skill = cfg.managed_skill_path()
            self.assertFalse(os.path.exists(live_skill))

            # adopt -> live file created, backup dir exists
            updated = adopt(outcome.staging_dir)
            self.assertTrue(any("SKILL.md" in p for p in updated))
            self.assertTrue(os.path.exists(live_skill))
            with open(live_skill) as f:
                self.assertIn("answer", f.read().lower())


class TestCopilotBackend(unittest.TestCase):
    """Pure-logic tests for CopilotCliBackend — no `copilot` CLI required."""

    def test_alias_resolution(self):
        from skillopt_sleep.backend import CopilotCliBackend, get_backend
        for name in ("copilot", "github_copilot", "copilot_cli", "gh_copilot"):
            self.assertIsInstance(get_backend(name), CopilotCliBackend, name)

    def test_parse_jsonl_concatenates_assistant_messages(self):
        from skillopt_sleep.backend import CopilotCliBackend
        raw = "\n".join([
            '{"type":"session.info","data":{}}',
            '{"type":"assistant.message","data":{"content":"hello"}}',
            'not-json-noise',
            '{"type":"user.message","data":{"content":"ignored"}}',
            '{"type":"assistant.message","data":{"content":"world"}}',
        ])
        self.assertEqual(CopilotCliBackend._parse_jsonl_response(raw), "hello\nworld")

    def test_parse_jsonl_ignores_non_assistant_and_blank(self):
        from skillopt_sleep.backend import CopilotCliBackend
        self.assertEqual(CopilotCliBackend._parse_jsonl_response(""), "")
        self.assertEqual(
            CopilotCliBackend._parse_jsonl_response('{"type":"result","data":{"content":"x"}}'),
            "",
        )
        # assistant.message with empty/missing content contributes nothing
        self.assertEqual(
            CopilotCliBackend._parse_jsonl_response(
                '{"type":"assistant.message","data":{"content":""}}\n'
                '{"type":"assistant.message","data":{}}'
            ),
            "",
        )

    def test_isolated_home_by_default(self):
        from skillopt_sleep.backend import CopilotCliBackend
        be = CopilotCliBackend()
        self.assertFalse(be.full_env)
        self.assertTrue(be.copilot_home)  # an isolated COPILOT_HOME is set

    def test_full_env_opt_out(self):
        from skillopt_sleep.backend import CopilotCliBackend
        prev = os.environ.get("SKILLOPT_SLEEP_COPILOT_FULL_ENV")
        os.environ["SKILLOPT_SLEEP_COPILOT_FULL_ENV"] = "1"
        try:
            be = CopilotCliBackend()
            self.assertTrue(be.full_env)
            self.assertEqual(be.copilot_home, "")  # real user environment used
        finally:
            if prev is None:
                os.environ.pop("SKILLOPT_SLEEP_COPILOT_FULL_ENV", None)
            else:
                os.environ["SKILLOPT_SLEEP_COPILOT_FULL_ENV"] = prev

    def test_home_override_env(self):
        from skillopt_sleep.backend import CopilotCliBackend
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "myhome")
            prev = os.environ.get("SKILLOPT_SLEEP_COPILOT_HOME")
            os.environ["SKILLOPT_SLEEP_COPILOT_HOME"] = target
            try:
                be = CopilotCliBackend()
                self.assertEqual(be.copilot_home, target)
                self.assertTrue(os.path.isdir(target))  # created on init
            finally:
                if prev is None:
                    os.environ.pop("SKILLOPT_SLEEP_COPILOT_HOME", None)
                else:
                    os.environ["SKILLOPT_SLEEP_COPILOT_HOME"] = prev

    def test_attempt_with_tools_honest_detection(self):
        # End-to-end (no real CLI): a tiny per-OS stub stands in for `copilot`.
        # It runs the local `search` shim the backend writes into its work dir
        # (so the calllog is written — honest detection) then prints one JSONL
        # assistant.message. Proves both the JSONL parse and that the tool call
        # is detected from the shim's log, not from a self-reported marker.
        import shutil
        import stat

        from skillopt_sleep.backend import CopilotCliBackend

        stub_dir = tempfile.mkdtemp(prefix="skillopt_sleep_stub_")
        try:
            if os.name == "nt":
                stub = os.path.join(stub_dir, "copilot.cmd")
                with open(stub, "w") as f:
                    # The backend writes `search.cmd`; run it (explicit `.\` so
                    # cmd's `call` resolves it from the cwd reliably) so the
                    # calllog is populated, then emit the JSONL line. None of
                    # `{ } " :` need escaping in batch echo (no > < | & ^ %).
                    f.write(
                        "@echo off\n"
                        'call .\\search.cmd "q" >nul 2>&1\n'
                        'echo {"type":"assistant.message","data":{"content":"Paris"}}\n'
                    )
            else:
                stub = os.path.join(stub_dir, "copilot")
                with open(stub, "w") as f:
                    f.write(
                        "#!/usr/bin/env bash\n"
                        './search "q" >/dev/null 2>&1\n'
                        "echo '{\"type\":\"assistant.message\",\"data\":{\"content\":\"Paris\"}}'\n"
                    )
                os.chmod(
                    stub,
                    os.stat(stub).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
                )

            be = CopilotCliBackend(copilot_path=stub, timeout=60)
            task = TaskRecord(id="t1", project="p", intent="What is the capital of France?")
            resp, called = be.attempt_with_tools(task, skill="", memory="", tools=["search"])

            self.assertEqual(resp, "Paris")  # JSONL parsed via _parse_jsonl_response
            self.assertEqual(called, ["search"])  # shim ran; detected from calllog
        finally:
            shutil.rmtree(stub_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

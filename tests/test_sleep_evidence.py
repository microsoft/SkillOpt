"""Tests for the evidence log, the prompt registry, and the dashboard API.

Pure-stdlib (unittest), deterministic, no API key, no network beyond
127.0.0.1, no third-party deps.

Run:  python -m unittest tests.test_sleep_evidence
"""
from __future__ import annotations

import http.client
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from skillopt_sleep import prompts as prompt_registry
from skillopt_sleep.config import load_config
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.evidence import EvidenceLog, read_events
from skillopt_sleep.experiments.personas import researcher_persona
from skillopt_sleep.mine import assign_splits


def _events_by(events, stage=None, event=None):
    out = events
    if stage is not None:
        out = [e for e in out if e.get("stage") == stage]
    if event is not None:
        out = [e for e in out if e.get("event") == event]
    return out


class TestEvidenceLog(unittest.TestCase):
    def test_append_redact_truncate_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "e.jsonl")
            ev = EvidenceLog(path, max_chars=200)
            ev.log("replay", "model_call", prompt="x" * 500,
                   secret="api_key=sk-abcdefghijklmnop")
            ev.log("gate", "decision", action="accept")
            events = read_events(path)
            self.assertEqual([e["seq"] for e in events], [1, 2])
            self.assertIn("truncated", events[0]["prompt"])
            self.assertLessEqual(len(events[0]["prompt"]), 260)
            self.assertNotIn("sk-abcdefghijklmnop", json.dumps(events))
            self.assertIn("REDACTED", events[0]["secret"])

    def test_reader_skips_corrupt_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "e.jsonl")
            ev = EvidenceLog(path)
            ev.log("cycle", "start")
            with open(path, "a", encoding="utf-8") as f:
                f.write("{not json\n")
            ev.log("cycle", "end")
            self.assertEqual(len(read_events(path)), 2)


class TestPromptRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.dict(os.environ, {
            "SKILLOPT_SLEEP_PROMPTS_PATH": os.path.join(self.tmp.name, "prompts.json")})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_defaults_match_legacy_prompts(self):
        # The registry must reproduce the exact legacy wording by default.
        self.assertIn("You are SkillOpt's optimizer", prompt_registry.get_prompt("reflect"))
        self.assertIn("RECURRING tasks", prompt_registry.get_prompt("miner"))
        self.assertIn("Return ONLY the final answer text", prompt_registry.get_prompt("attempt"))
        rendered = prompt_registry.render("attempt", {
            "__SKILL__": "S", "__MEMORY__": "M", "__INTENT__": "I", "__CONTEXT__": "C"})
        self.assertIn("# Skill\nS", rendered)
        self.assertNotIn("__SKILL__", rendered)

    def test_override_takes_effect_without_restart(self):
        self.assertFalse(prompt_registry.is_overridden("judge"))
        prompt_registry.save_overrides({"judge": "CUSTOM __RUBRIC__ / __RESPONSE__"})
        self.assertTrue(prompt_registry.is_overridden("judge"))
        self.assertEqual(
            prompt_registry.render("judge", {"__RUBRIC__": "r", "__RESPONSE__": "x"}),
            "CUSTOM r / x")
        # empty value reverts to default
        prompt_registry.save_overrides({"judge": None})
        self.assertFalse(prompt_registry.is_overridden("judge"))
        self.assertIn("Score how well", prompt_registry.get_prompt("judge"))

    def test_unknown_names_are_ignored(self):
        out = prompt_registry.save_overrides({"nope": "x", "miner": "M __PROMPTS__"})
        self.assertEqual(set(out), {"miner"})
        prompt_registry.save_overrides({"miner": ""})


class TestCycleEvidence(unittest.TestCase):
    def _run(self, **cfg_extra):
        proj = tempfile.mkdtemp()
        home = tempfile.mkdtemp()
        cfg = load_config(
            invoked_project=proj, projects="invoked", backend="mock",
            claude_home=os.path.join(home, ".claude"), auto_adopt=False,
            **cfg_extra)
        tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)
        outcome = run_sleep_cycle(cfg, seed_tasks=tasks)
        return outcome

    def test_evidence_written_with_full_chain(self):
        outcome = self._run()
        path = os.path.join(outcome.staging_dir, "evidence.jsonl")
        self.assertTrue(os.path.exists(path), "evidence.jsonl missing from staging dir")
        events = read_events(path)
        # chain: cycle start .. task_ready .. replay results (phased) ..
        # reflect edits .. gate baseline/trial/decision .. staged .. cycle end
        self.assertTrue(_events_by(events, "cycle", "start"))
        self.assertTrue(_events_by(events, "mine", "task_ready"))
        splits = {e["split"] for e in _events_by(events, "mine", "task_ready")}
        self.assertIn("train", splits)
        results = _events_by(events, "replay", "result")
        self.assertTrue(results)
        phases = {e["phase"] for e in results}
        self.assertIn("baseline_val", phases)
        self.assertIn("final_val", phases)
        self.assertTrue(_events_by(events, "reflect", "edits_returned"))
        self.assertTrue(_events_by(events, "gate", "baseline"))
        decision = _events_by(events, "gate", "decision")
        self.assertEqual(len(decision), 1)
        self.assertIn("formula", decision[0])
        self.assertTrue(_events_by(events, "stage", "staged"))
        end = _events_by(events, "cycle", "end")
        self.assertEqual(len(end), 1)
        self.assertEqual(end[0]["outcome"], "completed")
        # the report landed in the SAME pre-created folder as the evidence
        self.assertTrue(os.path.exists(os.path.join(outcome.staging_dir, "report.md")))

    def test_evidence_can_be_disabled(self):
        outcome = self._run(evidence_log=False)
        self.assertFalse(
            os.path.exists(os.path.join(outcome.staging_dir, "evidence.jsonl")))

    def test_no_tasks_night_is_not_adoptable_but_keeps_evidence(self):
        from skillopt_sleep.staging import latest_staging
        proj = tempfile.mkdtemp()
        home = tempfile.mkdtemp()
        cfg = load_config(
            invoked_project=proj, projects="invoked", backend="mock",
            claude_home=os.path.join(home, ".claude"))
        outcome = run_sleep_cycle(cfg, seed_tasks=[])
        self.assertEqual(outcome.staging_dir, "")
        # an evidence-only folder exists but latest_staging must skip it
        self.assertIsNone(latest_staging(proj))


class TestDashboardApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from skillopt_sleep.dashboard import DashboardHandler, _RunState
        from http.server import ThreadingHTTPServer
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env = mock.patch.dict(os.environ, {
            "SKILLOPT_SLEEP_PROMPTS_PATH": os.path.join(cls.tmp.name, "prompts.json")})
        cls.env.start()
        DashboardHandler.project = cls.tmp.name
        DashboardHandler.run_state = _RunState()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.env.stop()
        cls.tmp.cleanup()

    def _get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        r = conn.getresponse()
        body = json.loads(r.read().decode("utf-8")) if "json" in r.getheader("Content-Type", "") else r.read()
        conn.close()
        return r.status, body

    def _post(self, path, obj):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(obj).encode("utf-8")
        conn.request("POST", path, body=payload,
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        body = json.loads(r.read().decode("utf-8"))
        conn.close()
        return r.status, body

    def test_overview_and_html(self):
        status, body = self._get("/api/overview")
        self.assertEqual(status, 200)
        self.assertEqual(body["project"], self.tmp.name)
        self.assertEqual({p["name"] for p in body["prompts"]},
                         {"miner", "attempt", "judge", "reflect"})
        status, html = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Control Panel", html)

    def test_prompt_roundtrip(self):
        status, body = self._post("/api/prompts", {"updates": {"miner": "X __PROMPTS__"}})
        self.assertEqual(status, 200)
        mined = [p for p in body["prompts"] if p["name"] == "miner"][0]
        self.assertEqual(mined["override"], "X __PROMPTS__")
        self._post("/api/prompts", {"updates": {"miner": None}})

    def test_unknown_night_404(self):
        status, _body = self._get("/api/night/nope")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()

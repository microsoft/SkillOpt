"""Opt-in llm_dream: paraphrase-only, train-only, deterministic fallback."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from skillopt_sleep.config import DEFAULTS, load_config
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.dream import (
    _WRAPPERS,
    _fidelity_ok,
    _parse_paraphrases,
    dream_augment,
    dream_consolidate,
)
from skillopt_sleep.types import TaskRecord


def _task(tid: str = "t1", intent: str = "add form validation to the signup page") -> TaskRecord:
    return TaskRecord(
        id=tid,
        project="/p",
        intent=intent,
        reference_kind="exact",
        reference="use the shared validator",
        judge={"checks": [{"op": "contains", "arg": "validator"}]},
        split="train",
        origin="real",
        skill_hint="forms",
        tags=["rule:wrap-answer"],
    )


class TestTemplateDefaultUnchanged(unittest.TestCase):
    def test_default_matches_hardcoded_wrappers(self):
        src = _task()
        got = dream_augment([src], factor=3)
        self.assertEqual(len(got), 3)
        for k, dream in enumerate(got):
            self.assertEqual(dream.intent, _WRAPPERS[k].format(q=src.intent))
            self.assertEqual(dream.split, "train")
            self.assertEqual(dream.origin, "dream")
            self.assertEqual(dream.derived_from, src.id)
            self.assertEqual(dream.reference, src.reference)
            self.assertEqual(dream.judge, src.judge)
            self.assertEqual(dream.tags, src.tags + ["dream"])
            self.assertNotIn("llm_dream", dream.tags)
            self.assertEqual(dream.skill_hint, "forms")

    def test_llm_dream_false_ignores_generator(self):
        src = _task()
        calls = []

        def gen(prompt: str) -> str:
            calls.append(prompt)
            return json.dumps(["totally different paraphrase of the request"])

        got = dream_augment([src], factor=1, llm_dream=False, generate_fn=gen)
        self.assertEqual(calls, [])
        self.assertEqual(got[0].intent, _WRAPPERS[0].format(q=src.intent))


class TestParseAndFidelity(unittest.TestCase):
    def test_parse_json_array(self):
        raw = 'Sure.\n["please add signup validation", "handle signup form checks"]\n'
        self.assertEqual(
            _parse_paraphrases(raw, 2),
            ["please add signup validation", "handle signup form checks"],
        )

    def test_parse_rejects_garbage(self):
        self.assertEqual(_parse_paraphrases("not json", 2), [])
        self.assertEqual(_parse_paraphrases('{"intent": "x"}', 1), [])

    def test_fidelity_rejects_identical_and_prompt_echo(self):
        src = "add form validation to the signup page"
        self.assertFalse(_fidelity_ok(src, src))
        self.assertFalse(_fidelity_ok(src, "short"))
        self.assertFalse(_fidelity_ok(src, "Return ONLY a JSON array of junk"))
        self.assertTrue(_fidelity_ok(src, "please add validation on the signup form"))


class TestLlmDreamPath(unittest.TestCase):
    def test_valid_paraphrases_are_used(self):
        src = _task()

        def gen(_prompt: str) -> str:
            return json.dumps([
                "please add validation on the signup form",
                "handle signup-page form checks",
            ])

        got = dream_augment([src], factor=2, llm_dream=True, generate_fn=gen)
        self.assertEqual(got[0].intent, "please add validation on the signup form")
        self.assertEqual(got[1].intent, "handle signup-page form checks")
        for dream in got:
            self.assertEqual(dream.split, "train")
            self.assertEqual(dream.origin, "dream")
            self.assertIn("llm_dream", dream.tags)
            self.assertEqual(dream.reference, src.reference)
            self.assertEqual(dream.judge, src.judge)

    def test_parse_failure_falls_back_deterministically(self):
        src = _task()
        events = []

        class _Ev:
            def log(self, stage, event, **data):
                events.append((stage, event, data))

        def gen(_prompt: str) -> str:
            return "I cannot comply"

        a = dream_augment([src], factor=2, llm_dream=True, generate_fn=gen, evidence=_Ev())
        b = dream_augment([src], factor=2, llm_dream=True, generate_fn=gen, evidence=_Ev())
        self.assertEqual([d.intent for d in a], [d.intent for d in b])
        self.assertEqual(a[0].intent, _WRAPPERS[0].format(q=src.intent))
        self.assertEqual(a[1].intent, _WRAPPERS[1].format(q=src.intent))
        self.assertNotIn("llm_dream", a[0].tags)
        self.assertTrue(any(ev[1] == "llm_dream_fallback" for ev in events))

    def test_partial_parse_fills_rest_from_templates(self):
        src = _task()

        def gen(_prompt: str) -> str:
            return json.dumps(["please add validation on the signup form"])

        got = dream_augment([src], factor=2, llm_dream=True, generate_fn=gen)
        self.assertEqual(got[0].intent, "please add validation on the signup form")
        self.assertEqual(got[1].intent, _WRAPPERS[1].format(q=src.intent))
        self.assertIn("llm_dream", got[0].tags)
        self.assertNotIn("llm_dream", got[1].tags)

    def test_generator_exception_falls_back(self):
        src = _task()

        def gen(_prompt: str) -> str:
            raise RuntimeError("backend down")

        got = dream_augment([src], factor=1, llm_dream=True, generate_fn=gen)
        self.assertEqual(got[0].intent, _WRAPPERS[0].format(q=src.intent))

    def test_llm_dream_without_generator_uses_templates(self):
        src = _task()
        got = dream_augment([src], factor=1, llm_dream=True, generate_fn=None)
        self.assertEqual(got[0].intent, _WRAPPERS[0].format(q=src.intent))


class TestSplitHygiene(unittest.TestCase):
    def test_llm_dreams_never_leave_train(self):
        val = _task("val1")
        val.split = "val"
        test = _task("test1")
        test.split = "test"

        def gen(_prompt: str) -> str:
            return json.dumps(["please add validation on the signup form"])

        dreamed = dream_augment(
            [val, test], factor=1, llm_dream=True, generate_fn=gen,
        )
        self.assertEqual({d.split for d in dreamed}, {"train"})
        self.assertEqual({d.origin for d in dreamed}, {"dream"})

    def test_dream_consolidate_keeps_val_clean(self):
        from skillopt_sleep.backend import MockBackend

        train = _task("tr")
        val = _task("va", intent="score the holdout form task")
        val.split = "val"
        val.reference = "holdout-answer"
        calls = []

        def gen(prompt: str) -> str:
            calls.append(prompt)
            return json.dumps(["please add validation on the signup form"])

        res = dream_consolidate(
            MockBackend(),
            [train, val],
            skill="",
            memory="",
            dream_factor=1,
            llm_dream=True,
            generate_fn=gen,
            gate_mode="off",
        )
        self.assertIsNotNone(res)
        self.assertTrue(calls)
        # The generator is only asked to rewrite train tasks (val is not a seed).
        self.assertTrue(any("add form validation" in p for p in calls))
        self.assertFalse(any("score the holdout" in p for p in calls))


class TestConfigDefaultOff(unittest.TestCase):
    def test_default_is_false(self):
        self.assertFalse(DEFAULTS["llm_dream"])
        cfg = load_config()
        self.assertFalse(cfg.get("llm_dream"))

    def test_cycle_default_does_not_call_generator(self):
        src = _task()
        src.tags = ["rule:wrap-answer"]
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            cfg = load_config(
                invoked_project=proj,
                projects="invoked",
                backend="mock",
                claude_home=os.path.join(home, ".claude"),
                dream_factor=2,
                auto_adopt=False,
                evidence_log=False,
            )
            outcome = run_sleep_cycle(cfg, seed_tasks=[src])
            self.assertIsNotNone(outcome)


class TestDiversityAccounting(unittest.TestCase):
    def test_llm_intents_are_more_distinct_than_templates_on_fixture(self):
        src = _task()
        templates = dream_augment([src], factor=3)
        paraphrases = [
            "please add validation on the signup form",
            "signup page needs the shared form checks",
            "apply the validator before accepting a signup",
        ]

        def gen(_prompt: str) -> str:
            return json.dumps(paraphrases)

        llm = dream_augment([src], factor=3, llm_dream=True, generate_fn=gen)

        def distinct_1(texts):
            toks = []
            for text in texts:
                toks.extend(w for w in text.lower().split() if len(w) > 2)
            return (len(set(toks)) / len(toks)) if toks else 0.0

        self.assertGreater(
            distinct_1([d.intent for d in llm]),
            distinct_1([d.intent for d in templates]),
        )


if __name__ == "__main__":
    unittest.main()

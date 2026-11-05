"""Tests for pluggable sources, backends, and public harvest helpers (#314).

Pure-stdlib (unittest), deterministic, no API key, no external dependencies.
"""
from __future__ import annotations

import unittest
from unittest import mock

from skillopt_sleep.backend import (
    Backend,
    get_backend,
    get_supported_backends,
    register_backend,
    unregister_backend,
)
from skillopt_sleep.config import load_config
from skillopt_sleep.harvest import (
    detect_feedback,
    is_agent_session,
    is_headless_replay,
    is_meta_prompt,
    redact_secrets,
)
from skillopt_sleep.harvest_sources import (
    get_supported_sources,
    harvest_for_config,
    normalize_transcript_source,
    register_source,
    unregister_source,
)
from skillopt_sleep.types import SessionDigest


def sample_harvest(cfg, **kwargs) -> list[SessionDigest]:
    return [
        SessionDigest(
            session_id="sample-1",
            project="test-proj",
            started_at="2026-11-05T12:00:00Z",
            ended_at="2026-11-05T12:05:00Z",
            user_prompts=["write a test"],
            n_user_turns=1,
        )
    ]


class SampleCustomBackend(Backend):
    name = "sample-custom"

    def attempt(self, task, skill, memory, sample_id=0):
        return "sample output"

    def judge(self, task, response):
        return 1.0, 1.0, "sample passed"

    def reflect(self, failures, successes, skill, memory, **kwargs):
        return []


class TestPluggableSourcesAndBackends(unittest.TestCase):
    def tearDown(self):
        unregister_source("test_custom_src")
        unregister_backend("test_custom_be")

    def test_custom_source_registration_and_harvest(self):
        called = []

        def my_harvester(cfg, **kwargs):
            called.append(True)
            return [SessionDigest(session_id="custom-sess-1", project="test-proj")]

        register_source("test_custom_src", my_harvester)
        self.assertIn("test_custom_src", get_supported_sources())
        self.assertEqual(
            normalize_transcript_source("TEST_CUSTOM_SRC"), "test_custom_src"
        )

        cfg = load_config(transcript_source="test_custom_src")
        digests = harvest_for_config(cfg)
        self.assertTrue(called)
        self.assertEqual(len(digests), 1)
        self.assertEqual(digests[0].session_id, "custom-sess-1")

    def test_entry_point_source_discovery(self):
        ep = mock.MagicMock()
        ep.name = "ep_source"
        ep.load.return_value = sample_harvest

        with mock.patch("skillopt_sleep.harvest_sources.load_entry_point_sources", return_value={"ep_source": ep.load}):
            self.assertIn("ep_source", get_supported_sources())
            self.assertEqual(normalize_transcript_source("ep_source"), "ep_source")
            cfg = load_config(transcript_source="ep_source")
            digests = harvest_for_config(cfg)
            self.assertEqual(len(digests), 1)
            self.assertEqual(digests[0].session_id, "sample-1")

    def test_source_module_invocation(self):
        cfg = load_config(
            source_module="tests.test_sleep_pluggable_registry:sample_harvest"
        )
        digests = harvest_for_config(cfg)
        self.assertEqual(len(digests), 1)
        self.assertEqual(digests[0].session_id, "sample-1")

    def test_unknown_source_raises_informative_error(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_transcript_source("unknown_xyz")
        self.assertIn("Unknown transcript source 'unknown_xyz'", str(ctx.exception))
        self.assertIn("Supported sources:", str(ctx.exception))

    def test_custom_backend_registration(self):
        register_backend("test_custom_be", SampleCustomBackend)
        self.assertIn("test_custom_be", get_supported_backends())

        be = get_backend("test_custom_be")
        self.assertIsInstance(be, SampleCustomBackend)
        self.assertEqual(be.name, "sample-custom")

    def test_entry_point_backend_discovery(self):
        ep = mock.MagicMock()
        ep.name = "ep_backend"
        ep.load.return_value = lambda **kw: SampleCustomBackend()

        with mock.patch("skillopt_sleep.backend.load_entry_point_backends", return_value={"ep_backend": ep.load}):
            self.assertIn("ep_backend", get_supported_backends())
            be = get_backend("ep_backend")
            self.assertIsInstance(be, SampleCustomBackend)

    def test_backend_module_invocation(self):
        be = get_backend(
            "",
            backend_module="tests.test_sleep_pluggable_registry:SampleCustomBackend",
        )
        self.assertIsInstance(be, SampleCustomBackend)

    def test_unknown_backend_raises_informative_error(self):
        with self.assertRaises(ValueError) as ctx:
            get_backend("nonexistent_backend")
        self.assertIn("Unknown backend: 'nonexistent_backend'", str(ctx.exception))
        self.assertIn("Supported backends are:", str(ctx.exception))

    def test_public_harvest_helpers(self):
        # 1. is_meta_prompt
        self.assertTrue(is_meta_prompt("/clear"))
        self.assertTrue(is_meta_prompt("<system-reminder>...</system-reminder>"))
        self.assertFalse(is_meta_prompt("Please write a python test"))

        # 2. detect_feedback
        self.assertTrue(any("neg:" in f for f in detect_feedback("this is still broken")))
        self.assertTrue(any("pos:" in f for f in detect_feedback("looks good thanks")))

        # 3. is_agent_session
        agent_digest = SessionDigest(
            session_id="agent-s1",
            project="test-proj",
            user_prompts=["You are a Claude-Mem observer"],
        )
        user_digest = SessionDigest(
            session_id="user-s1",
            project="test-proj",
            user_prompts=["Help me refactor the database layer"],
        )
        self.assertTrue(is_agent_session(agent_digest))
        self.assertFalse(is_agent_session(user_digest))

        # 4. is_headless_replay
        replay_digest = SessionDigest(
            session_id="replay-s1",
            project="test-proj",
            user_prompts=["## SKILL\nRule A\nNow do task"],
            n_user_turns=1,
        )
        self.assertTrue(is_headless_replay(replay_digest))

        # 5. redact_secrets
        redacted = redact_secrets("sk-ant-api03-abcdef1234567890abcdef1234567890abcdef123456")
        self.assertNotIn("1234567890", redacted)


if __name__ == "__main__":
    unittest.main()

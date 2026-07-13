"""Tests for the MiniMax backend, focusing on optimizer/target deployment
routing (regression for the #116 follow-up: optimizer calls must honour
OPTIMIZER_DEPLOYMENT, not silently reuse TARGET_DEPLOYMENT)."""
from __future__ import annotations

import unittest
from unittest import mock

from skillopt.model import minimax_backend as mm


def _fake_response(_payload, _timeout):
    return {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class TestMiniMaxDeploymentRouting(unittest.TestCase):
    def setUp(self):
        self._saved = (mm.TARGET_DEPLOYMENT, mm.OPTIMIZER_DEPLOYMENT)
        mm.set_target_deployment("target-model")
        mm.set_optimizer_deployment("optimizer-model")

    def tearDown(self):
        mm.TARGET_DEPLOYMENT, mm.OPTIMIZER_DEPLOYMENT = self._saved

    def _captured_model(self, fn, *args, **kwargs):
        seen = {}

        def spy(payload, timeout):
            seen["model"] = payload["model"]
            return _fake_response(payload, timeout)

        with mock.patch.object(mm, "_post_chat_completion", side_effect=spy):
            fn(*args, **kwargs)
        return seen["model"]

    def test_target_text_uses_target_deployment(self):
        self.assertEqual(self._captured_model(mm.chat_target, "sys", "usr"), "target-model")

    def test_optimizer_text_uses_optimizer_deployment(self):
        # The core bug: before the fix this sent "target-model".
        self.assertEqual(self._captured_model(mm.chat_optimizer, "sys", "usr"), "optimizer-model")

    def test_optimizer_messages_uses_optimizer_deployment(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertEqual(self._captured_model(mm.chat_optimizer_messages, msgs), "optimizer-model")

    def test_target_messages_uses_target_deployment(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertEqual(self._captured_model(mm.chat_target_messages, msgs), "target-model")

    def test_optimizer_falls_back_to_target_when_unset(self):
        # Empty optimizer deployment -> setter fills default; explicitly clear it
        # to confirm the `or None` fallback path uses TARGET_DEPLOYMENT.
        mm.OPTIMIZER_DEPLOYMENT = ""
        self.assertEqual(self._captured_model(mm.chat_optimizer, "sys", "usr"), "target-model")


if __name__ == "__main__":
    unittest.main()

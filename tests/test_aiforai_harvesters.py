from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skillopt_sleep.aiforai.harvesters.base import (
    detect_feedback,
    detect_skill_mentions,
    iter_jsonl,
    redact_text,
    within_lookback,
)


class HarvesterBaseTests(unittest.TestCase):
    def test_iter_jsonl_skips_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"a": 1}\nnot-json\n{"b": 2}\n', encoding="utf-8")

            rows = list(iter_jsonl(str(path)))

            self.assertEqual(rows, [{"a": 1}, {"b": 2}])

    def test_detect_feedback_supports_chinese_and_english(self) -> None:
        signals = detect_feedback("这个还是不对, please fix it")

        self.assertIn("neg:还是不对", signals)
        self.assertIn("neg:fix it", signals)

    def test_detect_skill_mentions(self) -> None:
        mentions = detect_skill_mentions("Use $ai-model-rd-protocol for this training run.")

        self.assertEqual(mentions, ["ai-model-rd-protocol"])

    def test_redact_text_masks_secret_like_values(self) -> None:
        redacted = redact_text("OPENAI_API_KEY=sk-abcdef1234567890 token=abc123")

        self.assertIn("OPENAI_API_KEY=<redacted>", redacted)
        self.assertIn("token=<redacted>", redacted)

    def test_within_lookback_accepts_recent_epoch_ms(self) -> None:
        now_ms = 1_800_000_000_000
        recent_ms = now_ms - 60_000
        old_ms = now_ms - 10 * 24 * 3600 * 1000

        self.assertTrue(within_lookback(recent_ms, lookback_days=1, now_ms=now_ms))
        self.assertFalse(within_lookback(old_ms, lookback_days=1, now_ms=now_ms))


if __name__ == "__main__":
    unittest.main()

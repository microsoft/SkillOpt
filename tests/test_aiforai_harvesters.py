from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.harvesters.base import (
    detect_feedback,
    detect_skill_mentions,
    flatten_text,
    iter_jsonl,
    redact_text,
    within_lookback,
)
from skillopt_sleep.aiforai.harvesters.claude import ClaudeHarvester
from skillopt_sleep.aiforai.harvesters.codewhale import CodeWhaleHarvester
from skillopt_sleep.aiforai.harvesters.codex import CodexHarvester


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

    def test_redact_text_masks_structured_secret_values(self) -> None:
        text = '{"token":"abc123", "api_key": "secret456"}\ntoken: abc123\napi_key = secret456'

        redacted = redact_text(text)

        self.assertIn('"token":"<redacted>"', redacted)
        self.assertIn('"api_key": "<redacted>"', redacted)
        self.assertIn("token: <redacted>", redacted)
        self.assertIn("api_key = <redacted>", redacted)

    def test_flatten_text_ignores_empty_text_before_nested_content(self) -> None:
        for value in (
            {"text": None, "content": [{"type": "text", "text": "real"}]},
            {"text": "", "content": [{"type": "text", "text": "real"}]},
        ):
            with self.subTest(value=value):
                self.assertEqual(flatten_text(value), "real")

    def test_within_lookback_accepts_recent_epoch_ms(self) -> None:
        now_ms = 1_800_000_000_000
        recent_ms = now_ms - 60_000
        old_ms = now_ms - 10 * 24 * 3600 * 1000

        self.assertTrue(within_lookback(recent_ms, lookback_days=1, now_ms=now_ms))
        self.assertFalse(within_lookback(old_ms, lookback_days=1, now_ms=now_ms))


class CodexHarvesterTests(unittest.TestCase):
    def _write_codex_thread(
        self,
        *,
        root: Path,
        codex_home: Path,
        session_path: Path,
        events: list[dict[str, object]],
        thread_id: str = "thr1",
    ) -> AiforaiConfig:
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )
        codex_home.mkdir(parents=True, exist_ok=True)
        db_path = codex_home / "state_5.sqlite"
        con = sqlite3.connect(db_path)
        con.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, "
            "created_at_ms INTEGER, updated_at_ms INTEGER, cwd TEXT NOT NULL, "
            "title TEXT NOT NULL, git_branch TEXT, model TEXT, reasoning_effort TEXT, "
            "agent_role TEXT)"
        )
        con.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                thread_id,
                str(session_path),
                1_800_000_000_000,
                1_800_000_001_000,
                "/repo",
                "train",
                "main",
                "gpt-5.5",
                "xhigh",
                "worker",
            ),
        )
        con.commit()
        con.close()
        return AiforaiConfig(
            target_skill_repo=str(root / "AIForAI"),
            codex_home=str(codex_home),
            lookback_days=30,
        )

    def test_codex_harvester_reads_threads_and_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex_home = root / ".codex"
            session_path = codex_home / "sessions/2026/06/09/rollout.jsonl"
            cfg = self._write_codex_thread(
                root=root,
                codex_home=codex_home,
                session_path=session_path,
                events=[
                    {"type": "user_message", "message": "Use ai-model-rd-protocol to plan training"},
                    {"type": "agent_message", "message": "Need training contract"},
                    {"type": "function_call", "name": "exec_command", "arguments": "{}"},
                ],
            )

            sessions = CodexHarvester(now_ms=1_800_000_010_000).harvest(cfg)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].source_agent, "codex")
            self.assertEqual(sessions[0].session_id, "thr1")
            self.assertEqual(sessions[0].cwd, "/repo")
            self.assertEqual(sessions[0].tools_used, ["exec_command"])
            self.assertEqual(sessions[0].skill_mentions, ["ai-model-rd-protocol"])

    def test_codex_harvester_prefers_nested_payload_types_and_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex_home = root / ".codex"
            session_path = codex_home / "sessions/2026/06/10/live-rollout.jsonl"
            cfg = self._write_codex_thread(
                root=root,
                codex_home=codex_home,
                session_path=session_path,
                events=[
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "user_message",
                            "message": "Use ai-model-rd-protocol to plan training",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Need training contract"}],
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": '{"cmd": "python /repo/train.py"}',
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "developer",
                            "content": [{"type": "output_text", "text": "You are Codex"}],
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "system",
                            "content": [{"type": "output_text", "text": "System policy"}],
                        },
                    },
                ],
            )

            sessions = CodexHarvester(now_ms=1_800_000_010_000).harvest(cfg)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(
                sessions[0].user_prompts,
                ["Use ai-model-rd-protocol to plan training"],
            )
            self.assertEqual(sessions[0].assistant_finals, ["Need training contract"])
            self.assertEqual(sessions[0].tools_used, ["exec_command"])
            self.assertEqual(sessions[0].files_touched, ["/repo/train.py"])
            self.assertEqual(sessions[0].skill_mentions, ["ai-model-rd-protocol"])
            self.assertNotIn("You are Codex", sessions[0].user_prompts)
            self.assertNotIn("You are Codex", sessions[0].assistant_finals)
            self.assertNotIn("System policy", sessions[0].user_prompts)
            self.assertNotIn("System policy", sessions[0].assistant_finals)

    def test_codex_harvester_warns_for_rollout_outside_codex_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex_home = root / ".codex"
            session_path = root / "external" / "rollout.jsonl"
            cfg = self._write_codex_thread(
                root=root,
                codex_home=codex_home,
                session_path=session_path,
                events=[
                    {"type": "user_message", "message": "Use ai-model-rd-protocol to plan training"},
                ],
            )

            sessions = CodexHarvester(now_ms=1_800_000_010_000).harvest(cfg)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].event_count, 0)
            self.assertEqual(sessions[0].user_prompts, [])
            self.assertEqual(sessions[0].assistant_finals, [])
            self.assertEqual(sessions[0].tools_used, [])
            self.assertIn(
                f"rollout_path outside codex_home: {session_path}",
                sessions[0].parse_warnings,
            )


class ClaudeHarvesterTests(unittest.TestCase):
    def test_claude_harvester_wraps_project_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claude_home = root / ".claude"
            transcript = claude_home / "projects/proj/session1.jsonl"
            transcript.parent.mkdir(parents=True)
            transcript.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "user",
                                "timestamp": "2026-06-09T00:00:00Z",
                                "cwd": "/repo",
                                "gitBranch": "main",
                                "message": {"role": "user", "content": "Use ai-model-rd-protocol"},
                            }
                        ),
                        json.dumps(
                            {
                                "type": "assistant",
                                "timestamp": "2026-06-09T00:01:00Z",
                                "message": {
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": "final"}],
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = AiforaiConfig(target_skill_repo=str(root / "AIForAI"), claude_home=str(claude_home))

            sessions = ClaudeHarvester().harvest(cfg)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].source_agent, "claude")
            self.assertEqual(sessions[0].cwd, "/repo")
            self.assertEqual(sessions[0].assistant_finals, ["final"])

    def test_claude_harvester_redacts_prompts_and_detects_skill_mentions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claude_home = root / ".claude"
            transcript = claude_home / "projects/proj/session1.jsonl"
            transcript.parent.mkdir(parents=True)
            transcript.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "user",
                                "timestamp": "2026-06-09T00:00:00Z",
                                "cwd": "/repo",
                                "gitBranch": "main",
                                "message": {
                                    "role": "user",
                                    "content": "Use ai-model-rd-protocol OPENAI_API_KEY=sk-abcdef1234567890",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "assistant",
                                "timestamp": "2026-06-09T00:01:00Z",
                                "message": {
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": "final"}],
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = AiforaiConfig(target_skill_repo=str(root / "AIForAI"), claude_home=str(claude_home))

            sessions = ClaudeHarvester().harvest(cfg)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(
                sessions[0].user_prompts,
                ["Use ai-model-rd-protocol OPENAI_API_KEY=<redacted>"],
            )
            self.assertEqual(sessions[0].skill_mentions, ["ai-model-rd-protocol"])
            self.assertNotIn("sk-abcdef1234567890", "\n".join(sessions[0].user_prompts))


class CodeWhaleHarvesterTests(unittest.TestCase):
    def test_codewhale_harvester_reads_runtime_thread_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cw_home = root / ".codewhale"
            ds_home = root / ".deepseek"
            runtime = cw_home / "tasks/runtime"
            (runtime / "threads").mkdir(parents=True)
            (runtime / "events").mkdir(parents=True)
            (runtime / "threads/thr1.json").write_text(
                json.dumps({"id": "thr1", "cwd": "/repo", "created_at": "2026-06-09T00:00:00Z"}),
                encoding="utf-8",
            )
            (runtime / "events/thr1.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"role": "user", "content": "请用 ai-model-rd-protocol 做数据获取计划"}),
                        json.dumps({"role": "assistant", "content": "需要 Data Acquisition Hygiene Gate"}),
                        json.dumps({"tool": "mcp_k8s-management_run_task"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = AiforaiConfig(
                target_skill_repo=str(root / "AIForAI"),
                codewhale_home=str(cw_home),
                deepseek_home=str(ds_home),
            )

            sessions = CodeWhaleHarvester().harvest(cfg)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].source_agent, "codewhale")
            self.assertEqual(sessions[0].session_id, "thr1")
            self.assertIn("mcp_k8s-management_run_task", sessions[0].tools_used)
            self.assertEqual(sessions[0].skill_mentions, ["ai-model-rd-protocol"])

    def test_codewhale_harvester_uses_deepseek_fallback_when_runtime_has_no_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cw_home = root / ".codewhale"
            ds_home = root / ".deepseek"
            runtime = cw_home / "tasks/runtime"
            (runtime / "threads").mkdir(parents=True)
            (ds_home / "sessions").mkdir(parents=True)
            (runtime / "threads/thr1.json").write_text(
                json.dumps({"id": "thr1", "cwd": "/repo", "created_at": "2026-06-09T00:00:00Z"}),
                encoding="utf-8",
            )
            deepseek_path = ds_home / "sessions/thr1.json"
            deepseek_path.write_text(
                json.dumps(
                    {
                        "id": "thr1",
                        "cwd": "/repo",
                        "created_at": "2026-06-09T00:00:00Z",
                        "messages": [
                            {"role": "user", "content": "Use ai-model-rd-protocol for planning"},
                            {"role": "assistant", "content": "DeepSeek fallback content"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cfg = AiforaiConfig(
                target_skill_repo=str(root / "AIForAI"),
                codewhale_home=str(cw_home),
                deepseek_home=str(ds_home),
            )

            sessions = CodeWhaleHarvester().harvest(cfg)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].source_agent, "codewhale")
            self.assertEqual(sessions[0].session_id, "thr1")
            self.assertEqual(sessions[0].raw_path, str(deepseek_path))
            self.assertEqual(sessions[0].user_prompts, ["Use ai-model-rd-protocol for planning"])
            self.assertEqual(sessions[0].assistant_finals, ["DeepSeek fallback content"])
            self.assertEqual(sessions[0].skill_mentions, ["ai-model-rd-protocol"])
            self.assertEqual(sessions[0].event_count, 2)
            self.assertEqual(sessions[0].parse_warnings, [])


if __name__ == "__main__":
    unittest.main()

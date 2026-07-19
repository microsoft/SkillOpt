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
from unittest import mock

from skillopt_sleep.backend import MockBackend, exact_score, keyword_soft_score
from skillopt_sleep.config import load_config
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.experiments.personas import programmer_persona, researcher_persona
from skillopt_sleep.harvest import _detect_feedback, _is_meta_prompt, digest_transcript
from skillopt_sleep.memory import apply_edits, current_learned_lines, extract_learned, set_learned
from skillopt_sleep.mine import assign_splits, filter_tasks_for_target, heuristic_mine, mine
from skillopt_sleep.staging import adopt
from skillopt_sleep.types import EditRecord, SessionDigest, SleepReport, TaskRecord


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

    def _write_jsonl(self, path, records):
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def test_digest_codex_archived_session_sanitizes_and_skips_meta(self):
        from skillopt_sleep.harvest_codex import digest_codex_archived_session

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rollout-example.jsonl")
            self._write_jsonl(path, [
                {"type": "turn_context", "timestamp": "2026-06-12T10:00:00Z",
                 "payload": {"cwd": "/repo/Yoshi", "type": None}},
                {"type": "response_item", "timestamp": "2026-06-12T10:00:01Z",
                 "payload": {"type": "message", "role": "developer",
                             "content": [{"type": "text", "text": "do not copy"}]}},
                {"type": "response_item", "timestamp": "2026-06-12T10:00:02Z",
                 "payload": {"type": "user_message",
                             "message": "# AGENTS.md instructions for /repo/Yoshi\n"
                                        "<INSTRUCTIONS>do not keep</INSTRUCTIONS>"}},
                {"type": "response_item", "timestamp": "2026-06-12T10:00:03Z",
                 "payload": {"type": "user_message",
                             "message": "run deploy with sk-1234567890abcdef and token local-secret"}},
                {"type": "response_item", "timestamp": "2026-06-12T10:00:04Z",
                 "payload": {"type": "function_call", "name": "exec_command",
                             "arguments": "raw args should not copy"}},
                {"type": "response_item", "timestamp": "2026-06-12T10:00:05Z",
                 "payload": {"type": "function_call_output",
                             "output": "raw output should not copy"}},
                {"type": "response_item", "timestamp": "2026-06-12T10:00:06Z",
                 "payload": {"type": "agent_message", "message": "done"}},
            ])

            digest = digest_codex_archived_session(path, project="/repo/Yoshi")

        self.assertIsNotNone(digest)
        joined = "\n".join(digest.user_prompts + digest.assistant_finals)
        self.assertEqual(digest.project, "/repo/Yoshi")
        self.assertIn("[REDACTED_OPENAI_KEY]", joined)
        self.assertIn("token [REDACTED]", joined)
        self.assertIn("exec_command", digest.tools_used)
        self.assertNotIn("AGENTS.md instructions", joined)
        self.assertNotIn("do not copy", joined)
        self.assertNotIn("raw args should not copy", joined)
        self.assertNotIn("raw output should not copy", joined)

    def test_digest_cursor_transcript_redacts_and_keeps_only_message_text_and_tool_names(self):
        from skillopt_sleep.harvest_cursor import digest_cursor_transcript

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cursor-session.jsonl")
            self._write_jsonl(path, [
                {
                    "role": "user",
                    "message": {
                        "content": [{
                            "type": "text",
                            "text": (
                                "<attached_files>never-copy-attachment-metadata</attached_files>\n"
                                "<user_query>\n"
                                "Deploy with sk-1234567890abcdef and token=local-secret\n"
                                "</user_query>"
                            ),
                        }],
                    },
                },
                {
                    "role": "assistant",
                    "message": {
                        "content": [{
                            "type": "tool_use",
                            "name": "shell.execute",
                            "input": {"token": "never-copy-tool-arguments"},
                        }],
                    },
                },
                {
                    "role": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "Deployment finished."},
                            {
                                "type": "tool_use",
                                "name": "read_file",
                                "input": {"path": "never-copy-tool-arguments"},
                            },
                        ],
                    },
                },
                {"type": "tool_result", "output": "never-copy-tool-output"},
                {"type": "turn_ended", "status": "error"},
            ])
            with open(path, "a", encoding="utf-8") as f:
                f.write("null\n")
                f.write("[]\n")
                f.write('"non-object"\n')
                f.write("{malformed jsonl record\\n")

            digest = digest_cursor_transcript(path, project="/repo/Cursor Project")

        self.assertIsNotNone(digest)
        joined = "\n".join(digest.user_prompts + digest.assistant_finals)
        self.assertEqual(digest.project, "/repo/Cursor Project")
        self.assertEqual(len(digest.user_prompts), 1)
        self.assertIn("[REDACTED_OPENAI_KEY]", joined)
        self.assertIn("token=[REDACTED]", joined)
        self.assertEqual(digest.tools_used, ["shell.execute", "read_file"])
        self.assertIn("neg:cursor_turn_error", digest.feedback_signals)
        self.assertNotIn("never-copy-tool-arguments", joined)
        self.assertNotIn("never-copy-tool-output", joined)
        self.assertNotIn("never-copy-attachment-metadata", joined)

    def test_harvest_cursor_scopes_orders_filters_mtime_and_skips_replays(self):
        from skillopt_sleep.__main__ import _cfg_from_args
        from skillopt_sleep.harvest_cursor import (
            CURSOR_REPLAY_SENTINEL,
            cursor_project_slug,
            harvest_cursor,
        )
        from skillopt_sleep.harvest_sources import harvest_for_config

        def write_cursor_session(cursor_home, project, session_id, prompt, mtime, extra_prompt=""):
            project_dir = os.path.join(
                cursor_home,
                "projects",
                cursor_project_slug(project),
            )
            path = os.path.join(
                project_dir,
                "agent-transcripts",
                session_id,
                f"{session_id}.jsonl",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(os.path.join(project_dir, ".workspace-trusted"), "w", encoding="utf-8") as f:
                json.dump({"workspacePath": project}, f)
            records = [
                {
                    "role": "user",
                    "message": {
                        "content": [{
                            "type": "text",
                            "text": f"<user_query>\n{prompt}\n</user_query>",
                        }],
                    },
                },
                {
                    "role": "assistant",
                    "message": {"content": [{"type": "text", "text": "done"}]},
                },
            ]
            if extra_prompt:
                records.extend([
                    {
                        "role": "user",
                        "message": {
                            "content": [{
                                "type": "text",
                                "text": f"<user_query>\n{extra_prompt}\n</user_query>",
                            }],
                        },
                    },
                    {
                        "role": "assistant",
                        "message": {"content": [{"type": "text", "text": "done again"}]},
                    },
                ])
            self._write_jsonl(path, records)
            os.utime(path, (mtime, mtime))
            return path

        with tempfile.TemporaryDirectory() as tmp:
            cursor_home = os.path.join(tmp, ".cursor")
            project = os.path.join(tmp, "project with spaces")
            other_project = os.path.join(tmp, "other project")
            old_time = 1_700_000_000
            new_time = old_time + 3_600
            main_path = write_cursor_session(
                cursor_home,
                project,
                "older",
                "fix the first issue",
                old_time,
            )
            subagent_path = os.path.join(os.path.dirname(main_path), "subagents", "worker.jsonl")
            os.makedirs(os.path.dirname(subagent_path), exist_ok=True)
            self._write_jsonl(subagent_path, [
                {"role": "user", "message": {"content": "machine-generated subagent task"}},
                {"role": "assistant", "message": {"content": "subagent result"}},
            ])
            write_cursor_session(cursor_home, other_project, "newer", "fix the second issue", new_time)
            write_cursor_session(
                cursor_home,
                other_project,
                "generated-replay",
                CURSOR_REPLAY_SENTINEL + "\n## CURRENT SKILL",
                new_time + 1,
                extra_prompt="continue the internal replay",
            )

            invoked = harvest_cursor(
                os.path.join(cursor_home, "projects"),
                scope="invoked",
                invoked_project=os.path.join(project, "src", "package"),
            )
            all_digests = harvest_cursor(
                os.path.join(cursor_home, "projects"),
                scope="all",
                since_iso="2023-11-14T23:00:00Z",
                limit=1,
            )

            Args = type("Args", (), {
                "project": project,
                "scope": "",
                "backend": "cursor",
                "model": "",
                "codex_path": "",
                "cursor_path": "",
                "claude_home": "",
                "codex_home": "",
                "cursor_home": cursor_home,
                "source": "cursor",
                "lookback_hours": 0,
                "edit_budget": 0,
                "max_sessions": 0,
                "max_tasks": 0,
                "target_skill_path": "",
                "preferences": "",
                "progress": False,
                "auto_adopt": False,
            })
            cfg = _cfg_from_args(Args())
            configured = harvest_for_config(cfg)

        self.assertEqual([d.session_id for d in invoked], ["older"])
        self.assertEqual([d.session_id for d in all_digests], ["newer"])
        self.assertEqual(invoked[0].project, project)
        self.assertEqual(all_digests[0].project, other_project)
        self.assertEqual([d.session_id for d in configured], ["older"])
        self.assertEqual(cfg.get("transcript_source"), "cursor")
        self.assertEqual(cfg.get("backend"), "cursor")

    def test_harvest_cursor_prefers_longest_workspace_and_falls_back_to_slug(self):
        from skillopt_sleep.harvest_cursor import cursor_project_slug, harvest_cursor

        def write_session(projects_dir, storage_name, workspace, session_id):
            project_dir = os.path.join(projects_dir, storage_name)
            path = os.path.join(project_dir, "agent-transcripts", session_id, f"{session_id}.jsonl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._write_jsonl(path, [
                {"role": "user", "message": {"content": "please fix this project"}},
                {"role": "assistant", "message": {"content": "fixed"}},
            ])
            if workspace is not None:
                with open(os.path.join(project_dir, ".workspace-trusted"), "w", encoding="utf-8") as f:
                    json.dump(workspace, f)
            return path

        with tempfile.TemporaryDirectory() as tmp:
            projects_dir = os.path.join(tmp, ".cursor", "projects")
            parent = os.path.join(tmp, "repo")
            nested = os.path.join(parent, "packages", "app")
            write_session(projects_dir, "parent-store", {"workspacePath": parent}, "parent")
            write_session(projects_dir, "nested-store", {"workspacePath": nested}, "nested")
            fallback = os.path.join(tmp, "fallback")
            write_session(
                projects_dir,
                cursor_project_slug(fallback),
                ["invalid metadata shape"],
                "fallback",
            )
            metadata_free = os.path.join(tmp, "metadata-free")
            write_session(
                projects_dir,
                cursor_project_slug(metadata_free),
                None,
                "metadata-free",
            )
            mixed_parent = os.path.join(tmp, "mixed-parent")
            mixed_nested = os.path.join(mixed_parent, "nested")
            write_session(projects_dir, "mixed-parent-store", {"workspacePath": mixed_parent}, "mixed-parent")
            write_session(
                projects_dir,
                cursor_project_slug(mixed_nested),
                None,
                "mixed-nested",
            )

            nested_digests = harvest_cursor(
                projects_dir,
                scope="invoked",
                invoked_project=os.path.join(nested, "src"),
            )
            fallback_digests = harvest_cursor(
                projects_dir,
                scope="invoked",
                invoked_project=fallback,
            )
            metadata_free_digests = harvest_cursor(
                projects_dir,
                scope="invoked",
                invoked_project=os.path.join(metadata_free, "packages", "app"),
            )
            mixed_digests = harvest_cursor(
                projects_dir,
                scope="invoked",
                invoked_project=mixed_nested,
            )

        self.assertEqual([digest.session_id for digest in nested_digests], ["nested"])
        self.assertEqual(nested_digests[0].project, nested)
        self.assertEqual([digest.session_id for digest in fallback_digests], ["fallback"])
        self.assertEqual(fallback_digests[0].project, fallback)
        self.assertEqual(
            [digest.session_id for digest in metadata_free_digests],
            ["metadata-free"],
        )
        self.assertEqual(metadata_free_digests[0].project, metadata_free)
        self.assertEqual([digest.session_id for digest in mixed_digests], ["mixed-nested"])
        self.assertEqual(mixed_digests[0].project, mixed_nested)

    def test_harvest_cursor_uses_numeric_mtime_for_aware_and_local_cutoffs(self):
        from datetime import datetime, timedelta, timezone

        from skillopt_sleep.harvest_cursor import cursor_project_slug, harvest_cursor

        with tempfile.TemporaryDirectory() as tmp:
            project = os.path.join(tmp, "project")
            project_dir = os.path.join(tmp, ".cursor", "projects", cursor_project_slug(project))
            os.makedirs(project_dir)
            with open(os.path.join(project_dir, ".workspace-trusted"), "w", encoding="utf-8") as f:
                json.dump({"workspacePath": project}, f)

            cutoff = 1_700_000_000
            for session_id, modified in (("before", cutoff - 1), ("equal", cutoff), ("after", cutoff + 1)):
                path = os.path.join(
                    project_dir,
                    "agent-transcripts",
                    session_id,
                    f"{session_id}.jsonl",
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self._write_jsonl(path, [
                    {"role": "user", "message": {"content": f"task {session_id}"}},
                    {"role": "assistant", "message": {"content": "done"}},
                ])
                os.utime(path, (modified, modified))

            aware = datetime.fromtimestamp(cutoff, timezone(timedelta(hours=5))).isoformat()
            local = datetime.fromtimestamp(cutoff).replace(microsecond=0).isoformat()
            aware_result = harvest_cursor(projects_dir=os.path.dirname(project_dir), since_iso=aware)
            local_result = harvest_cursor(projects_dir=os.path.dirname(project_dir), since_iso=local)

        self.assertEqual([digest.session_id for digest in aware_result], ["after"])
        self.assertEqual([digest.session_id for digest in local_result], ["after"])

    def test_harvest_cursor_filters_only_exact_internal_replay_sentinel(self):
        from skillopt_sleep.harvest_cursor import CURSOR_REPLAY_SENTINEL, cursor_project_slug, harvest_cursor

        with tempfile.TemporaryDirectory() as tmp:
            project = os.path.join(tmp, "project")
            project_dir = os.path.join(tmp, ".cursor", "projects", cursor_project_slug(project))
            for session_id, prompt in (
                ("internal", CURSOR_REPLAY_SENTINEL + "\nrun replay"),
                ("real", f"Please explain what {CURSOR_REPLAY_SENTINEL} means"),
                ("grader", "You are a strict grader helping me review this response"),
                ("skill", "Please explain the ## CURRENT SKILL section"),
            ):
                path = os.path.join(project_dir, "agent-transcripts", session_id, f"{session_id}.jsonl")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self._write_jsonl(path, [
                    {"role": "user", "message": {"content": prompt}},
                    {"role": "assistant", "message": {"content": "answer"}},
                ])

            digests = harvest_cursor(os.path.join(tmp, ".cursor", "projects"), scope="all")

        self.assertEqual(
            sorted(digest.session_id for digest in digests),
            ["grader", "real", "skill"],
        )

    def test_auto_source_keeps_existing_codex_then_claude_precedence(self):
        from skillopt_sleep.harvest_sources import harvest_for_config

        cfg = load_config(transcript_source="auto", invoked_project="/repo/project")
        expected = [SessionDigest(session_id="claude-session", project="/repo/project")]
        with mock.patch("skillopt_sleep.harvest_sources.harvest_codex", return_value=[]), \
             mock.patch("skillopt_sleep.harvest_sources.harvest", return_value=expected), \
             mock.patch("skillopt_sleep.harvest_sources.harvest_cursor") as cursor_harvest:
            self.assertEqual(harvest_for_config(cfg), expected)

        cursor_harvest.assert_not_called()

    def test_harvest_codex_filters_project_and_cli_source(self):
        from skillopt_sleep.__main__ import _cfg_from_args
        from skillopt_sleep.harvest_sources import harvest_for_config

        with tempfile.TemporaryDirectory() as tmp:
            codex_home = os.path.join(tmp, ".codex")
            sessions = os.path.join(codex_home, "archived_sessions")
            os.makedirs(sessions)
            self._write_jsonl(os.path.join(sessions, "rollout-yoshi.jsonl"), [
                {"type": "turn_context", "timestamp": "2026-06-12T10:00:00Z",
                 "payload": {"cwd": "/repo/Yoshi", "type": None}},
                {"type": "response_item", "timestamp": "2026-06-12T10:00:01Z",
                 "payload": {"type": "user_message", "message": "fix Yoshi"}},
                {"type": "response_item", "timestamp": "2026-06-12T10:00:02Z",
                 "payload": {"type": "agent_message", "message": "fixed"}},
            ])
            self._write_jsonl(os.path.join(sessions, "rollout-other.jsonl"), [
                {"type": "turn_context", "timestamp": "2026-06-12T10:00:00Z",
                 "payload": {"cwd": "/repo/Other", "type": None}},
                {"type": "response_item", "timestamp": "2026-06-12T10:00:01Z",
                 "payload": {"type": "user_message", "message": "fix Other"}},
            ])

            Args = type("Args", (), {
                "project": "/repo/Yoshi",
                "scope": "",
                "backend": "",
                "model": "",
                "codex_path": "",
                "claude_home": "",
                "codex_home": codex_home,
                "source": "codex",
                "lookback_hours": 0,
                "edit_budget": 0,
                "auto_adopt": False,
            })

            cfg = _cfg_from_args(Args())
            digests = harvest_for_config(cfg, limit=10)

        self.assertEqual(cfg.get("transcript_source"), "codex")
        self.assertEqual(len(digests), 1)
        self.assertEqual(digests[0].session_id, "rollout-yoshi")
        self.assertEqual(digests[0].user_prompts, ["fix Yoshi"])

    def test_cli_exposes_limits_progress_and_target_skill_path(self):
        from skillopt_sleep.__main__ import _cfg_from_args

        with tempfile.TemporaryDirectory() as project:
            Args = type("Args", (), {
                "project": project,
                "scope": "",
                "backend": "codex",
                "model": "",
                "codex_path": "",
                "claude_home": "",
                "codex_home": "",
                "source": "codex",
                "lookback_hours": 0,
                "edit_budget": 2,
                "max_sessions": 5,
                "max_tasks": 3,
                "target_skill_path": ".agents/skills/taste-skill/SKILL.md",
                "preferences": "Always use async/await",
                "progress": True,
                "auto_adopt": False,
            })

            cfg = _cfg_from_args(Args())

            self.assertEqual(cfg.get("backend"), "codex")
            self.assertEqual(cfg.get("preferences"), "Always use async/await")
            self.assertEqual(cfg.get("max_sessions_per_night"), 5)
            self.assertEqual(cfg.get("max_tasks_per_night"), 3)
            self.assertTrue(cfg.get("progress"))
            self.assertEqual(
                cfg.managed_skill_path(),
                os.path.abspath(os.path.join(project, ".agents/skills/taste-skill/SKILL.md")),
            )

    def test_cli_report_payload_includes_rejected_edits(self):
        from skillopt_sleep.__main__ import _report_payload

        report = SleepReport(
            night=1,
            project="/p",
            edits=[EditRecord("skill", "add", "accepted rule")],
            rejected_edits=[EditRecord("skill", "add", "rejected rule")],
        )
        outcome = type("Outcome", (), {"staging_dir": "", "adopted": False})()

        payload = _report_payload(report, outcome)

        self.assertEqual(payload["n_accepted_edits"], 1)
        self.assertEqual(payload["n_rejected_edits"], 1)
        self.assertEqual(payload["rejected_edits"][0]["content"], "rejected rule")

    def test_tasks_file_roundtrip_and_split_assignment(self):
        from skillopt_sleep.tasks_file import load_tasks_file, make_tasks_payload, write_tasks_file

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tasks.json")
            payload = make_tasks_payload(
                [
                    TaskRecord(id="t1", project="/p", intent="configure MCP server"),
                    TaskRecord(id="t2", project="/p", intent="resolve Git conflict"),
                ],
                project="/p",
                transcript_source="codex",
                n_sessions=2,
                target_skill_path="/p/.agents/skills/yoshi-monorepo/SKILL.md",
            )

            written = write_tasks_file(path, payload)
            tasks, meta = load_tasks_file(written, holdout_fraction=0.5, seed=1)

        self.assertEqual(meta["target_skill_path"], "/p/.agents/skills/yoshi-monorepo/SKILL.md")
        self.assertEqual([t.id for t in tasks], ["t1", "t2"])
        self.assertIn("val", {t.split for t in tasks})

    def test_cfg_uses_tasks_file_target_skill_path_metadata(self):
        from skillopt_sleep.__main__ import _cfg_from_args

        Args = type("Args", (), {
            "project": "/repo/Yoshi",
            "scope": "",
            "backend": "",
            "model": "",
            "codex_path": "",
            "claude_home": "",
            "codex_home": "",
            "source": "",
            "lookback_hours": 0,
            "edit_budget": 0,
            "max_sessions": 0,
            "max_tasks": 0,
            "target_skill_path": "",
            "progress": False,
            "auto_adopt": False,
        })

        cfg = _cfg_from_args(Args(), task_meta={
            "target_skill_path": ".agents/skills/yoshi-monorepo/SKILL.md",
        })

        self.assertEqual(
            cfg.managed_skill_path(),
            os.path.abspath("/repo/Yoshi/.agents/skills/yoshi-monorepo/SKILL.md"),
        )

    def test_cmd_run_uses_tasks_file_without_harvest(self):
        from contextlib import redirect_stdout
        from io import StringIO

        from skillopt_sleep.__main__ import cmd_run
        from skillopt_sleep.tasks_file import make_tasks_payload, write_tasks_file

        with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as home:
            target = os.path.join(project, ".agents/skills/yoshi-monorepo/SKILL.md")
            os.makedirs(os.path.dirname(target))
            with open(target, "w", encoding="utf-8") as f:
                f.write("# Yoshi Monorepo\n")
            tasks_path = os.path.join(home, "reviewed-tasks.json")
            write_tasks_file(
                tasks_path,
                make_tasks_payload(
                    [
                        TaskRecord(id="t1", project=project, intent="configure MCP server"),
                        TaskRecord(id="t2", project=project, intent="resolve Git conflict"),
                    ],
                    project=project,
                    n_sessions=2,
                    target_skill_path=target,
                ),
            )
            Args = type("Args", (), {
                "project": project,
                "scope": "",
                "backend": "mock",
                "model": "",
                "codex_path": "",
                "claude_home": os.path.join(home, ".claude"),
                "codex_home": "",
                "source": "",
                "lookback_hours": 0,
                "edit_budget": 2,
                "max_sessions": 5,
                "max_tasks": 3,
                "target_skill_path": "",
                "tasks_file": tasks_path,
                "progress": False,
                "auto_adopt": False,
                "json": True,
            })

            out = StringIO()
            with redirect_stdout(out):
                rc = cmd_run(Args(), dry=True)
            payload = json.loads(out.getvalue())

        self.assertEqual(rc, 0)
        self.assertEqual(payload["n_sessions"], 0)
        self.assertEqual(payload["n_tasks"], 2)
        self.assertEqual(payload["tasks_file"], tasks_path)

    def test_cmd_run_refuses_unreviewed_tasks_file_for_real_backend(self):
        from contextlib import redirect_stderr
        from io import StringIO

        from skillopt_sleep.__main__ import cmd_run
        from skillopt_sleep.tasks_file import make_tasks_payload, write_tasks_file

        with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as home:
            tasks_path = os.path.join(home, "reviewed-tasks.json")
            write_tasks_file(
                tasks_path,
                make_tasks_payload(
                    [TaskRecord(id="t1", project=project, intent="configure MCP server")],
                    project=project,
                    target_skill_path=os.path.join(project, ".agents/skills/yoshi-monorepo/SKILL.md"),
                ),
            )
            Args = type("Args", (), {
                "project": project,
                "scope": "",
                "backend": "codex",
                "model": "",
                "codex_path": "",
                "claude_home": os.path.join(home, ".claude"),
                "codex_home": "",
                "source": "",
                "lookback_hours": 0,
                "edit_budget": 2,
                "max_sessions": 0,
                "max_tasks": 0,
                "target_skill_path": "",
                "tasks_file": tasks_path,
                "progress": False,
                "auto_adopt": False,
                "json": True,
            })

            err = StringIO()
            with redirect_stderr(err):
                rc = cmd_run(Args(), dry=True)

        self.assertEqual(rc, 2)
        self.assertIn("unreviewed tasks file", err.getvalue())


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

    def test_target_filter_prefers_matching_skill_terms(self):
        skill = """# Yoshi Monorepo

## MCP Setup Requests
Configure Codex MCP servers from linked setup docs.

## Local Git Conflicts
Resolve local Git conflicts during merge, rebase, or cherry-pick.
"""
        tasks = [
            TaskRecord(id="ios", project="/p", intent="polish SwiftUI onboarding spacing"),
            TaskRecord(id="mcp", project="/p", intent="configure an MCP server from docs"),
            TaskRecord(id="git", project="/p", intent="resolve a local Git conflict"),
            TaskRecord(id="api", project="/p", intent="deploy the Rails API with Kamal"),
        ]

        filtered = filter_tasks_for_target(
            tasks,
            skill,
            ".agents/skills/yoshi-monorepo/SKILL.md",
        )

        self.assertEqual({t.id for t in filtered}, {"mcp", "git"})

    def test_mine_oversamples_before_target_filtering(self):
        skill = """# Yoshi Monorepo

## MCP Setup Requests
Configure Codex MCP servers.

## Local Git Conflicts
Resolve local Git conflicts.
"""
        digests = [
            self._digest(["polish SwiftUI onboarding spacing"], ["neg:missed"]),
            self._digest(["configure an MCP server from docs"], ["neg:missed"]),
            self._digest(["resolve a local Git conflict"], ["neg:missed"]),
        ]

        tasks = mine(
            digests,
            max_tasks=2,
            candidate_limit=3,
            target_skill_text=skill,
            target_skill_path=".agents/skills/yoshi-monorepo/SKILL.md",
            seed=42,
        )

        self.assertEqual({t.intent for t in tasks}, {
            "configure an MCP server from docs",
            "resolve a local Git conflict",
        })

    def test_cursor_miner_failure_is_not_swallowed(self):
        from skillopt_sleep.backend import CursorBackendError

        def failed_miner(_digests):
            raise CursorBackendError("Cursor Agent authentication failed")

        with self.assertRaises(CursorBackendError):
            mine(
                [self._digest(["configure an MCP server"], ["neg:failed"])],
                llm_miner=failed_miner,
            )


class TestConsolidateGate(unittest.TestCase):
    def test_accepts_helpful_rejects_harmful(self):
        be = MockBackend()
        tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)
        res = consolidate(be, tasks, set_learned("", []), "", edit_budget=4,
                          gate_metric="mixed", night=1)
        self.assertTrue(res.accepted)
        self.assertGreater(res.candidate_score, res.baseline_score)

    def test_consolidate_records_holdout_detail(self):
        # observability: a 0.0 night must carry per-task evidence (was empty
        # response vs failing checks?) so it is diagnosable, not a black box.
        be = MockBackend()
        tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)
        res = consolidate(be, tasks, set_learned("", []), "", edit_budget=4,
                          gate_metric="mixed", night=1)
        self.assertTrue(res.holdout_detail)  # non-empty per-task rows
        row = res.holdout_detail[0]
        for k in ("id", "hard", "soft", "response_len", "why"):
            self.assertIn(k, row)

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
        from skillopt_sleep.types import ReplayResult
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
        from skillopt_sleep.types import ReplayResult
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

    def test_reflect_records_last_raw(self):
        # the optimizer's raw reply must be retained so a no-edits night is
        # diagnosable (empty/non-JSON reflect vs genuinely no failures).
        from skillopt_sleep.backend import CliBackend
        from skillopt_sleep.types import ReplayResult

        class CapBackend(CliBackend):
            name = "cap"
            def _call(self, prompt, *, max_tokens=1024):
                return '[{"op":"add","content":"a learned rule","rationale":"x"}]'

        be = CapBackend()
        t = TaskRecord(id="t", project="/p", intent="x", reference_kind="rule",
                       judge={"checks": [{"op": "contains", "arg": "z"}]})
        be.reflect([(t, ReplayResult(id="t", hard=0.0, fail_reason="failed: contains=z"))],
                   [], "skill", "", edit_budget=2, evolve_skill=True, evolve_memory=False)
        self.assertIn("a learned rule", be.last_reflect_raw)

    def test_replay_records_cost(self):
        from skillopt_sleep.backend import MockBackend
        from skillopt_sleep.replay import replay_one
        t = TaskRecord(id="t", project="/p", intent="hello world",
                       reference_kind="exact", reference="hi")
        r = replay_one(MockBackend(), t, "some skill text", "")
        self.assertGreater(r.tokens, 0)
        self.assertGreaterEqual(r.latency_ms, 0.0)


class TestCodexBackend(unittest.TestCase):
    def test_codex_cli_backend_runs_exec_in_project_dir(self):
        from skillopt_sleep.backend import CodexCliBackend

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            out_path = cmd[cmd.index("-o") + 1]
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("ok")

            class Proc:
                returncode = 0
                stdout = ""
                stderr = ""

            return Proc()

        with tempfile.TemporaryDirectory() as project:
            expected_project = os.path.abspath(project)
            backend = CodexCliBackend(codex_path="codex", project_dir=project)

            with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
                self.assertEqual(backend._call("hello"), "ok")

            self.assertEqual(len(calls), 1)
            cmd, kwargs = calls[0]
            self.assertEqual(kwargs["cwd"], expected_project)
            self.assertIn("-C", cmd)
            self.assertEqual(cmd[cmd.index("-C") + 1], expected_project)

    def test_codex_call_retries_transient_failure_not_silent_zero(self):
        """A transient timeout must be RETRIED, not silently returned as "" — an
        empty reply scores 0 on every judge and zeroes the held-out baseline,
        making a flaky backend look identical to 'nothing to learn'."""
        import subprocess as _sp

        from skillopt_sleep.backend import CodexCliBackend

        calls = {"n": 0}

        def fake_run(cmd, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _sp.TimeoutExpired(cmd, kwargs.get("timeout", 1))
            out_path = cmd[cmd.index("-o") + 1]
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("real answer")

            class Proc:
                returncode = 0
                stdout = ""
                stderr = ""

            return Proc()

        backend = CodexCliBackend(codex_path="codex")
        with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run), \
             mock.patch("time.sleep", lambda *_a, **_k: None):
            out = backend._call("hello")
        self.assertEqual(out, "real answer")     # recovered on retry
        self.assertGreaterEqual(calls["n"], 2)   # proves it did not silently return "" once

    def test_codex_auth_error_surfaces_not_scored_as_response(self):
        """An auth 401 must become a clear last_call_error + EMPTY response (not the
        9k-char error text scored as a 0 'answer'), and must NOT be retried — the
        exact failure that silently stalled learning (refresh_token_reused)."""
        from skillopt_sleep.backend import CodexCliBackend

        calls = {"n": 0}

        def fake_run(cmd, **kwargs):
            calls["n"] += 1
            out_path = cmd[cmd.index("-o") + 1]
            open(out_path, "w").close()  # empty output file (codex wrote nothing)

            class Proc:
                returncode = 1
                stdout = ""
                stderr = "ERROR codex_core::auth: 401 Unauthorized: refresh_token_reused"

            return Proc()

        be = CodexCliBackend(codex_path="codex")
        with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run), \
             mock.patch("time.sleep", lambda *_a, **_k: None):
            out = be._call("hi")
        self.assertEqual(out, "")                                   # NOT the error text
        self.assertIn("refresh_token_reused", be.last_call_error)   # surfaced for the operator
        self.assertEqual(calls["n"], 1)                             # failed fast, no wasted retries

    def test_codex_attempt_with_tools_surfaces_error_not_silent(self):
        """A failed tool-rollout (non-zero codex exec) on the tool path must set
        last_call_error and return an empty response — not a silent empty->0 the
        diagnostics can't see (the gap a _call-only fix would otherwise leave)."""
        from skillopt_sleep.backend import CodexCliBackend

        def fake_run(cmd, **kwargs):
            class Proc:
                returncode = 1
                stdout = ""
                stderr = "ERROR codex_core::auth: 401 Unauthorized: refresh_token_reused"
            return Proc()  # writes nothing to out_path -> empty response

        be = CodexCliBackend(codex_path="codex")
        task = TaskRecord(id="t", project="/p", intent="answer the question",
                          reference_kind="rule",
                          judge={"checks": [{"op": "tool_called", "arg": "search"}]})
        with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
            resp, called = be.attempt_with_tools(task, "", "", ["search"])
        self.assertEqual(resp, "")                     # no leaked error text as a "response"
        self.assertIn("exited 1", be.last_call_error)  # failure surfaced for diagnostics
        self.assertEqual(called, [])                   # no tool actually ran

    def test_codex_resolve_path_windows(self):
        from skillopt_sleep.backend import resolve_codex_path
        with mock.patch("sys.platform", "win32"), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch.dict("os.environ", {
                 "APPDATA": r"C:\Users\Sparsh\AppData\Roaming",
                 "USERPROFILE": r"C:\Users\Sparsh",
                 "NVM_HOME": r"C:\Users\Sparsh\nvm"
             }), \
             mock.patch("os.path.exists", return_value=True):
            path = resolve_codex_path("")
            self.assertEqual(path, r"C:\Users\Sparsh\AppData\Roaming\npm\codex.cmd")

    def test_codex_attempt_with_tools_windows(self):
        from skillopt_sleep.backend import CodexCliBackend
        be = CodexCliBackend(codex_path="codex")
        task = TaskRecord(id="t", project="/p", intent="answer the question",
                          reference_kind="rule",
                          judge={"checks": [{"op": "tool_called", "arg": "search"}]})
        calls = []
        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            class Proc:
                returncode = 0
                stdout = ""
                stderr = ""
            return Proc()

        with mock.patch("os.name", "nt"), \
             mock.patch("shutil.rmtree"), \
             mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
            orig_mkdtemp = tempfile.mkdtemp
            temp_dirs = []
            def fake_mkdtemp(*args, **kwargs):
                d = orig_mkdtemp(*args, **kwargs)
                temp_dirs.append(d)
                return d
            with mock.patch("tempfile.mkdtemp", side_effect=fake_mkdtemp):
                be.attempt_with_tools(task, "", "", ["search"])
            
            self.assertEqual(len(temp_dirs), 1)
            work_dir = temp_dirs[0]
            shim_path = os.path.join(work_dir, "search.cmd")
            try:
                self.assertTrue(os.path.exists(shim_path))
                with open(shim_path, "r") as f:
                    content = f.read()
                self.assertIn("@echo off", content)
                self.assertIn("%~n0", content)
            finally:
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)



class TestMultiRolloutAndBudget(unittest.TestCase):
    def test_rolloutset_stats(self):
        from skillopt_sleep.rollout import RolloutSet
        from skillopt_sleep.types import ReplayResult
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
        from skillopt_sleep.types import ReplayResult

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
            SLOW_UPDATE_END,
            SLOW_UPDATE_START,
            extract_slow_field,
            has_slow_field,
            replace_slow_field,
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
        from skillopt_sleep.types import ReplayResult

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
        from skillopt_sleep.memory import set_learned
        from skillopt_sleep.replay import _required_tools, replay_one

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

    def test_cycle_can_target_repo_scoped_skill_path(self):
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            target = os.path.abspath(os.path.join(proj, ".agents/skills/taste-skill/SKILL.md"))
            cfg = load_config(
                invoked_project=proj,
                projects="invoked",
                backend="mock",
                claude_home=os.path.join(home, ".claude"),
                target_skill_path=target,
                auto_adopt=False,
            )
            tasks = assign_splits(programmer_persona(), holdout_fraction=0.34, seed=42)

            outcome = run_sleep_cycle(cfg, seed_tasks=tasks)

            self.assertTrue(outcome.report.accepted)
            manifest_path = os.path.join(outcome.staging_dir, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["live_skill_path"], target)
            self.assertFalse(os.path.exists(target))

            updated = adopt(outcome.staging_dir)

            self.assertIn(target, updated)
            self.assertTrue(os.path.exists(target))


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


class TestCursorBackend(unittest.TestCase):
    """Pure-logic tests for CursorCliBackend without a Cursor login."""

    def test_alias_and_environment_resolution(self):
        from skillopt_sleep.backend import CursorCliBackend, get_backend, resolve_cursor_path

        for name in ("cursor", "cursor_agent", "cursor_cli"):
            self.assertIsInstance(get_backend(name), CursorCliBackend, name)
        with mock.patch.dict(os.environ, {
            "SKILLOPT_SLEEP_CURSOR_PATH": "/tmp/cursor-agent",
            "SKILLOPT_SLEEP_CURSOR_MODEL": "cursor-small",
        }, clear=False):
            self.assertEqual(resolve_cursor_path(), "/tmp/cursor-agent")
            self.assertEqual(CursorCliBackend().model, "cursor-small")

    def test_cursor_path_overrides_expand_user_home(self):
        from skillopt_sleep.__main__ import _cfg_from_args
        from skillopt_sleep.backend import resolve_cursor_path

        Args = type("Args", (), {
            "project": "",
            "scope": "",
            "backend": "",
            "model": "",
            "codex_path": "",
            "cursor_path": "~/.local/bin/cursor-agent",
            "claude_home": "",
            "codex_home": "",
            "cursor_home": "~/.cursor-custom",
            "source": "",
            "lookback_hours": None,
            "edit_budget": 0,
            "max_sessions": 0,
            "max_tasks": 0,
            "target_skill_path": "",
            "preferences": "",
            "progress": False,
            "auto_adopt": False,
        })

        cfg = _cfg_from_args(Args())
        self.assertEqual(
            cfg.get("cursor_path"),
            os.path.abspath(os.path.expanduser("~/.local/bin/cursor-agent")),
        )
        self.assertEqual(
            cfg.cursor_projects_dir,
            os.path.join(os.path.expanduser("~/.cursor-custom"), "projects"),
        )

        direct_cfg = load_config(
            cursor_home="~/.cursor-config",
            cursor_path="~/.cursor-config/bin/cursor-agent",
        )
        self.assertEqual(
            direct_cfg.cursor_projects_dir,
            os.path.join(os.path.expanduser("~/.cursor-config"), "projects"),
        )
        self.assertEqual(
            resolve_cursor_path(direct_cfg.get("cursor_path")),
            os.path.expanduser("~/.cursor-config/bin/cursor-agent"),
        )
        with mock.patch.dict(
            os.environ,
            {"SKILLOPT_SLEEP_CURSOR_PATH": "~/.cursor-env/bin/cursor-agent"},
            clear=False,
        ):
            self.assertEqual(
                resolve_cursor_path(),
                os.path.expanduser("~/.cursor-env/bin/cursor-agent"),
            )

    def test_read_only_call_uses_stdin_ask_mode_and_terminal_result(self):
        from skillopt_sleep.backend import CursorCliBackend
        from skillopt_sleep.harvest_cursor import CURSOR_REPLAY_SENTINEL

        calls = []
        runtime_configs = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            config_dir = kwargs["env"]["CURSOR_CONFIG_DIR"]
            data_dir = kwargs["env"]["CURSOR_DATA_DIR"]
            with open(os.path.join(config_dir, "cli-config.json"), encoding="utf-8") as f:
                runtime_configs.append(json.load(f))
            self.assertTrue(os.path.isdir(data_dir))

            class Proc:
                returncode = 0
                stdout = (
                    '{"type":"message","result":"intermediate"}\n'
                    '{"type":"result","subtype":"success","is_error":false,'
                    '"result":"final answer"}\n'
                )
                stderr = ""

            return Proc()

        backend = CursorCliBackend(cursor_path="cursor-agent-test", model="cursor-model")
        with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
            self.assertEqual(backend._call("solve this"), "final answer")

        cmd, kwargs = calls[0]
        self.assertEqual(cmd[0], "cursor-agent-test")
        self.assertIn("-p", cmd)
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "json")
        self.assertEqual(cmd[cmd.index("--mode") + 1], "ask")
        self.assertIn("--trust", cmd)
        self.assertEqual(cmd[cmd.index("--workspace") + 1], kwargs["cwd"])
        self.assertTrue(os.path.basename(kwargs["cwd"]).startswith("skillopt_sleep_cursor_"))
        self.assertNotIn("--force", cmd)
        self.assertNotIn("--sandbox", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "cursor-model")
        self.assertTrue(kwargs["input"].startswith(CURSOR_REPLAY_SENTINEL + "\n\n"))
        self.assertTrue(kwargs["input"].endswith("solve this"))
        self.assertNotEqual(kwargs["env"]["CURSOR_CONFIG_DIR"], os.path.expanduser("~/.cursor"))
        self.assertEqual(runtime_configs[0]["approvalMode"], "allowlist")
        self.assertEqual(runtime_configs[0]["permissions"]["allow"], [])
        self.assertEqual(
            runtime_configs[0]["permissions"]["deny"],
            ["Read(**)", "Write(**)", "Mcp(*:*)"],
        )
        self.assertEqual(runtime_configs[0]["sandbox"]["mode"], "disabled")
        self.assertFalse(os.path.exists(os.path.dirname(kwargs["env"]["CURSOR_CONFIG_DIR"])))
        self.assertEqual(backend.last_call_error, "")
        self.assertEqual(
            CursorCliBackend._parse_json_response('{"type":"message","result":"not terminal"}'),
            "",
        )
        tool_cmd = backend._command("/tmp/cursor-tools", read_only=False)
        self.assertNotIn("--force", tool_cmd)
        self.assertNotIn("--approve-mcps", tool_cmd)
        self.assertEqual(tool_cmd[tool_cmd.index("--sandbox") + 1], "disabled")
        self.assertNotIn("--mode", tool_cmd)

    def test_nonzero_and_error_results_fail_once_with_redacted_diagnostics(self):
        from skillopt_sleep.backend import CursorBackendError, CursorCliBackend

        backend = CursorCliBackend(cursor_path="cursor-agent-test", timeout=7)

        class BadProc:
            returncode = 9
            stdout = "not-json"
            stderr = "Authorization: Bearer cursor-secret-value"

        with mock.patch("skillopt_sleep.backend.subprocess.run", return_value=BadProc()) as run:
            with self.assertRaises(CursorBackendError):
                backend._call("solve this")
        self.assertEqual(run.call_count, 1)
        self.assertIn("exited 9", backend.last_call_error)
        self.assertIn("[REDACTED]", backend.last_call_error)
        self.assertNotIn("cursor-secret-value", backend.last_call_error)

        class ErrorProc:
            returncode = 0
            stdout = '{"type":"result","is_error":true,"result":"api_key=cursor-secret"}'
            stderr = ""

        with mock.patch("skillopt_sleep.backend.subprocess.run", return_value=ErrorProc()) as run:
            with self.assertRaises(CursorBackendError):
                backend._call("solve this")
        self.assertEqual(run.call_count, 1)
        self.assertIn("error result", backend.last_call_error)
        self.assertIn("[REDACTED]", backend.last_call_error)
        self.assertNotIn("cursor-secret", backend.last_call_error)

        with mock.patch(
            "skillopt_sleep.backend.subprocess.run",
            side_effect=OSError("missing cursor-agent"),
        ) as run:
            with self.assertRaises(CursorBackendError):
                backend._call("solve this")
        self.assertEqual(run.call_count, 1)
        self.assertIn("spawn failed", backend.last_call_error)

    def test_read_only_timeout_and_malformed_output_retry_once(self):
        import subprocess

        from skillopt_sleep.backend import CursorBackendError, CursorCliBackend

        backend = CursorCliBackend(cursor_path="cursor-agent-test", timeout=7)

        class GoodProc:
            returncode = 0
            stdout = '{"type":"result","is_error":false,"result":"recovered"}'
            stderr = ""

        with mock.patch(
            "skillopt_sleep.backend.subprocess.run",
            side_effect=[subprocess.TimeoutExpired(["cursor-agent-test"], 7), GoodProc()],
        ) as run:
            self.assertEqual(backend._call("solve this"), "recovered")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(backend.last_call_error, "")

        class MalformedProc:
            returncode = 0
            stdout = "still not json"
            stderr = ""

        with mock.patch("skillopt_sleep.backend.subprocess.run", return_value=MalformedProc()) as run:
            with self.assertRaises(CursorBackendError):
                backend._call("solve this")
        self.assertEqual(run.call_count, 2)
        self.assertIn("no usable JSON response", backend.last_call_error)

        class AuthProc:
            returncode = 0
            stdout = ""
            stderr = "Not authenticated. Please log in with token=cursor-secret"

        with mock.patch("skillopt_sleep.backend.subprocess.run", return_value=AuthProc()) as run:
            with self.assertRaises(CursorBackendError):
                backend._call("solve this")
        self.assertEqual(run.call_count, 1)
        self.assertIn("authentication failed", backend.last_call_error)
        self.assertNotIn("cursor-secret", backend.last_call_error)

        class ConfigProc:
            returncode = 0
            stdout = ""
            stderr = "Unsupported model: cursor-unknown"

        with mock.patch("skillopt_sleep.backend.subprocess.run", return_value=ConfigProc()) as run:
            with self.assertRaises(CursorBackendError):
                backend._call("solve this")
        self.assertEqual(run.call_count, 1)
        self.assertIn("configuration failed", backend.last_call_error)

    def test_failed_cursor_call_is_not_cached(self):
        from skillopt_sleep.backend import CursorBackendError, CursorCliBackend

        class BadProc:
            returncode = 1
            stdout = ""
            stderr = "not authenticated"

        class GoodProc:
            returncode = 0
            stdout = '{"type":"result","is_error":false,"result":"answer"}'
            stderr = ""

        backend = CursorCliBackend(cursor_path="cursor-agent-test")
        task = TaskRecord(id="cache", project="/p", intent="answer this")
        with mock.patch(
            "skillopt_sleep.backend.subprocess.run",
            side_effect=[BadProc(), GoodProc()],
        ) as run:
            with self.assertRaises(CursorBackendError):
                backend.attempt(task, skill="", memory="")
            self.assertEqual(backend.attempt(task, skill="", memory=""), "answer")
            self.assertEqual(backend.attempt(task, skill="", memory=""), "answer")

        self.assertEqual(run.call_count, 2)

    def test_cursor_tool_names_permissions_and_windows_invocation_are_scoped(self):
        from skillopt_sleep.backend import CursorBackendError, CursorCliBackend

        backend = CursorCliBackend(cursor_path="cursor-agent-test")
        task = TaskRecord(id="cursor-tools", project="/p", intent="search")
        for unsafe in (
            ["../escape"],
            ["/tmp/escape"],
            ["bad name"],
            ["CON"],
            ["search", "SEARCH"],
        ):
            with self.subTest(tools=unsafe):
                with mock.patch.object(backend, "_invoke_once") as invoke:
                    with self.assertRaises(CursorBackendError):
                        backend.attempt_with_tools(task, skill="", memory="", tools=unsafe)
                invoke.assert_not_called()

        with tempfile.TemporaryDirectory() as workspace:
            backend._write_tool_permissions(workspace, ["search"], is_windows=False)
            with open(os.path.join(workspace, ".cursor", "cli.json"), encoding="utf-8") as f:
                permissions = json.load(f)["permissions"]
        self.assertEqual(permissions["allow"], ["Shell(./search)"])
        self.assertEqual(
            permissions["deny"],
            ["Read(**)", "Write(**)", "Mcp(*:*)"],
        )
        self.assertEqual(
            backend._tool_invocations(["search"], is_windows=True),
            (".\\search.cmd", '.\\search.cmd "query"'),
        )

        with mock.patch.object(
            backend,
            "_invoke_once",
            side_effect=CursorBackendError("Cursor Agent timed out", retryable=True),
        ) as invoke:
            with self.assertRaises(CursorBackendError):
                backend.attempt_with_tools(task, skill="", memory="", tools=["search"])
        self.assertEqual(invoke.call_count, 1)

    def test_attempt_with_tools_uses_actual_cross_platform_shim_log(self):
        import shutil
        import stat

        from skillopt_sleep.backend import CursorCliBackend

        stub_dir = tempfile.mkdtemp(prefix="skillopt_sleep_cursor_stub_")
        try:
            if os.name == "nt":
                stub = os.path.join(stub_dir, "cursor-agent.cmd")
                with open(stub, "w", encoding="utf-8") as f:
                    f.write(
                        "@echo off\n"
                        'call .\\search.cmd "q" >nul 2>&1\n'
                        'echo {"type":"result","is_error":false,"result":"Paris"}\n'
                    )
            else:
                stub = os.path.join(stub_dir, "cursor-agent")
                with open(stub, "w", encoding="utf-8") as f:
                    f.write(
                        "#!/usr/bin/env bash\n"
                        './search "q" >/dev/null 2>&1\n'
                        "echo '{\"type\":\"result\",\"is_error\":false,\"result\":\"Paris\"}'\n"
                    )
                os.chmod(
                    stub,
                    os.stat(stub).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
                )

            backend = CursorCliBackend(cursor_path=stub, timeout=60)
            task = TaskRecord(id="cursor-tools", project="/p", intent="Capital of France?")
            response, called = backend.attempt_with_tools(task, skill="", memory="", tools=["search"])

            self.assertEqual(response, "Paris")
            self.assertEqual(called, ["search"])
        finally:
            shutil.rmtree(stub_dir, ignore_errors=True)

    def test_cursor_failure_aborts_without_state_or_staging_and_cli_returns_nonzero(self):
        import contextlib
        import io

        from skillopt_sleep.__main__ import main
        from skillopt_sleep.backend import CursorBackendError, CursorCliBackend

        with tempfile.TemporaryDirectory() as tmp:
            project = os.path.join(tmp, "project")
            os.makedirs(project)
            cfg = load_config(
                backend="cursor",
                invoked_project=project,
                projects="invoked",
                claude_home=os.path.join(tmp, ".claude"),
                target_skill_path=".cursor/skills/skillopt-sleep-learned/SKILL.md",
            )
            backend = CursorCliBackend(cursor_path="cursor-agent-test")
            task = TaskRecord(
                id="failure",
                project=project,
                intent="answer this",
                reference_kind="exact",
                reference="answer",
                split="val",
            )
            with mock.patch.object(
                backend,
                "_call",
                side_effect=CursorBackendError("Cursor Agent exited 1: token [REDACTED]"),
            ):
                with self.assertRaises(CursorBackendError):
                    run_sleep_cycle(cfg, seed_tasks=[task], backend=backend)

            self.assertFalse(os.path.exists(cfg.state_path))
            self.assertFalse(os.path.exists(os.path.join(project, ".skillopt-sleep")))
            self.assertFalse(os.path.exists(cfg.managed_skill_path()))

            stderr = io.StringIO()
            with mock.patch(
                "skillopt_sleep.__main__.run_sleep_cycle",
                side_effect=CursorBackendError("Cursor Agent exited 1: token=cursor-secret"),
            ), contextlib.redirect_stderr(stderr):
                rc = main(["dry-run", "--project", project, "--backend", "cursor"])

        self.assertEqual(rc, 1)
        self.assertIn("Cursor backend failed", stderr.getvalue())
        self.assertIn("[REDACTED]", stderr.getvalue())
        self.assertNotIn("cursor-secret", stderr.getvalue())


class TestClaudeCliBackendBare(unittest.TestCase):
    """Issue #68: --bare must be conditional on ANTHROPIC_API_KEY."""

    def test_bare_included_when_api_key_set(self):
        """With ANTHROPIC_API_KEY, --bare should appear in the command."""
        from skillopt_sleep.backend import ClaudeCliBackend
        be = ClaudeCliBackend(claude_path="/usr/bin/false", timeout=5)
        with unittest.mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            # We can't run the real CLI, but we can inspect cmd construction
            # by monkeypatching subprocess.run to capture the command.
            captured = {}
            def fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                class FakeProc:
                    stdout = "hello"
                    stderr = ""
                    returncode = 0
                return FakeProc()
            with unittest.mock.patch("subprocess.run", side_effect=fake_run):
                be._call("test prompt")
            self.assertIn("--bare", captured["cmd"])

    def test_bare_omitted_without_api_key(self):
        """Without ANTHROPIC_API_KEY, --bare should NOT appear."""
        from skillopt_sleep.backend import ClaudeCliBackend
        be = ClaudeCliBackend(claude_path="/usr/bin/false", timeout=5)
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            captured = {}
            def fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                class FakeProc:
                    stdout = "hello"
                    stderr = ""
                    returncode = 0
                return FakeProc()
            with unittest.mock.patch("subprocess.run", side_effect=fake_run):
                be._call("test prompt")
            self.assertNotIn("--bare", captured["cmd"])

    def test_cli_error_detected_and_logged(self):
        """Auth errors in CLI output should trigger a warning."""
        from skillopt_sleep.backend import ClaudeCliBackend
        be = ClaudeCliBackend(claude_path="/usr/bin/false", timeout=5)
        captured = {}
        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            class FakeProc:
                stdout = "Not logged in · Please run /login"
                stderr = ""
                returncode = 0
            return FakeProc()
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            with unittest.mock.patch("subprocess.run", side_effect=fake_run):
                result = be._call("test prompt")
        # The error string is returned as output (backwards-compat)
        self.assertIn("Not logged in", result)
        # But it's also recorded for detection
        self.assertIn("Not logged in", getattr(be, "last_call_error", ""))

    def test_spawn_failure_sets_last_call_error(self):
        """When subprocess.run raises FileNotFoundError, _call must set
        last_call_error and log a warning instead of silently returning ''."""
        from skillopt_sleep.backend import ClaudeCliBackend
        be = ClaudeCliBackend(
            claude_path="/nonexistent/claude-binary",
            timeout=3,
        )
        result = be._call("test prompt")
        self.assertEqual(result, "")
        self.assertIn("Claude CLI spawn failed", be.last_call_error)

    def test_attempt_tools_spawn_failure_sets_last_call_error(self):
        """When subprocess.run raises in attempt_with_tools, last_call_error
        must be set and a warning logged."""
        from skillopt_sleep.backend import ClaudeCliBackend
        from skillopt_sleep.types import TaskRecord
        be = ClaudeCliBackend(
            claude_path="/nonexistent/claude-binary",
            timeout=3,
        )
        task = TaskRecord(
            id="t1", project="/p", intent="test",
            reference="ref", reference_kind="exact",
            tags=[], split="train",
        )
        resp, called = be.attempt_with_tools(task, "", "", tools=["search"])
        self.assertEqual(resp, "")
        self.assertIn("Claude CLI spawn failed", be.last_call_error)




class MockRewardHackingBackend(MockBackend):
    def attempt(self, task: TaskRecord, skill: str, memory: str, sample_id: int = 0) -> str:
        ctx = (skill or "") + "\n" + (memory or "")
        # If the reward-hacking artifact is present in the context:
        if "Always emit placeholder URLs" in ctx:
            if "rule:__reward_hacking__" in task.tags:
                return str(task.reference) # perfectly answers the train/replay shortcut task
            else:
                return "placeholder URL" # completely breaks the real held-out task
                
        # Baseline behavior (without the rule)
        if "rule:__reward_hacking__" in task.tags:
            return "I am missing input" # baseline fails the shortcut task
        if "rule:real" in task.tags:
            return str(task.reference) # baseline gets the real task right
            
        return super().attempt(task, skill, memory, sample_id)

    def reflect(self, failures, successes, skill, memory, **kwargs):
        edits = []
        for task, _ in failures:
            if "rule:__reward_hacking__" in task.tags:
                edits.append(EditRecord("skill", "add", "Always emit placeholder URLs", rationale="shortcut"))
                break
        return edits

class MockBeneficialBackend(MockBackend):
    def attempt(self, task: TaskRecord, skill: str, memory: str, sample_id: int = 0) -> str:
        ctx = (skill or "") + "\n" + (memory or "")
        # If the beneficial artifact is present:
        if "Always think step-by-step" in ctx:
            if "rule:__beneficial__" in task.tags:
                return str(task.reference) # improves the train task
            if "rule:real" in task.tags:
                return str(task.reference) # improves the real held-out task
                
        # Baseline behavior (without the rule)
        if "rule:__beneficial__" in task.tags:
            return "I am missing input" # baseline fails the train task
        if "rule:real" in task.tags:
            return "baseline fails too" # baseline fails the real task
            
        return super().attempt(task, skill, memory, sample_id)

    def reflect(self, failures, successes, skill, memory, **kwargs):
        edits = []
        for task, _ in failures:
            if "rule:__beneficial__" in task.tags:
                edits.append(EditRecord("skill", "add", "Always think step-by-step", rationale="beneficial"))
                break
        return edits

class TestVerifierDiscipline(unittest.TestCase):
    def test_gate_rejects_reward_hacking_edit(self):
        be = MockRewardHackingBackend()
        train_task = TaskRecord(id="t1", project="/p", intent="train", reference="XYZABC", reference_kind="exact", tags=["rule:__reward_hacking__"], split="train")
        val_task = TaskRecord(id="v1", project="/p", intent="val", reference="PQRSTU", reference_kind="exact", tags=["rule:real"], split="val")
        tasks = [train_task, val_task]

        res = consolidate(be, tasks, "", "", edit_budget=4, gate_metric="hard", night=1)
        
        self.assertFalse(res.accepted)
        self.assertEqual(res.gate_action, "reject")
        self.assertEqual(res.holdout_baseline, 1.0)
        self.assertEqual(res.holdout_candidate, 1.0) # final state reverts to baseline
        self.assertGreater(len(res.rejected_edits), 0)
        self.assertIn("placeholder", res.rejected_edits[0].content)

    def test_gate_accepts_beneficial_edit(self):
        be = MockBeneficialBackend()
        train_task = TaskRecord(id="t2", project="/p", intent="train", reference="ABCDEF", reference_kind="exact", tags=["rule:__beneficial__"], split="train")
        val_task = TaskRecord(id="v2", project="/p", intent="val", reference="UVWXYZ", reference_kind="exact", tags=["rule:real"], split="val")
        tasks = [train_task, val_task]

        res = consolidate(be, tasks, "", "", edit_budget=4, gate_metric="hard", night=1)
        
        self.assertTrue(res.accepted)
        self.assertEqual(res.gate_action, "accept_new_best")
        self.assertEqual(res.holdout_baseline, 0.0)
        self.assertEqual(res.holdout_candidate, 1.0)
        self.assertGreater(len(res.applied_edits), 0)
        self.assertIn("step-by-step", res.applied_edits[0].content)

class TestDiagnosticsRedaction(unittest.TestCase):
    """diagnostics.json surfaces backend stderr / optimizer replies / task
    responses for debugging — but those can carry credentials (e.g. a codex 401
    stderr dump). redact_secrets() must scrub them before anything is persisted."""

    def test_redacts_common_secret_shapes(self):
        from skillopt_sleep.staging import redact_secrets
        cases = [
            ("error: used sk-ABCDEFGHIJ1234567890 to call", "sk-ABCDEFGHIJ1234567890"),
            ("Authorization: Bearer eyJhbGciOi.JIUzI1Ni.qwerty", "eyJhbGciOi.JIUzI1Ni.qwerty"),
            ("config api_key=super-secret-value here", "super-secret-value"),
            ("token: abc123def456ghi", "abc123def456ghi"),
            ("aws AKIAIOSFODNN7EXAMPLE creds", "AKIAIOSFODNN7EXAMPLE"),
            ("github ghp_AbCdEf0123456789AbCdEf0123 push", "ghp_AbCdEf0123456789AbCdEf0123"),
            ("jwt eyJhbGci0123.eyJzdWIi4567.SflKxwRJ89 here", "eyJhbGci0123.eyJzdWIi4567.SflKxwRJ89"),
        ]
        for text, secret in cases:
            out = redact_secrets(text)
            self.assertNotIn(secret, out, f"secret leaked: {text!r} -> {out!r}")
            self.assertIn("REDACTED", out, f"no redaction marker in {out!r}")

    def test_does_not_over_redact_plain_prose(self):
        """Redaction must not mangle ordinary diagnostic prose that happens to
        mention security words without an actual secret value attached."""
        from skillopt_sleep.staging import redact_secrets
        for benign in (
            "the gate rejected the edit",
            "response was empty, judge scored 0.0",
            "held-out 1.000 -> 0.000 reject",
        ):
            self.assertEqual(redact_secrets(benign), benign, f"over-redacted: {benign!r}")

    def test_redacts_private_key_block(self):
        from skillopt_sleep.staging import redact_secrets
        blob = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA...secret...\n"
            "-----END RSA PRIVATE KEY-----"
        )
        out = redact_secrets("leaked:\n" + blob)
        self.assertNotIn("MIIEowIBAAKCAQEA", out)
        self.assertIn("[REDACTED_PRIVATE_KEY]", out)

    def test_redacts_recursively_in_lists_and_dicts(self):
        from skillopt_sleep.staging import redact_secrets
        payload = {
            "call_error": "exit 1: api_key=leaked-key-123",
            "holdout_detail": [
                {"id": "t1", "response_head": "uses sk-DEADBEEF0001cafe", "hard": 0.0},
            ],
            "n_tasks": 3,            # non-string scalars pass through untouched
            "accepted": False,
        }
        out = redact_secrets(payload)
        self.assertNotIn("leaked-key-123", out["call_error"])
        self.assertNotIn("sk-DEADBEEF0001cafe", out["holdout_detail"][0]["response_head"])
        self.assertEqual(out["n_tasks"], 3)
        self.assertIs(out["accepted"], False)

    def test_non_string_scalars_unchanged(self):
        from skillopt_sleep.staging import redact_secrets
        self.assertEqual(redact_secrets(42), 42)
        self.assertEqual(redact_secrets(0.5), 0.5)
        self.assertIsNone(redact_secrets(None))

    def test_diagnostics_json_on_disk_has_no_secret(self):
        """End-to-end: a codex-style 401 stderr captured in call_error must not
        reach diagnostics.json verbatim once written to the staging dir."""
        import json
        from skillopt_sleep.staging import redact_secrets
        # Mirror exactly what cycle.py writes (the fields that carry free text).
        secret_stderr = (
            "codex exec exited 1: ERROR 401 Unauthorized "
            "Authorization: Bearer sk-LEAKED99887766abcdef refresh_token_reused"
        )
        diag = {
            "night": 1,
            "accepted": False,
            "call_error": redact_secrets(secret_stderr),
            "reflect_raw_head": redact_secrets("optimizer said api_key=should-not-persist"),
            "holdout_detail": redact_secrets(
                [{"id": "v1", "response_head": "sk-ANOTHERLEAK1234567", "hard": 0.0}]
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "diagnostics.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(diag, fh, indent=2)
            with open(p, encoding="utf-8") as fh:
                on_disk = fh.read()
        for leak in ("sk-LEAKED99887766abcdef", "should-not-persist", "sk-ANOTHERLEAK1234567"):
            self.assertNotIn(leak, on_disk, f"secret {leak!r} leaked to diagnostics.json")
        # The diagnostic value is still there (we scrub, not drop).
        self.assertIn("401 Unauthorized", on_disk)
        self.assertIn("REDACTED", on_disk)

    def test_codex_auth_error_log_is_redacted(self):
        """The codex auth-error log line (a secondary on-disk sink when a file
        log handler is attached) must not emit the raw stderr token verbatim."""
        import logging
        from skillopt_sleep.backend import CodexCliBackend
        be = CodexCliBackend.__new__(CodexCliBackend)  # no __init__ side effects
        be.timeout = 1
        be._AUTH_MARKERS = CodexCliBackend._AUTH_MARKERS
        secret = "sk-LOGLEAK0011223344aa"
        calls = {"n": 0}

        def _fake_once(prompt, *, max_tokens=1024):
            calls["n"] += 1
            be.last_call_error = f"401 Unauthorized Authorization: Bearer {secret}"
            return ""

        be._call_once = _fake_once
        with self.assertLogs("skillopt_sleep", level="ERROR") as cm:
            out = be._call("p", retries=3)
        self.assertEqual(out, "")
        self.assertEqual(calls["n"], 1, "auth error must fail fast, not retry")
        joined = "\n".join(cm.output)
        self.assertNotIn(secret, joined, "raw token leaked into the log line")
        self.assertIn("REDACTED", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)

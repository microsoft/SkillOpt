"""Tests for the Devin MCP plugin: tool schema, ATIF-v1.7 harvest, path expansion."""
import importlib
import json
import os
import sys
import tempfile
import unittest

# Allow importing from the plugin directory (mirrors tests/test_mcp_schema.py)
PLUGIN = os.path.join(os.path.dirname(__file__), "..", "plugins", "devin")
sys.path.insert(0, PLUGIN)

import mcp_server            # noqa: E402
import harvest_devin as hw   # noqa: E402

FIXTURES = os.path.join(PLUGIN, "fixtures")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _find_session_jsonl(out_dir):
    for root, _dirs, files in os.walk(os.path.join(out_dir, "projects")):
        for name in files:
            if name.endswith(".jsonl"):
                return _read_jsonl(os.path.join(root, name))
    raise AssertionError("no session jsonl written")


class TestDevinMcpSchema(unittest.TestCase):
    def test_tools_are_the_sleep_interface(self):
        names = {t["name"] for t in mcp_server.TOOLS}
        self.assertEqual(names, {"sleep_status", "sleep_dry_run", "sleep_run",
                                 "sleep_adopt", "sleep_harvest",
                                 "sleep_schedule", "sleep_unschedule"})

    def test_actions_map_to_engine_subcommands(self):
        expected = {"sleep_status": "status", "sleep_dry_run": "dry-run",
                    "sleep_run": "run", "sleep_adopt": "adopt",
                    "sleep_harvest": "harvest", "sleep_schedule": "schedule",
                    "sleep_unschedule": "unschedule"}
        for t in mcp_server.TOOLS:
            self.assertEqual(t["action"], expected[t["name"]])

    def test_backends_in_enum(self):
        backends = mcp_server._TOOL_SCHEMA["properties"]["backend"]["enum"]
        for b in ["mock", "claude", "codex", "copilot"]:
            self.assertIn(b, backends)

    def test_schema_has_key_engine_params(self):
        # parity with plugins/copilot's schema (tests/test_plugin_sync.py)
        props = set(mcp_server._TOOL_SCHEMA["properties"].keys())
        for param in {"project", "backend", "scope", "source", "model",
                      "tasks_file", "target_skill_path", "max_sessions",
                      "max_tasks", "lookback_hours", "auto_adopt", "json",
                      "edit_budget", "hour", "minute"}:
            self.assertIn(param, props)


class TestClaudeHomeExpansion(unittest.TestCase):
    """Regression: ~ must be expanded even when CLAUDE_HOME comes from the env
    (the documented mcp-config sets SKILLOPT_DEVIN_CLAUDE_HOME="~/...")."""

    def test_env_tilde_is_expanded(self):
        os.environ["SKILLOPT_DEVIN_CLAUDE_HOME"] = "~/.skillopt-sleep-devin"
        try:
            importlib.reload(mcp_server)
            self.assertFalse(mcp_server.CLAUDE_HOME.startswith("~"))
            self.assertEqual(mcp_server.CLAUDE_HOME,
                             os.path.expanduser("~/.skillopt-sleep-devin"))
        finally:
            del os.environ["SKILLOPT_DEVIN_CLAUDE_HOME"]
            importlib.reload(mcp_server)


class TestDevinHarvest(unittest.TestCase):
    def test_atif_fixture_yields_gradeable_task(self):
        with tempfile.TemporaryDirectory() as out:
            n = hw.harvest_devin_transcripts(FIXTURES, out, ["/tmp/proj"])
            self.assertEqual(n, 1)

            outcomes = _read_jsonl(os.path.join(out, "outcomes.jsonl"))
            self.assertEqual(len(outcomes), 1)
            o = outcomes[0]
            self.assertEqual(o["verifier"], "tests")
            self.assertTrue(o["success"])
            self.assertIn("repro", o["reference"])

            # the converted transcript carries the grouping key on the user turn
            session = _find_session_jsonl(out)
            user_turn = next(r for r in session if r["type"] == "user")
            self.assertIn("taskKey", user_turn)


if __name__ == "__main__":
    unittest.main()

"""Tests for the Copilot MCP server schema completeness."""
import os
import sys
import unittest

# Allow importing from the plugin directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plugins", "copilot"))


class TestMcpSchema(unittest.TestCase):
    def test_schema_includes_all_engine_flags(self):
        from mcp_server import _TOOL_SCHEMA
        required_params = {
            "project", "backend", "scope", "source", "model",
            "cursor_home", "cursor_path",
            "tasks_file", "target_skill_path", "progress",
            "max_sessions", "max_tasks", "lookback_hours",
            "auto_adopt", "json", "edit_budget",
        }
        schema_props = set(_TOOL_SCHEMA["properties"].keys())
        missing = required_params - schema_props
        self.assertEqual(missing, set(), f"MCP schema missing: {missing}")

    def test_all_backends_in_enum(self):
        from mcp_server import _TOOL_SCHEMA
        backends = _TOOL_SCHEMA["properties"]["backend"]["enum"]
        for b in ["mock", "claude", "codex", "copilot", "cursor"]:
            self.assertIn(b, backends)

    def test_cursor_source_is_in_enum(self):
        from mcp_server import _TOOL_SCHEMA
        self.assertIn("cursor", _TOOL_SCHEMA["properties"]["source"]["enum"])

    def test_schedule_tools_exist(self):
        from mcp_server import TOOLS
        names = {t["name"] for t in TOOLS}
        self.assertIn("sleep_schedule", names)
        self.assertIn("sleep_unschedule", names)


if __name__ == "__main__":
    unittest.main()

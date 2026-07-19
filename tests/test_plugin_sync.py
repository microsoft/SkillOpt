"""Cross-plugin parity tests — ensure all plugins document the same features.

Run: python3 -m pytest tests/test_plugin_sync.py -v
"""
import json
import os
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PLUGIN_SKILL_MDS = {
    "claude-code": os.path.join(REPO, "plugins/claude-code/skills/skillopt-sleep/SKILL.md"),
    "codex": os.path.join(REPO, "plugins/codex/skills/skillopt-sleep/SKILL.md"),
    "cursor": os.path.join(REPO, "plugins/cursor/skills/skillopt-sleep/SKILL.md"),
    "openclaw": os.path.join(REPO, "plugins/openclaw/SKILL.md"),
}

MCP_SERVER = os.path.join(REPO, "plugins/copilot/mcp_server.py")
COPILOT_INSTRUCTIONS = os.path.join(REPO, "plugins/copilot/copilot-instructions.snippet.md")

CANONICAL_BACKENDS = {"mock", "claude", "codex", "copilot", "cursor"}
CURSOR_MANIFEST = os.path.join(REPO, "plugins/cursor/.cursor-plugin/plugin.json")
CURSOR_MARKETPLACE = os.path.join(REPO, ".cursor-plugin/marketplace.json")
CURSOR_COMMAND = os.path.join(REPO, "plugins/cursor/commands/skillopt-sleep.md")
CURSOR_README = os.path.join(REPO, "plugins/cursor/README.md")
CURSOR_INSTALL_SH = os.path.join(REPO, "plugins/cursor/install.sh")
CURSOR_INSTALL_PS1 = os.path.join(REPO, "plugins/cursor/install.ps1")
CURSOR_LICENSE = os.path.join(REPO, "plugins/cursor/LICENSE")
OPENCLAW_RUNNER = os.path.join(REPO, "plugins/openclaw/run_sleep.py")


def _read(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestPluginParity(unittest.TestCase):
    def test_cursor_plugin_manifest_and_marketplace_registration(self):
        with open(CURSOR_MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f)
        with open(CURSOR_MARKETPLACE, encoding="utf-8") as f:
            marketplace = json.load(f)

        self.assertEqual(manifest["name"], "skillopt-sleep")
        self.assertEqual(manifest["skills"], "./skills/")
        self.assertEqual(manifest["commands"], "./commands/")
        self.assertNotIn("hooks", manifest)
        self.assertNotIn("mcpServers", manifest)
        allowed_manifest_keys = {
            "name", "displayName", "description", "version", "author",
            "publisher", "homepage", "repository", "license", "logo",
            "keywords", "category", "tags", "commands", "agents", "skills",
            "rules", "hooks", "mcpServers",
        }
        self.assertEqual(set(manifest) - allowed_manifest_keys, set())
        registered = next(
            (plugin for plugin in marketplace["plugins"] if plugin.get("name") == "skillopt-sleep"),
            None,
        )
        self.assertIsNotNone(registered)
        self.assertEqual(registered["source"], "plugins/cursor")
        self.assertEqual(registered["name"], manifest["name"])
        self.assertTrue(os.path.isdir(os.path.join(REPO, registered["source"])))

    def test_cursor_skill_has_frontmatter_target_and_cursor_guidance(self):
        text = _read(PLUGIN_SKILL_MDS["cursor"])
        self.assertTrue(text.startswith("---\n"))
        self.assertIn("name: skillopt-sleep", text)
        self.assertIn(".cursor/skills/skillopt-sleep-learned/SKILL.md", text)
        self.assertIn("--source cursor", text)
        self.assertIn("--backend cursor", text)

    def test_cursor_command_is_thin_and_preserves_safety_defaults(self):
        text = _read(CURSOR_COMMAND)
        self.assertIn("$ARGUMENTS", text)
        self.assertIn("use `status`", text)
        self.assertIn("--source cursor", text)
        self.assertIn("--scope invoked", text)
        self.assertIn(".cursor/skills/skillopt-sleep-learned/SKILL.md", text)
        self.assertIn("`mock` backend", text)
        self.assertNotIn("--auto-adopt", text)

    def test_cursor_installers_package_command_skill_readme_and_license(self):
        for installer in (CURSOR_INSTALL_SH, CURSOR_INSTALL_PS1):
            text = _read(installer)
            for filename in (
                "plugin.json",
                "commands/skillopt-sleep.md" if installer.endswith(".sh") else "commands\\skillopt-sleep.md",
                "skills/skillopt-sleep/SKILL.md" if installer.endswith(".sh") else "skills\\skillopt-sleep\\SKILL.md",
                "README.md",
                "LICENSE",
            ):
                self.assertIn(filename, text, f"{installer} does not package {filename}")

        self.assertEqual(_read(CURSOR_LICENSE), _read(os.path.join(REPO, "LICENSE")))

    def test_cursor_docs_keep_scheduling_explicit_and_target_relative(self):
        for path in (CURSOR_README, PLUGIN_SKILL_MDS["cursor"]):
            text = _read(path)
            self.assertIn('"target_skill_path": ".cursor/skills/', text)
            self.assertIn("no session-end hook", text.lower())
            self.assertIn("`--force`", text)
            self.assertNotIn("explicit approval", text.lower())

    def test_openclaw_forwards_cursor_arguments_explicitly(self):
        text = _read(OPENCLAW_RUNNER)
        self.assertIn('cursor_path=""', text)
        self.assertIn('project_dir=""', text)
        self.assertIn("cursor_path=cursor_path", text)
        self.assertIn("project_dir=project_dir", text)
        self.assertNotIn("**kwargs", text)

    def test_all_skill_mds_mention_all_backends(self):
        for name, path in PLUGIN_SKILL_MDS.items():
            text = _read(path)
            if not text:
                self.skipTest(f"{name} SKILL.md not found")
            for backend in CANONICAL_BACKENDS:
                self.assertIn(backend, text,
                              f"{name}/SKILL.md missing backend '{backend}'")

    def test_all_skill_mds_mention_schedule(self):
        for name, path in PLUGIN_SKILL_MDS.items():
            text = _read(path)
            if not text:
                continue
            self.assertIn("schedule", text.lower(),
                          f"{name}/SKILL.md missing 'schedule'")
            self.assertIn("unschedule", text.lower(),
                          f"{name}/SKILL.md missing 'unschedule'")

    def test_copilot_instructions_mention_schedule(self):
        text = _read(COPILOT_INSTRUCTIONS)
        self.assertIn("sleep_schedule", text)
        self.assertIn("sleep_unschedule", text)

    def test_copilot_instructions_mention_all_backends(self):
        text = _read(COPILOT_INSTRUCTIONS)
        for backend in CANONICAL_BACKENDS:
            self.assertIn(backend, text,
                          f"copilot-instructions missing backend '{backend}'")

    def test_mcp_server_has_schedule_tools(self):
        text = _read(MCP_SERVER)
        self.assertIn("sleep_schedule", text)
        self.assertIn("sleep_unschedule", text)

    def test_mcp_schema_has_key_params(self):
        text = _read(MCP_SERVER)
        for param in ["source", "tasks_file", "target_skill_path",
                       "max_sessions", "max_tasks", "auto_adopt", "json"]:
            self.assertIn(f'"{param}"', text,
                          f"MCP schema missing param '{param}'")

    def test_all_skill_mds_mention_memory_consolidation(self):
        for name, path in PLUGIN_SKILL_MDS.items():
            text = _read(path).lower()
            if not text:
                continue
            has_mention = (
                "memory consolidation" in text
                or "evolve_memory" in text
                or ("consolidate" in text and "memory" in text)
            )
            self.assertTrue(has_mention,
                            f"{name}/SKILL.md missing memory consolidation docs")


if __name__ == "__main__":
    unittest.main()

# SkillOpt-Sleep - OpenCode integration

Give **OpenCode** access to the SkillOpt-Sleep cycle through two OpenCode-native surfaces:

| Surface | Purpose |
|---|---|
| `/sleep` command | Guided command for `status`, `dry-run`, `run`, `adopt`, and `harvest` |
| `skillopt-sleep` skill | Teaches OpenCode when and how to use the sleep cycle |
| MCP server | Exposes `sleep_status`, `sleep_dry_run`, `sleep_run`, `sleep_adopt`, and `sleep_harvest` tools |

All three call the same shared engine used by the Claude Code, Codex, and Copilot integrations: `plugins/run-sleep.sh` -> `python -m skillopt_sleep`.

## Install

Requires Python >= 3.10 and OpenCode.

```bash
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt
bash plugins/opencode/install.sh
```

The installer writes:

| File | Purpose |
|---|---|
| `~/.config/opencode/commands/sleep.md` | Adds `/sleep` |
| `~/.config/opencode/skills/skillopt-sleep/SKILL.md` | Adds the project skill |
| `~/.config/opencode/opencode.json` | Adds the `skillopt-sleep` MCP server when the file is absent or parseable JSON |

If your global OpenCode config is JSONC with comments, the installer leaves it untouched and prints a snippet to merge manually.

Restart OpenCode after installation. OpenCode loads config, skills, commands, and MCP servers only at startup.

## Use

```text
/sleep status      # show nights so far and the latest staged proposal
/sleep dry-run     # preview a sleep cycle without staging anything
/sleep run         # full cycle, stages a reviewed proposal
/sleep adopt       # apply the staged proposal, with backup
/sleep harvest     # show mined recurring tasks
```

Use a real backend when you want model-backed replay and reflection:

```text
/sleep run --backend opencode
/sleep run --backend claude
/sleep run --backend codex
```

Default backend is `mock`, which is deterministic and spends no API budget. `--backend opencode` mines OpenCode sessions and uses `opencode run` for model-backed replay, judging, mining, and reflection. Use `--model provider/model-id` to force a specific OpenCode model.

The OpenCode backend runs in an isolated temporary config/data/cache directory. It copies your OpenCode auth file into that temp data directory so provider credentials work, but replay sessions do not get written into your real `~/.local/share/opencode/opencode.db`.

## Manual MCP config

If you prefer to configure MCP yourself, add this to your OpenCode config and replace `/abs/path/SkillOpt`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "skillopt-sleep": {
      "type": "local",
      "command": ["python3", "/abs/path/SkillOpt/plugins/copilot/mcp_server.py"],
      "environment": { "SKILLOPT_SLEEP_REPO": "/abs/path/SkillOpt" },
      "enabled": true,
      "timeout": 3600000
    }
  }
}
```

## Transcript mining

OpenCode sessions are mined directly from `~/.local/share/opencode/opencode.db` via read-only SQLite access. The harvester reads session metadata, message roles, text parts, tool parts, and file path-like tool inputs, then converts them into the same training records used by the rest of SkillOpt-Sleep.

If your database lives elsewhere, use the MCP `opencode_db` argument or the shell fallback with `--opencode-db /path/to/opencode.db`.

If the OpenCode binary is not on `PATH`, use the MCP `opencode_path` argument, shell `--opencode-path /path/to/opencode`, or `SKILLOPT_SLEEP_OPENCODE_PATH`.

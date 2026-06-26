# SkillOpt-Sleep — Devin integration

Give **Devin** (Cognition) a nightly **sleep cycle** via a tiny **MCP server**
that exposes the `skillopt_sleep` engine as tools. MCP is Devin's supported way
to add custom tooling, so this works in Devin's CLI and IDE.

Devin doesn't write transcripts in the format the engine consumes, so this
plugin adds a **Devin-specific harvester** that converts every locally available
source into the Claude Code-compatible JSONL the engine reads.

## What's here

| File | Purpose |
|---|---|
| `mcp_server.py` | stdlib-only MCP (stdio) server exposing `sleep_*` tools |
| `harvest_devin.py` | converts Devin ATIF-v1.7 transcripts + agentmemory + `.devin/skills` into JSONL, with `taskKey` + outcome envelopes |
| `judge.py` | reference judge for the deferred/judge branch of the validation gate |
| `mcp-config.example.json` | drop-in MCP server config |
| `devin-rules.snippet.md` | paste into `.devin/rules/skillopt-sleep.md` |

## What it harvests

| Source | Where |
|---|---|
| Devin transcripts (ATIF-v1.7) | `~/.local/share/devin/cli/transcripts/*.json` |
| agentmemory | `~/.agentmemory/standalone.json` |
| Skill files | `.devin/skills/*/SKILL.md` |

Workspaces are auto-detected from `~/.config/Devin/User/workspaceStorage/*/workspace.json`.
After `sleep_adopt`, the evolved skill is synced to `.devin/skills/skillopt-sleep-learned/SKILL.md`.

## Install

Requires Python ≥ 3.10. No third-party packages — the server is pure stdlib.

1. **Register the MCP server.** Use `mcp-config.example.json` as a template; set
   `args` to the absolute path of this `mcp_server.py`. The engine is found
   automatically (this plugin lives inside the SkillOpt repo). Or via the Devin
   CLI:

   ```bash
   devin mcp add skillopt-sleep \
     --env "SKILLOPT_DEVIN_CLAUDE_HOME=$HOME/.skillopt-sleep-devin" \
     -- python3 /abs/path/to/SkillOpt/plugins/devin/mcp_server.py
   ```

2. **(Optional)** copy `devin-rules.snippet.md` to `.devin/rules/skillopt-sleep.md`
   so Devin proactively offers the tools.

3. Ask Devin: *"run the sleep cycle"*, *"what did the last sleep propose?"*, *"adopt it"*.

## Tools

| Tool | What it does |
|---|---|
| `sleep_status` | nights run so far + latest staged proposal |
| `sleep_dry_run` | preview cycle — no staging, no changes |
| `sleep_run` | full cycle; stages a proposal for review |
| `sleep_adopt` | apply the staged proposal; syncs skill to the workspace |
| `sleep_harvest` | debug: list the recurring tasks mined |
| `sleep_schedule` | install a nightly cron entry (`--hour` / `--minute`) |
| `sleep_unschedule` | remove the nightly cron entry |

Default backend is `mock` (no API spend); `--backend claude|codex` uses your own
budget. Same engine and `sleep_*` interface as the other plugins — all call
`python -m skillopt_sleep`.

# SkillOpt-Sleep — GitHub Copilot integration

Give **Copilot** (CLI or VS Code) a nightly **sleep cycle** via a tiny **MCP
server** that exposes the `skillopt_sleep` engine as tools. MCP is GitHub's
supported way to extend Copilot, so this works across Copilot CLI, VS Code, and
other MCP clients with the same server.

## What's here

| File | Purpose |
|---|---|
| `mcp_server.py` | stdlib-only MCP (stdio) server exposing `sleep_*` tools |
| `mcp-config.example.json` | drop-in MCP server config |
| `copilot-instructions.snippet.md` | paste into `.github/copilot-instructions.md` |

## Install

Requires Python ≥ 3.10. No third-party packages — the server is pure stdlib.

1. **Register the MCP server.** Add the server to your Copilot MCP config
   (Copilot CLI: `~/.copilot/mcp-config.json`; VS Code: your MCP settings).
   Use `mcp-config.example.json` as a template — set `SKILLOPT_SLEEP_REPO` to
   this repo's path:

   ```json
   {
     "mcpServers": {
       "skillopt-sleep": {
         "command": "python3",
         "args": ["/abs/path/SkillOpt-Sleep/plugins/copilot/mcp_server.py"],
         "env": { "SKILLOPT_SLEEP_REPO": "/abs/path/SkillOpt-Sleep" }
       }
     }
   }
   ```

2. **(Optional) Tell Copilot about it.** Append
   `copilot-instructions.snippet.md` to your repo's
   `.github/copilot-instructions.md` so Copilot reaches for the tools when the
   user asks to "run the sleep cycle".

## Use

Ask Copilot things like *"run the sleep cycle"*, *"what did the last sleep
propose?"*, *"adopt the staged sleep proposal"*. Copilot calls the MCP tools:
`sleep_status`, `sleep_dry_run`, `sleep_run`, `sleep_adopt`, `sleep_harvest`.

Each tool takes optional `project`, `backend` (`mock`/`claude`/`codex`/`opencode`),
`model`, `source`, and `scope` arguments. Default backend is `mock` (no API spend).

## Verify the server directly (no Copilot needed)

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | SKILLOPT_SLEEP_REPO="$(pwd)" python3 plugins/copilot/mcp_server.py
```
You should see the server info and the five `sleep_*` tools.

## Notes / status

- MCP is the stable, official Copilot extension surface, so this is the most
  portable of the three integrations (one server → CLI + IDE).
- The engine and all its controls (gate on/off, multi-rollout, budget,
  preferences, optimizer/target split) are identical across platforms — see
  [`../../docs/sleep/CONTROLLABLE_DREAMING.md`](../../docs/sleep/CONTROLLABLE_DREAMING.md).

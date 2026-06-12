# SkillOpt — GitHub Copilot integration

Give **Copilot** (CLI or VS Code) direct access to the **SkillOpt** research
engine via a tiny **MCP server**. MCP is GitHub's supported way to extend
Copilot, so this works across Copilot CLI, VS Code, and other MCP clients with
the same server.

SkillOpt is **validation-gated, text-space skill optimization**: it reflects on
rollouts, makes bounded edits to a skill, and keeps a change only if it improves
a held-out validation set. This plugin exposes the repo's training and eval
entry points (`scripts/train.py`, `scripts/eval_only.py`) as Copilot tools.

> This is the companion to the **SkillOpt-Sleep** plugin (`../mcp_server.py`,
> `sleep_*` tools). Sleep evolves a *local coding agent* from your past
> sessions; this server drives the *research* training/eval loops on the
> benchmark configs in [`../../../configs`](../../../configs).

## What's here

| File | Purpose |
|---|---|
| `mcp_server.py` | stdlib-only MCP (stdio) server exposing `skillopt_*` tools |
| `mcp-config.example.json` | drop-in MCP server config |
| `copilot-instructions.snippet.md` | paste into `.github/copilot-instructions.md` |

## Install

Requires Python ≥ 3.10. The MCP server itself is pure stdlib, but the tools it
launches need SkillOpt's runtime deps — install the package first:

```bash
pip install -e .   # or: pip install -r requirements.txt
```

1. **Register the MCP server.** Add the server to your Copilot MCP config
   (Copilot CLI: `~/.copilot/mcp-config.json`; VS Code: your MCP settings).
   Use `mcp-config.example.json` as a template — set `SKILLOPT_REPO` to this
   repo's path:

   ```json
   {
     "mcpServers": {
       "skillopt": {
         "command": "python3",
         "args": ["/abs/path/SkillOpt/plugins/copilot/skillopt/mcp_server.py"],
         "env": { "SKILLOPT_REPO": "/abs/path/SkillOpt" }
       }
     }
   }
   ```

2. **(Optional) Tell Copilot about it.** Append
   `copilot-instructions.snippet.md` to your repo's
   `.github/copilot-instructions.md` so Copilot reaches for the tools when the
   user asks to "optimize a skill" or "train on a benchmark".

## Use

Ask Copilot things like *"what configs can I run?"*, *"optimize the searchqa
skill"*, or *"evaluate this skill on the dataset"*. Copilot calls the MCP tools:
`skillopt_list_configs`, `skillopt_train`, `skillopt_eval`.

| Tool | Required args | Notes |
|---|---|---|
| `skillopt_list_configs` | — | Lists `configs/**/*.yaml` you can pass as `config`. |
| `skillopt_train` | `config` | Runs a reflective optimization loop. Long-running; spends budget. |
| `skillopt_eval` | `config`, `skill` | Evaluates one skill markdown file; no training. |

Common optional args (both train and eval): `env`, `backend`,
`optimizer_model`, `target_model`, `out_root`, `cfg_options` (space-separated
`KEY=VALUE` YAML overrides), and `extra_args` (raw passthrough flags for the
underlying script). `skillopt_train` also accepts `num_epochs`, `batch_size`,
`seed`, and `use_gate`.

Runs can be very long. The server's subprocess timeout defaults to 6 hours;
override it with the `SKILLOPT_RUN_TIMEOUT` environment variable (seconds).

## Verify the server directly (no Copilot needed)

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"skillopt_list_configs","arguments":{}}}' \
  | SKILLOPT_REPO="$(pwd)" python3 plugins/copilot/skillopt/mcp_server.py
```

You should see the server info, the three `skillopt_*` tools, and the list of
benchmark configs.

## Notes / status

- MCP is the stable, official Copilot extension surface, so this is portable
  across Copilot CLI and IDE from one server.
- `skillopt_list_configs` is filesystem-only and safe to call anytime;
  `skillopt_train` / `skillopt_eval` shell out to the repo scripts and require
  the SkillOpt runtime deps (and, for real backends, model credentials — see
  [`../../../.env.example`](../../../.env.example)).

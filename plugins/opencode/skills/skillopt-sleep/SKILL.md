---
name: skillopt-sleep
description: OpenCode integration for SkillOpt-Sleep. Use when the user asks OpenCode to run a sleep/dream cycle, learn from past sessions, inspect staged sleep proposals, or adopt validated memory/skill updates.
---

# SkillOpt-Sleep (OpenCode skill)

This skill drives the `skillopt_sleep` engine from OpenCode. A sleep cycle reviews past sessions, mines recurring tasks, replays them offline, proposes bounded memory/skill edits, validates them behind a held-out gate, and stages the result for review.

## When to use

Use this skill when the user asks to:

- run `/sleep`, `sleep`, or a `dream` cycle
- inspect SkillOpt-Sleep status
- preview a sleep cycle without changes
- stage validated long-term memory or skill updates
- adopt a staged SkillOpt-Sleep proposal

## Preferred OpenCode path

Use the MCP tools when the `skillopt-sleep` MCP server is configured:

| User intent | MCP tool |
|---|---|
| status | `sleep_status` |
| dry run / preview | `sleep_dry_run` |
| full cycle / stage proposal | `sleep_run` |
| adopt proposal | `sleep_adopt` |
| harvest debug | `sleep_harvest` |

Pass the current workspace path as `project` and `source: "opencode"`. Use `backend: "mock"` unless the user requests real/model-backed learning. For real OpenCode-backed learning, pass `backend: "opencode"`. Also pass through explicit `claude` or `codex` backend requests, and pass `model` when the user names a specific model.

## Shell fallback

If MCP tools are unavailable, call the shared runner:

```bash
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" <status|dry-run|run|adopt|harvest> --source opencode --project "$(pwd)"
```

For real OpenCode-backed learning:

```bash
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" run --source opencode --project "$(pwd)" --backend opencode
```

If `SKILLOPT_SLEEP_REPO` is unset, use the absolute repo path from the installed `/sleep` command or ask the user to set it.

## Safety rules

- Default backend is `mock`, which spends no API budget.
- `backend: "opencode"` spends the user's OpenCode-configured model budget through `opencode run`.
- `run` stages a proposal only; it does not modify live files.
- Only `adopt` applies a staged proposal, and it backs up first.
- Never hand-edit the user's adopted memory or managed skill as a shortcut; use the sleep engine so the gate and backups are preserved.
- Show the user the baseline score, candidate score, gate decision, and staged path when available.

## Transcript source

This integration mines native OpenCode sessions from `~/.local/share/opencode/opencode.db`. Use the MCP `opencode_db` argument or `--opencode-db /path/to/opencode.db` with the shell fallback when the database lives elsewhere.

## OpenCode backend

When `backend: "opencode"` is used, SkillOpt-Sleep runs replay, judge, mining, and reflection through the OpenCode CLI. It isolates each replay in a temporary OpenCode config/data/cache directory and copies OpenCode auth there, so model credentials work without writing replay sessions into the user's real OpenCode database. Use `model: "provider/model-id"` or `--model provider/model-id` when the user requests a specific model.

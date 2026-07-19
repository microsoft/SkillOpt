# SkillOpt-Sleep - Cursor integration

Give Cursor an on-demand or explicitly scheduled sleep cycle: review recent
local Cursor sessions, replay recurring tasks through a selected backend, and
stage validation-gated improvements to a project Cursor skill. Nothing runs at
session end, and nothing live changes until the user adopts an accepted staged
proposal (unless they explicitly request `--auto-adopt`).

This package is a native Cursor plugin containing a command and an agent skill.
It does not install hooks or an MCP server.

## Requirements

- Cursor with plugin and agent-skill support.
- Python 3.10 or newer.
- Either a SkillOpt source checkout or an installed `skillopt-sleep` command.
- For `--backend cursor`, an installed and authenticated Cursor Agent CLI
  (`cursor-agent`). The default `mock` backend needs no provider login or spend.

The plugin and transcript harvester work on native Windows. Cursor documents
the Agent CLI for Windows through WSL; run provider-backed `--backend cursor`
inside WSL unless a native `cursor-agent` is available in your environment.

## Install the local plugin

Clone the repository, then run the installer for your platform.

macOS or Linux:

```bash
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt
bash plugins/cursor/install.sh
export SKILLOPT_SLEEP_REPO="$(pwd)"
```

Windows PowerShell:

```powershell
git clone https://github.com/microsoft/SkillOpt.git
Set-Location SkillOpt
powershell -File plugins/cursor/install.ps1
[System.Environment]::SetEnvironmentVariable("SKILLOPT_SLEEP_REPO", "$(pwd)", "User")
```

The installer copies the plugin to
`~/.cursor/plugins/local/skillopt-sleep` (or
`%USERPROFILE%\.cursor\plugins\local\skillopt-sleep`). Quit and reopen Cursor
after changing user environment variables, then confirm that **SkillOpt-Sleep**
appears in Settings > Plugins under Installed.

The plugin and engine have separate installation boundaries. The copied plugin
teaches Cursor how to operate SkillOpt-Sleep; the engine still runs from the
source checkout through `plugins/run-sleep.sh` / `plugins/run-sleep.ps1`, or
from an installed command:

```bash
uv tool install skillopt
# or: python -m pip install skillopt
```

Use a release that includes Cursor source/backend support when choosing the
installed-command route. The source-checkout route uses the implementation in
the checkout directly.

## Use from Cursor

Run the native command, for example:

```text
/skillopt-sleep status
/skillopt-sleep dry-run --backend mock --max-sessions 5 --max-tasks 3
/skillopt-sleep run --backend cursor --max-sessions 5 --max-tasks 3 --progress
/skillopt-sleep adopt
```

The `skillopt-sleep` agent skill remains independently available if a Cursor
version does not surface plugin commands.

The Cursor-visible target is always:

```text
.cursor/skills/skillopt-sleep-learned/SKILL.md
```

Use `--target-skill-path` with that value on harvest, dry-run, and run commands.
Without an explicit target, the shared engine defaults to a Claude-managed
skill, which Cursor does not load as a project skill.

### Source-checkout commands

macOS or Linux:

```bash
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" status --project "$(pwd)"
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" dry-run \
  --project "$(pwd)" --source cursor --backend mock \
  --target-skill-path .cursor/skills/skillopt-sleep-learned/SKILL.md
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" run \
  --project "$(pwd)" --source cursor --backend cursor \
  --target-skill-path .cursor/skills/skillopt-sleep-learned/SKILL.md \
  --max-sessions 5 --max-tasks 3 --progress
```

Windows PowerShell:

```powershell
powershell -File "$env:SKILLOPT_SLEEP_REPO\plugins\run-sleep.ps1" status --project "$(pwd)"
powershell -File "$env:SKILLOPT_SLEEP_REPO\plugins\run-sleep.ps1" dry-run `
  --project "$(pwd)" --source cursor --backend mock `
  --target-skill-path .cursor/skills/skillopt-sleep-learned/SKILL.md
powershell -File "$env:SKILLOPT_SLEEP_REPO\plugins\run-sleep.ps1" run `
  --project "$(pwd)" --source cursor --backend cursor `
  --target-skill-path .cursor/skills/skillopt-sleep-learned/SKILL.md `
  --max-sessions 5 --max-tasks 3 --progress
```

### Installed-command equivalents

```bash
skillopt-sleep status --project "$(pwd)"
skillopt-sleep dry-run --project "$(pwd)" --source cursor --backend mock \
  --target-skill-path .cursor/skills/skillopt-sleep-learned/SKILL.md
skillopt-sleep run --project "$(pwd)" --source cursor --backend cursor \
  --target-skill-path .cursor/skills/skillopt-sleep-learned/SKILL.md \
  --max-sessions 5 --max-tasks 3 --progress
skillopt-sleep adopt --project "$(pwd)"
```

`--source cursor` reads local JSONL transcripts below
`~/.cursor/projects/<workspace>/agent-transcripts/`. Use
`--cursor-home /path/to/.cursor` for a different Cursor home. Invoked scope
selects the current workspace; `--scope all` includes every Cursor workspace.
The source converter retains user/assistant text, tool names, and explicit turn
errors, but excludes raw tool arguments and outputs.

`--backend cursor` invokes the authenticated Cursor Agent CLI. Use
`--cursor-path /path/to/cursor-agent` or `SKILLOPT_SLEEP_CURSOR_PATH` if it is
not on PATH, and `--model` or `SKILLOPT_SLEEP_CURSOR_MODEL` to override its
model. Ordinary model calls run in read-only Ask mode. Tool-validated replays
run in an isolated temporary workspace and isolated Cursor config. Agent-mode
sandboxing is disabled so the local headless configuration allowlists only the
generated tool shims; file reads, file writes, and MCP tools are denied. The
backend does not use `--force` or automatic MCP approval. Organization-enforced
Cursor policies still apply.

The shared engine also supports `mock`, `claude`, `codex`, `copilot`,
`handoff`, and `azure_openai` backends. Cursor is the native model-driven
choice for this integration; `mock` remains the no-provider default.

## Review sensitive data before provider calls

Harvesting is local and read-only, and `--backend mock` makes no provider calls.
Known secret-shaped strings are redacted from harvested Cursor content, and raw
tool payloads are excluded, but pattern-based redaction is not a guarantee.
A real backend sends truncated transcript excerpts and derived tasks to that
backend's provider for mining, replay, judging, and reflection.

For sensitive work, split the flow at the review boundary:

```bash
skillopt-sleep harvest --project "$(pwd)" --source cursor \
  --target-skill-path .cursor/skills/skillopt-sleep-learned/SKILL.md \
  --max-sessions 5 --max-tasks 3 --output reviewed-tasks.json

skillopt-sleep dry-run --project "$(pwd)" --backend cursor \
  --tasks-file reviewed-tasks.json --progress --json
```

Inspect and redact the JSON, then set its top-level `"reviewed"` field to
`true`. Real backends reject task files that remain unreviewed. Keep raw
transcripts, credentials, and task files out of commits.

## Scheduling

Runs remain user-triggered unless the user explicitly schedules them. Before
scheduling, put the Cursor source and target in
`~/.skillopt-sleep/config.json`, because the scheduler persists the project,
backend, time, and optional auto-adopt flag, but not command-line source or
target overrides:

```json
{
  "transcript_source": "cursor",
  "target_skill_path": ".cursor/skills/skillopt-sleep-learned/SKILL.md",
  "cursor_home": "/absolute/path/to/.cursor",
  "backend": "cursor"
}
```

Then schedule or remove the managed entry:

```bash
skillopt-sleep schedule --project "$(pwd)" --backend cursor --hour 3 --minute 17
skillopt-sleep unschedule --project "$(pwd)"
```

On Unix this uses cron; on Windows it uses Task Scheduler. Scheduled runs stage
proposals for later review by default. Do not add `--auto-adopt` unless the user
has explicitly chosen unattended adoption.

## Adoption and memory

`run` stages accepted proposals under
`<project>/.skillopt-sleep/staging/<timestamp>/`. Read the staged `report.md`
and show the held-out baseline-to-candidate score plus exact edits before
running `adopt`. Adoption backs up an existing target before replacing it.

The shared engine may also propose project `CLAUDE.md` memory updates; existing
memory behavior is unchanged. To restrict a Cursor setup to the explicit Cursor
skill, set `"evolve_memory": false` in `~/.skillopt-sleep/config.json`.

There is deliberately no session-end hook or automatic plugin execution.

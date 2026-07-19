---
name: skillopt-sleep
description: "Use when the user wants Cursor to learn from recent local sessions, asks for an offline sleep or dream cycle, wants to consolidate recurring work into a Cursor skill, or requests SkillOpt-Sleep status, harvest, dry-run, run, scheduling, review, or adoption. Drives the validation-gated skillopt_sleep engine with Cursor transcripts and the optional Cursor Agent CLI backend."
---

# SkillOpt-Sleep for Cursor

SkillOpt-Sleep reviews recent local Cursor sessions, mines recurring tasks,
replays those tasks, and proposes bounded improvements to a project Cursor
skill. With the default gate enabled, a proposal is accepted only when it
improves the held-out score. A normal run stages the proposal for review;
nothing live changes until explicit adoption. There is no model-weight training.

This plugin has no session-end hook and no MCP server. Run the cycle only when
the user asks, or install a schedule only when the user explicitly requests one.

## Cursor target

Always use this project-relative target for Cursor-visible learning:

```text
.cursor/skills/skillopt-sleep-learned/SKILL.md
```

Pass it through `--target-skill-path` on `harvest`, `dry-run`, and `run`.
Without an explicit target, the shared engine uses a Claude-managed skill under
`~/.claude/skills`, which is not the intended Cursor project skill.

The shared engine can also evolve project `CLAUDE.md`. If that secondary memory
target is unwanted, set `"evolve_memory": false` in
`~/.skillopt-sleep/config.json` before running.

## Choose the runner

Use one of these supported command paths consistently:

1. Source checkout on macOS/Linux:
   `bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" <action> ...`
2. Source checkout on Windows:
   `powershell -File "$env:SKILLOPT_SLEEP_REPO\plugins\run-sleep.ps1" <action> ...`
3. Installed engine on any platform:
   `skillopt-sleep <action> ...`

If `SKILLOPT_SLEEP_REPO` is not set and `skillopt-sleep` is unavailable, stop
and explain that the engine must be installed or a SkillOpt checkout must be
selected. Do not substitute a hand-written edit for the engine workflow.

## Core workflow

1. **Harvest** local Cursor JSONL transcripts read-only.
2. **Mine** recurring, checkable task records from session digests.
3. **Replay** tasks under the current skill and memory through the selected
   backend.
4. **Reflect** on failures and propose bounded edits.
5. **Gate** the candidate on held-out real tasks.
6. **Stage** accepted proposals under
   `<project>/.skillopt-sleep/staging/<timestamp>/`.
7. **Adopt** only after review, backing up existing live targets first.

## Commands

Use the installed-command form below, or replace `skillopt-sleep` with the
platform-specific source runner described above.

```bash
TARGET_SKILL=.cursor/skills/skillopt-sleep-learned/SKILL.md

# Inspect current state and the latest staged proposal.
skillopt-sleep status --project "$(pwd)"

# Inspect mined tasks without provider spend.
skillopt-sleep harvest --project "$(pwd)" --source cursor \
  --target-skill-path "$TARGET_SKILL" --max-sessions 5 --max-tasks 3

# First smoke check: deterministic and no provider calls.
skillopt-sleep dry-run --project "$(pwd)" --source cursor --backend mock \
  --target-skill-path "$TARGET_SKILL" --max-sessions 5 --max-tasks 3 --json

# Model-driven optimization through the authenticated Cursor Agent CLI.
skillopt-sleep run --project "$(pwd)" --source cursor --backend cursor \
  --target-skill-path "$TARGET_SKILL" \
  --max-sessions 5 --max-tasks 3 --progress

# Apply the latest accepted staged proposal after review.
skillopt-sleep adopt --project "$(pwd)"
```

Actions are `status`, `harvest`, `dry-run`, `run`, `adopt`, `schedule`, and
`unschedule`.

- Default backend is `mock`, which is deterministic and makes no provider calls.
- `--backend cursor` uses the user's authenticated Cursor Agent CLI budget for
  model-driven mining, replay, judging, and reflection.
- `--source cursor` reads
  `~/.cursor/projects/<workspace>/agent-transcripts/*/*.jsonl`.
- `--cursor-home PATH` overrides the Cursor home used for harvesting.
- `--scope invoked` selects the current workspace; `--scope all` includes every
  Cursor workspace.
- `--cursor-path PATH` or `SKILLOPT_SLEEP_CURSOR_PATH` selects a non-default
  `cursor-agent` executable.
- `--model NAME` or `SKILLOPT_SLEEP_CURSOR_MODEL` overrides the Cursor model.
- Keep live runs bounded with `--max-sessions`, `--max-tasks`, and `--progress`.
- A held-out gain is evidence for that run, not a promise of general improvement.

Available backends are:

- `mock` - deterministic, with no provider calls (default);
- `cursor` - the authenticated Cursor Agent CLI;
- `claude` - the authenticated Claude CLI;
- `codex` - the authenticated Codex CLI;
- `copilot` - the authenticated GitHub Copilot CLI;
- `handoff` - prompt/answer files for an interactive agent session;
- `azure_openai` - the configured Azure OpenAI endpoint.

Ordinary Cursor backend calls use read-only Ask mode. A tool-validated replay
uses an isolated temporary workspace and isolated Cursor config. Agent-mode
sandboxing is disabled so the local headless configuration allowlists only the
generated tool shims; file reads, file writes, and MCP tools are denied. It does
not use `--force` or automatic MCP approval. Organization-enforced Cursor
policies still apply. Do not weaken those boundaries to make a replay pass.

## Reviewable data path

Cursor harvesting retains user/assistant text, tool names, and explicit turn
errors while excluding raw tool arguments, tool outputs, and non-message
records. Known secret-shaped strings are redacted, but pattern-based redaction
cannot guarantee that a transcript is safe to send to a provider.

For sensitive sessions, export tasks before any real-backend replay:

```bash
TARGET_SKILL=.cursor/skills/skillopt-sleep-learned/SKILL.md
skillopt-sleep harvest --project "$(pwd)" --source cursor \
  --target-skill-path "$TARGET_SKILL" \
  --max-sessions 5 --max-tasks 3 --output reviewed-tasks.json
```

Inspect and redact the file, then set its top-level `"reviewed"` field to
`true`. Only then run:

```bash
skillopt-sleep dry-run --project "$(pwd)" --backend cursor \
  --tasks-file reviewed-tasks.json --progress --json
```

Real backends reject task files that remain unreviewed. Never include raw
transcripts, credentials, secrets, or sensitive task content in messages,
commits, or generated summaries.

## Scheduling

Scheduling is opt-in. The scheduler persists project, backend, time, and the
optional auto-adopt flag, but not `--source`, Cursor path/home/model overrides,
or `--target-skill-path`. Before scheduling a Cursor cycle, set at least these values in
`~/.skillopt-sleep/config.json`:

```json
{
  "transcript_source": "cursor",
  "target_skill_path": ".cursor/skills/skillopt-sleep-learned/SKILL.md",
  "backend": "cursor"
}
```

Then run:

```bash
skillopt-sleep schedule --project "$(pwd)" --backend cursor --hour 3 --minute 17
skillopt-sleep unschedule --project "$(pwd)"
```

The scheduler uses cron on Unix and Task Scheduler on Windows. Scheduled runs
stage proposals by default. Use `--auto-adopt` only when the user has explicitly
requested unattended adoption.

## Report results

For `dry-run` and `run`, report:

- session and task counts;
- held-out baseline and candidate scores;
- gate action and accepted/rejected edit counts;
- exact proposed edits;
- staging directory, when one was created.

Read staged `report.md` before summarizing a run. Offer adoption only after the
user reviews an accepted proposal that is still staged. Never claim broad
improvement from one run.

## Hard rules

- Harvest is read-only. Never edit Cursor transcript files.
- Never hand-edit the target skill or `CLAUDE.md` as a substitute for adoption.
- Do not run a real backend on sensitive content without confirming its data
  boundary or using the reviewed-task workflow.
- Do not add a session-end hook or imply that installing this plugin schedules
  anything.
- Show validation evidence before recommending adoption.
- Treat generated edits as proposals, not as source of truth.

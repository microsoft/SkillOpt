---
name: skillopt-sleep
description: "Use when the user wants Codex to self-improve from past usage, asks about a nightly/offline 'sleep' or 'dream' cycle, wants Codex to review past sessions, learn preferences, consolidate memory/skills, run dry-run/run/adopt/status for SkillOpt-Sleep, or schedule background self-optimization. Drives the skillopt_sleep engine: harvest past sessions -> mine recurring tasks -> replay through a selected backend -> consolidate validated memory + skills behind a held-out gate."
---

# SkillOpt-Sleep: usage-driven self-evolution for a local Codex agent

SkillOpt-Sleep gives the user's Codex agent a sleep cycle. On demand or on a
nightly schedule, it reviews past local sessions, re-runs recurring tasks
through the selected backend, and proposes changes to a configured skill and to
the project's `CLAUDE.md`. With the default validation gate enabled, it keeps
only changes that improve a held-out score. Live files change only through
explicit adoption or a user-requested `--auto-adopt`. There is no model-weight
training.

The current shared engine does **not** write `AGENTS.md`. For a Codex-visible
result, always select a Codex skill explicitly with `--target-skill-path` (for
example `.agents/skills/<name>/SKILL.md`). If project `CLAUDE.md` is not a
desired secondary target, set `"evolve_memory": false` in
`~/.skillopt-sleep/config.json` before running.

## When to use

Trigger when the user wants any of:

- Codex to learn from past sessions or get better the more they use it;
- a nightly/scheduled or on-demand sleep/dream/offline self-improvement run;
- to review past sessions and distill recurring tasks;
- to consolidate feedback into memory or managed skills;
- to run `status`, `harvest`, `dry-run`, `run`, or `adopt` for SkillOpt-Sleep.

## The cycle

1. **Harvest** - read local session transcripts according to the engine
   configuration and normalize them into session digests.
2. **Mine** - turn digests into recurring `TaskRecord`s with outcomes and
   checkable references where possible.
3. **Replay** - re-run mined tasks through the selected backend under the
   current skill and memory.
4. **Consolidate** - reflect on failures and propose bounded edits.
5. **Gate** - with the default gate enabled, accept edits only when the held-out
   validation score improves.
6. **Stage** - write the proposal under
   `<project>/.skillopt-sleep/staging/<date>/`; nothing live changes.
7. **Adopt** - explicitly, or through user-requested auto-adopt, copy staged
   files over live files with backups for existing targets.

## How to drive it

Invoke the bundled runner via shell (Codex `exec` has shell access). The runner
finds the engine and a Python >= 3.10 automatically.

```bash
# point at the repo if it isn't auto-detected from CWD:
export SKILLOPT_SLEEP_REPO=/path/to/SkillOpt
TARGET_SKILL=.agents/skills/example/SKILL.md
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" status --project "$(pwd)"
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" harvest --project "$(pwd)" \
  --source codex --target-skill-path "$TARGET_SKILL"
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" dry-run --project "$(pwd)" \
  --source codex --target-skill-path "$TARGET_SKILL" --backend mock
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" run --project "$(pwd)" \
  --source codex --target-skill-path "$TARGET_SKILL" --backend codex \
  --max-sessions 5 --max-tasks 3 --progress
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" adopt --project "$(pwd)"
```

On Windows (CMD / PowerShell):
```cmd
:: CMD
set SKILLOPT_SLEEP_REPO=C:\path\to\SkillOpt-Sleep
"%SKILLOPT_SLEEP_REPO%\plugins\run-sleep.cmd" status --project "%CD%"
```
```powershell
# PowerShell
$env:SKILLOPT_SLEEP_REPO = "C:\path\to\SkillOpt-Sleep"
powershell -File "$env:SKILLOPT_SLEEP_REPO\plugins\run-sleep.ps1" status --project "$(pwd)"
```

Actions are `status`, `harvest`, `dry-run`, `run`, `adopt`, `schedule`, and `unschedule`.

- Default backend is `mock`, which is deterministic and spends no API budget.
- `--backend codex` uses the user's Codex budget for model-driven optimization.
  An accepted held-out gain is run-specific evidence, not a guarantee of
  broader improvement; results depend on the tasks, model, and checks.
- `--source codex` reads Codex Desktop archived sessions from `~/.codex/archived_sessions`;
  use `--codex-home /path/to/.codex` if the archive lives elsewhere.
- `--target-skill-path` is required for a Codex skill target. Without it, the
  shared default is a Claude-managed skill under `~/.claude/skills/`, not an
  `.agents` skill.
- Keep `dry-run --backend mock` as the first smoke check unless the user
  explicitly asked for a real optimization run.

### Scheduling

```bash
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" schedule --project "$(pwd)" \
  --backend codex --hour 3 --minute 17
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" unschedule --project "$(pwd)"
```

The scheduler persists the project, backend, time, and optional auto-adopt flag;
it does not persist `--source` or `--target-skill-path` from this command. Before
scheduling a Codex-targeted run, set `"transcript_source": "codex"` and an
absolute `"target_skill_path"` in `~/.skillopt-sleep/config.json`. On systems
without `crontab`, `schedule` prints a line for manual installation.
`unschedule --all` removes every managed entry.

### All backends

- `--backend mock` — deterministic, no API spend (default)
- `--backend claude` — uses the Claude CLI
- `--backend codex` — uses the Codex CLI
- `--backend copilot` — uses the GitHub Copilot CLI
- `--backend cursor` — uses the Cursor Agent CLI
- `--backend handoff` — emits prompt/answer files for an interactive session
- `--backend azure_openai` — uses the configured Azure OpenAI endpoint

### Additional flags

| Flag | Description |
|------|-------------|
| `--auto-adopt` | Auto-adopt if the gate passes (default: stage only) |
| `--edit-budget N` | Max bounded edits per night (default: 4) |
| `--lookback-hours N` | Harvest window in hours (default: 72) |
| `--json` | Machine-readable JSON output |

### Config keys (`~/.skillopt-sleep/config.json`)

- **`preferences`** — free-text house rules for the optimizer
- **`gate_mode`** — `on` (validation-gated, default) or `off` (greedy)
- **`gate_metric`** — `hard` | `soft` | `mixed` (default)
- **`dream_rollouts`** — >1 for multi-rollout contrastive reflection
- **`recall_k`** — >0 recalls similar past tasks from the archive

### Memory consolidation

The shared sleep cycle consolidates project **memory** (`CLAUDE.md`) and the
selected **skill** (`SKILL.md`) by default. It does not update `AGENTS.md`.
Each target is independently toggleable through `evolve_memory` /
`evolve_skill`, and both are gated by the same held-out validation score.

## Steps

1. Run the requested action; capture stdout.
2. For `dry-run` and `run`, report the held-out baseline -> candidate score,
   gate action, task count, session count, and exact proposed edits.
3. If a staging directory is printed, read `report.md` before summarizing.
4. `run` stages by default; if `--auto-adopt` was explicitly supplied, report
   the paths it updated instead of claiming nothing changed.
5. Offer adoption only after the user has reviewed a still-staged proposal.
6. Never hand-edit the configured `CLAUDE.md` or target skill as a substitute
   for the engine's adopt path; adoption is the safety boundary and backs up
   existing targets first.

## Hard rules

- Harvest is read-only. Do not edit archived sessions or raw transcripts.
- Codex transcript harvesting removes known secret-shaped strings, developer
  instructions, and raw tool payloads, but pattern-based redaction is not a
  guarantee. A real backend still sends truncated transcript/task content to
  its provider. Review sensitive sessions and provider policy first; prefer a
  reviewed `--tasks-file` workflow when the data boundary matters.
- Keep raw secrets, credentials, private user data, and transcript contents out
  of messages, logs, generated artifacts, and commits.
- Show validation evidence before recommending adoption.
- Treat generated edits as proposals, not as source of truth.
- Do not rely on deprecated custom prompts or `/sleep` slash commands for this
  Codex integration. This skill is the entrypoint.

## Validate

```bash
python -m skillopt_sleep dry-run --project "$(pwd)" --source codex \
  --target-skill-path .agents/skills/example/SKILL.md --backend mock --json
python -m skillopt_sleep.experiments.run_gbrain --backend codex \
  --seeds brief-writer --data-root /path/to/gbrain-evals/eval/data/skillopt-v1 \
  --nights 2 --limit-replay 3 --limit-holdout 3
```

In the recorded `brief-writer` gbrain run, the deliberately deficient fixture
went 0.00 -> 1.00 on that run's held-out set. Treat this as reproducible
benchmark evidence for that configuration, not a guarantee for other skills,
tasks, or models; see the
[recorded results](https://github.com/microsoft/SkillOpt/blob/main/docs/sleep/RESULTS.md)
for context and limitations.

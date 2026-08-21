# SkillOpt-Sleep — DeepSeek Harness (dsh) integration

Give your **DeepSeek Harness** agent a nightly **sleep cycle**: it reviews past
sessions offline, replays your recurring tasks on your own API budget, and
consolidates what it learns into validated skills behind a held-out gate. Same
engine as the Claude Code / Codex / Cursor integrations (`skillopt_sleep`),
wired into dsh's plugin system as native tools plus a bundled skill.

DeepSeek Harness is the "everything is a plugin" agent framework
([deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)).
Plugins are TypeScript modules exporting an `apply(ctx)` function that register
capabilities (tools, services, events, settings) on the Cordis context.

## What this integration adds

| Component | Purpose |
|---|---|
| `src/index.js` | dsh plugin entry: registers 7 `skillopt_*` tools + Schemastery config |
| `cordis.patch.yml` | bundle patch layer — drop `dsh-skillopt` into any profile's bundles |
| `skills/skillopt-sleep/SKILL.md` | agent skill: when to use the tools, operating rules, data-boundary rules |
| `scripts/sleep.py` | bootstrap/self-check runner (same command shape the tools use) |
| `package.json` | npm package metadata (bundle manifest) |

## Tools

| Tool | skillopt_sleep action | Behavior |
|---|---|---|
| `skillopt_status` | `status` | state, engine availability, latest staged proposal & report |
| `skillopt_dry_run` | `dry-run` | full preview (harvest+mine+replay), stages nothing |
| `skillopt_run` | `run` | full cycle, stages a proposal (live files unchanged) |
| `skillopt_adopt` | `adopt` | apply latest staged proposal (with backup) — the live-change boundary |
| `skillopt_harvest` | `harvest` | read-only show/export of mined tasks |
| `skillopt_schedule` / `skillopt_unschedule` | `schedule` / `unschedule` | install/remove the nightly cron entry |

## Prerequisites

- DeepSeek Harness (dsh) installed
- Python 3.10+ with the SkillOpt-Sleep engine:

```bash
pip install skillopt          # or use this source checkout
```

## Install

### As a bundle in a profile

Add `dsh-skillopt` to the profile's bundles, or in the profile `cordis.patch.yml`:

```yaml
- insert:
    - id: skillopt
      name: './src/index.js'
      config:
        backend: mock          # or codex / claude / cursor / pi / opencode / handoff …
        project: /path/to/project
        preferences: 'Always use async/await'
```

### Local patch overlay (dev)

Run from a DeepSeek Harness **source checkout** (the official dev workflow,
`pnpm` resolves the workspace `dsh` bin):

```bash
pnpm dsh web --patch ./plugins/dsh/cordis.patch.yml
```

If `dsh` is installed **globally** (npm install -g), use it directly:

```bash
dsh web --patch ./plugins/dsh/cordis.patch.yml
```

Either way the patch inserts the `skillopt` plugin row into the profile; then
ask the agent: "Use skillopt_status to check the sleep cycle state."

## Config keys

| Key | Default | Purpose |
|---|---|---|
| `pythonCmd` | `python` | Python interpreter for the engine |
| `module` | `skillopt_sleep` | engine Python module |
| `project` | — | default project directory |
| `backend` | — | `mock\|claude\|codex\|copilot\|cursor\|pi\|opencode\|handoff\|azure_openai` |
| `source` | — | `claude\|codex\|copilot\|cursor\|pi\|opencode\|auto` |
| `model` | — | backend model override |
| `maxTasks` / `maxSessions` | — | mine/harvest caps |
| `editBudget` | — | bounded edits per cycle |
| `preferences` | — | house rules for the reflection prior |
| `jsonOutput` | `false` | machine-readable JSON output |

Advanced engine keys (`gate_mode`, `gate_metric`, `gate_no_regression`,
`dream_rollouts`, `recall_k`, `evolve_memory`/`evolve_skill`) go in
`~/.skillopt-sleep/config.json` — the same file shared by all integrations.

## Data boundary

- Harvest is read-only; `mock`/`handoff` make no network calls.
- `run` stages proposals; `adopt` is the normal live-change boundary and backs up first.
- Real backends send truncated transcript excerpts and derived tasks to the
  selected provider. For sensitive sessions, export tasks first (`skillopt_harvest`
  with `output=`), redact, set `"reviewed": true`, then replay — real backends
  refuse unreviewed task files.
- Outbound prompts are not guaranteed secret-free; review source & provider policy.

## Validate (no API spend)

```bash
python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves
```

See the [SkillOpt-Sleep documentation](../../docs/sleep/README.md) for recorded
results, limitations, and the supported integration surface.

---
name: skillopt-sleep
description: "Use when the user wants the dsh agent to self-improve from past usage, asks about a nightly/offline 'sleep' or 'dream' cycle, skill/memory consolidation, or says things like 'make my agent better the more I use it', 'review my past sessions', 'learn my preferences', 'consolidate what you learned', 'run the sleep cycle', or wants to schedule background self-optimization. Drives the skillopt_sleep engine through the skillopt_* tools: harvest past sessions -> mine recurring tasks -> replay via a selected backend -> consolidate validated skills behind a held-out gate."
---

# SkillOpt-Sleep: usage-driven self-evolution for the dsh agent

SkillOpt-Sleep is Microsoft's [SkillOpt](https://github.com/microsoft/SkillOpt)
deployment-time companion engine: it reviews your past sessions (harvest), mines
recurring tasks (mine), replays them through a selected backend (replay), and
consolidates what it learns into skill documents behind a **held-out validation
gate** (consolidate).

This skill drives the engine through the 7 `skillopt_*` tools exposed by the
dsh-skillopt plugin. The default `mock` backend makes no model calls, which is
useful for verifying the plumbing; a real backend consumes your API budget.

## When to use

- "make my agent better the more I use it" / "learn my preferences across sessions"
- a one-off **offline self-evolution / sleep / dream** run (immediate or scheduled)
- review past sessions/trajectories and distill recurring tasks
- consolidate feedback into `AGENTS.md` / `SKILL.md` / managed skills
- schedule (cron) the cycle, or adopt a staged proposal

## The cycle (six stages)

1. **Harvest** — read-only scan of supported local session records → digests
2. **Mine** — digests → recurring task records (intent + outcome labels + checkable refs)
3. **Replay** — re-run tasks under the current skill+memory with the selected backend → (hard, soft) scores
4. **Consolidate** — reflect on failures → propose bounded edits → **validation gate** on a held-out slice (default: accept only on strict improvement)
5. **Stage** — write accepted proposals to `<project>/.skillopt-sleep/staging/<timestamp>/`. **Live files are unchanged.** A rejected run still has a report but no proposal files.
6. **Adopt** — explicit (or operator-configured `--auto-adopt`) copies staged files over live ones, backing up first.

## Driving it

Prefer the tools over hand-editing files:

| Tool | Behavior |
|---|---|
| `skillopt_status` | state, engine availability, latest staged proposal & report |
| `skillopt_dry_run` | full preview (harvest+mine+replay), stages nothing |
| `skillopt_run` | full cycle, stages a proposal (live files unchanged by default) |
| `skillopt_adopt` | apply latest staged proposal (with backup) — the live-change boundary |
| `skillopt_harvest` | read-only show/export of mined tasks |
| `skillopt_schedule` / `skillopt_unschedule` | install/remove the nightly cron entry for this project |

Typical flow:

```text
# 1. check state (default mock backend, zero cost)
skillopt_status

# 2. preview the cycle
skillopt_dry_run project=<dir> source=<claude|codex|…>

# 3. real run (consumes the selected backend's API budget)
skillopt_run project=<dir> backend=<codex|claude|…> preferences="Prefer pytest; keep commits imperative."

# 4. review the report, then adopt
skillopt_adopt project=<dir>

# 5. schedule nightly at 03:17
skillopt_schedule project=<dir> hour=3 minute=17 backend=<codex>
```

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `project` | config or cwd | project directory to evolve |
| `backend` | `mock` | `mock\|claude\|codex\|copilot\|cursor\|pi\|opencode\|handoff\|azure_openai` (mock = no model calls) |
| `source` | config | transcript source: `claude\|codex\|copilot\|cursor\|pi\|opencode\|auto` |
| `model` | backend default | replay model override |
| `max_tasks` | 40 | mined-task cap |
| `preferences` | empty | house rules for the reflection prior (e.g. "always use async/await") |

## Configuration (cordis.yml / bundle patch)

```yaml
- insert:
    - id: skillopt
      name: './src/index.js'
      config:
        backend: codex
        project: /path/to/project
        preferences: 'Always use async/await'
        # auto-adopt is OPERATOR-ONLY — the model cannot set it
        autoAdopt: false
```

Advanced engine keys go in `~/.skillopt-sleep/config.json`:
`gate_mode` (on/off), `gate_metric` (hard/soft/mixed), `gate_no_regression`,
`dream_rollouts`, `recall_k`, `evolve_memory` / `evolve_skill`.

## Hard rules

- **Never** hand-edit `AGENTS.md` / `SKILL.md` around `skillopt_adopt`; let the
  engine's explicit adopt (or operator-configured `--auto-adopt`) apply the
  staging manifest, backing up live files first.
- Harvest is read-only; `mock` replay has no side effects.
- Real backends send truncated transcript excerpts and derived tasks to the
  selected provider for mining/replay/judging/reflection. For sensitive
  sessions, export tasks first (`skillopt_harvest output=<file>`), redact, set
  the top-level `"reviewed"` to `true`, then replay with `--tasks-file`; real
  backends refuse unreviewed task files.
- Show the user the **held-out baseline → candidate** score and the exact
  proposed edits before suggesting adoption. Evidence before adoption.

## Validate / demo (no API spend)

```bash
pip install skillopt
python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves
```

Deterministic synthetic demo: the score rises and the gate blocks a regression.
It validates the mechanism, not effectiveness on your own tasks.

See the [SkillOpt-Sleep docs](https://github.com/microsoft/SkillOpt/tree/main/docs/sleep)
for recorded results and limitations.

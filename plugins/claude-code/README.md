# SkillOpt-Sleep (Claude Code plugin)

> Give your local Claude agent a **sleep cycle**. Every night it reviews your
> past sessions offline, replays your recurring tasks on your own API budget,
> and consolidates what it learns into **validated** memory (`CLAUDE.md`) and
> skills (`SKILL.md`). Your agent gets better the more you use it — no
> model-weight training.

SkillOpt-Sleep is the **deployment-time** companion to
[SkillOpt](https://github.com/microsoft/SkillOpt). SkillOpt trains a skill
offline on a benchmark; SkillOpt-Sleep applies the same discipline to *your own
daily usage*: bounded text edits, accepted only through a held-out validation
gate, with rejected candidates recorded in the cycle report for review.

It synthesizes three ideas:

| Idea | Contribution |
|---|---|
| **SkillOpt** | skill/memory = trainable text; bounded add/delete/replace edits; **held-out gate** keeps only changes that help. |
| **Claude Dreams** | offline consolidation over past sessions; input never mutated; output **reviewed then adopted**. |
| **Agent sleep** | periodic offline replay turns short-term episodes into long-term skill. |

## What it does (one "night")

```
harvest ~/.claude transcripts → mine recurring tasks → replay offline
   → consolidate (reflect → bounded edit → GATE) → stage proposal → (you) adopt
```

Nothing live is modified until **you** run `/skillopt-sleep adopt` (the Dreams "review,
then adopt or discard" contract). Every adopt backs up the prior file first.

## Install

**Requirements:** Python ≥ 3.10. A real CLI backend additionally requires its
corresponding `claude`, `codex`, or `cursor-agent` executable on `PATH` and
authenticated.

```bash
# 1) get the code (the plugin ships inside the SkillOpt repo)
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt

# 2) add the plugin to Claude Code as a local marketplace
/plugin marketplace add ./plugins/claude-code
/plugin install skillopt-sleep@skillopt-sleep

# 3) verify
/skillopt-sleep status
```

The plugin's bundled runner (`scripts/sleep.sh`) auto-selects a Python ≥ 3.10
interpreter and calls the `skillopt_sleep` engine. A source checkout needs no
`pip install`. If the marketplace cache does not contain a usable source tree,
the shared runner falls back first to a `skillopt-sleep` executable on `PATH`
(including `uv tool`/`pipx` installs), then to an importable Python module. Use
`uv tool install skillopt` or `pip install skillopt` for that fallback.

> **Version note.** This page tracks `main`. PyPI 0.2.0 provides the base Sleep
> CLI, but handoff mode and `--preferences` require a source checkout from
> `main` until the next release.

## Quick start

```bash
# from inside any project you use with Claude Code:
/skillopt-sleep dry-run     # preview what it would learn; no changes staged
/skillopt-sleep run         # full cycle: stages a reviewed proposal (still no live edits)
/skillopt-sleep status      # see history + the latest staged proposal
/skillopt-sleep adopt       # apply the staged proposal to CLAUDE.md / SKILL.md (with backup)

/skillopt-sleep-handoff run # same cycle, but THIS session answers the model calls
                            # (no claude -p subprocess, no API key — subscription-friendly)
```

Or call the engine directly (Python ≥ 3.10):

```bash
python -m skillopt_sleep run --project "$(pwd)" --scope invoked --backend mock
python -m skillopt_sleep run --project "$(pwd)" --backend claude   # real lift via Claude
python -m skillopt_sleep run --project "$(pwd)" --backend codex    # real lift via Codex
python -m skillopt_sleep run --project "$(pwd)" --backend cursor   # real lift via Cursor Agent
```

Default backend is **`mock`** — deterministic, no API spend — so you can try the
plumbing for free. Switch to `--backend claude`, `--backend codex`, or
`--backend cursor` for
model-driven mining and optimization on your own budget; an accepted gain is
task- and model-dependent, not guaranteed.

### Data boundary for real backends

Harvesting `~/.claude` is local and read-only, and the `mock` backend makes no
provider calls. A real backend sends truncated transcript excerpts and derived
tasks to the selected provider for mining, replay, judging, and reflection.
Outbound prompts are not currently guaranteed to be secret-free. Review your
session data and provider policy before using a real backend on a sensitive
project; the [shared integration guide](../README.md#data-boundary) describes a
reviewed task-file workflow.

### Handoff mode (session answers the model calls)

`--backend handoff` runs the cycle without any model subprocess: the engine
executes the deterministic stages and writes every model call it needs to
`.skillopt-sleep-handoff/PROMPTS.md` + `pending.json` (exit code 3). You (or
the `/skillopt-sleep-handoff` command, which automates the loop with isolated
fresh-context subagents) write each raw answer to `answers/<id>.md` and re-run
the same command; it resumes from the answers and either finishes or stages
the next batch. Typically 3–6 rounds per night.

```bash
python -m skillopt_sleep run --backend handoff --project "$(pwd)"
# ... answer .skillopt-sleep-handoff/PROMPTS.md into answers/<id>.md ...
python -m skillopt_sleep run --backend handoff --project "$(pwd)"   # resume
```

Answer every prompt in a **fresh context** — a session that has already seen
the mined tasks and their references would contaminate the held-out gate.
Details: [the plugins README](../README.md#handoff-backend).

## Does it actually improve? (real models, public benchmark)

SkillOpt-Sleep is validated against [gbrain-evals](https://github.com/garrytan/gbrain-evals)'
public `skillopt-v1` suite — the same benchmark gbrain scores its own skill
optimizer against. We take a deliberately **deficient** skill and run one sleep
night; held-out scoring is done by a local rule judge (no judge-API, no way to
grade its own homework).

| Backend | Seed | Held-out before → after | Nights |
|---|---|---|---|
| **Claude (Haiku 4.5)** | brief-writer | **0.00 → 1.00** | 1 |
| **Codex** | brief-writer | **0.00 → 1.00** | 2 |

Both took a brief-writer with no risks section / no confidence level and, within
1–2 nights, proposed gated edits that lifted the held-out score to perfect —
into the protected `LEARNED` block, nothing else touched. The Codex 2-night
trace even shows the optimizer **diagnosing its own residual failure** and
adding a meta-rule to fix it. See the recorded results and limitations in
[`docs/sleep/RESULTS.md`](../../docs/sleep/RESULTS.md).

Reproduce:

```bash
git clone https://github.com/garrytan/gbrain-evals /tmp/gbrain-evals
python -m skillopt_sleep.experiments.run_gbrain --backend claude --model haiku \
  --seeds brief-writer --data-root /tmp/gbrain-evals/eval/data/skillopt-v1 \
  --nights 1 --limit-replay 3 --limit-holdout 3
python -m skillopt_sleep.experiments.run_gbrain --backend codex \
  --seeds brief-writer --data-root /tmp/gbrain-evals/eval/data/skillopt-v1 \
  --nights 1 --limit-replay 3 --limit-holdout 3
```

## Deterministic proof (no API, no keys)

```bash
python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves
python -m skillopt_sleep.experiments.run_experiment --persona programmer  --assert-improves
```

Each prints the held-out score rising from baseline toward 1.0 as the gate
accepts the general rules your tasks need, and confirms the gate **rejects** an
injected harmful edit. Context for the measured experiments is in
[`docs/sleep/RESULTS.md`](../../docs/sleep/RESULTS.md).

## Schedule it nightly

```bash
/skillopt-sleep schedule --hour 3 --minute 17
/skillopt-sleep unschedule
```

The built-in scheduler creates a managed cron entry and logs under the project.
The scheduled run stages proposals unless `--auto-adopt` is explicitly selected.

## Safety

- **Read-only** harvest of `~/.claude`. `mock` replay has no side effects.
- Proposals are **staged**, never auto-applied (unless you opt in with `--auto-adopt`).
- Every adopt writes a backup under the staging dir's `backup/`.
- `--max-sessions` and `--max-tasks` bound work, but the main CLI does not enforce
  a hard token or wall-clock budget.
- Real backends share truncated session/task content with the selected provider;
  do not assume outbound prompts have been fully redacted.

## Status

The engine, deterministic experiment, Claude/Codex CLI backends, handoff mode,
and staged adoption flow are implemented. Advanced experiment-harness flags are
not automatically available on the nightly CLI; see the
[shared integration reference](../README.md#supported-cli-surface).

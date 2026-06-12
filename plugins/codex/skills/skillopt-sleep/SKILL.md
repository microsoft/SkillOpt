---
name: skillopt-sleep
description: Nightly offline self-evolution for a Codex agent. Reviews past sessions, replays recurring tasks, and consolidates validated memory + skills behind a held-out gate. Use when the user wants Codex to learn from past usage, run a "sleep"/"dream" cycle, or schedule offline self-optimization.
---

# SkillOpt-Sleep (Codex skill)

This skill drives the `skillopt_sleep` engine — an offline "sleep cycle" that
makes a Codex agent better at the user's recurring work without retraining.

## When to use

Trigger when the user wants to: review past sessions, learn their preferences,
consolidate feedback into long-term memory/skills, run a nightly/offline
self-improvement cycle, or adopt a staged proposal.

## How to run it

Invoke the bundled runner via shell (Codex `exec` has shell access). The runner
finds the engine and a Python ≥ 3.10 automatically:

```bash
# point at the repo if it isn't auto-detected from CWD:
export SKILLOPT_SLEEP_REPO=/path/to/SkillOpt-Sleep
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" <action> --project "$(pwd)" --source codex
```

`<action>` ∈ `status | dry-run | run | adopt | harvest`. `--source codex`
reads Codex Desktop archived sessions from `~/.codex/archived_sessions`; use
`--codex-home /path/to/.codex` if the archive lives elsewhere. Use
`--backend codex` for real improvement on the user's own Codex budget (default
`mock` = no spend).

## Steps

1. Run the requested action; capture stdout.
2. For `run`/`dry-run`: read the staged `report.md` it prints and show the user
   the held-out baseline → candidate score and the exact proposed edits.
3. `run` only **stages** a proposal under `<project>/.skillopt-sleep/staging/`;
   nothing live changes until `adopt`. Offer `/sleep adopt`.
4. Never hand-edit the user's `AGENTS.md` / skills yourself — only `adopt` does,
   and it backs up first.

## Validate

```bash
python -m skillopt_sleep.experiments.run_gbrain --backend codex \
  --seeds brief-writer --data-root /path/to/gbrain-evals/eval/data/skillopt-v1 \
  --nights 2 --limit-replay 3 --limit-holdout 3
```
A deficient skill goes 0.00 → 1.00 on a held-out set; the optimizer's edits are
gated on real-task performance.

---
name: skillopt-sleep
description: "Use when the user wants Codex to self-improve from past usage, asks about a nightly/offline 'sleep' or 'dream' cycle, wants Codex to review past sessions, learn preferences, consolidate memory/skills, run dry-run/run/adopt/status for SkillOpt-Sleep, or schedule offline self-optimization. Drives the skillopt_sleep engine: harvest past sessions -> mine recurring tasks -> replay offline -> consolidate validated memory + skills behind a held-out gate."
---

# SkillOpt-Sleep: offline self-evolution for a local Codex agent

SkillOpt-Sleep gives the user's Codex agent a sleep cycle. While the user is
offline or on demand, it reviews past local sessions, re-runs recurring tasks
on the user's own budget, and consolidates what it learns into memory and
skills. It keeps only changes that pass a held-out validation gate, and live
files change only after the user explicitly adopts a staged proposal. There is
no model-weight training.

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
3. **Replay** - re-run mined tasks offline under the current skill and memory.
4. **Consolidate** - reflect on failures and propose bounded edits.
5. **Gate** - accept edits only when the held-out validation score improves.
6. **Stage** - write the proposal under
   `<project>/.skillopt-sleep/staging/<date>/`; nothing live changes.
7. **Adopt** - only after explicit user approval, copy staged files over live
   files with backups.

## How to drive it

Invoke the bundled runner via shell (Codex `exec` has shell access). The runner
finds the engine and a Python >= 3.10 automatically.

```bash
# point at the repo if it isn't auto-detected from CWD:
export SKILLOPT_SLEEP_REPO=/path/to/SkillOpt-Sleep

bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" status --project "$(pwd)"
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" harvest --project "$(pwd)"
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" dry-run --project "$(pwd)" --backend mock
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" run --project "$(pwd)" --backend codex
bash "$SKILLOPT_SLEEP_REPO/plugins/run-sleep.sh" adopt --project "$(pwd)"
```

Actions are `status`, `harvest`, `dry-run`, `run`, and `adopt`.

- Default backend is `mock`, which is deterministic and spends no API budget.
- `--backend codex` uses the user's Codex budget for real improvement.
- Keep `dry-run --backend mock` as the first smoke check unless the user
  explicitly asked for a real optimization run.

## Steps

1. Run the requested action; capture stdout.
2. For `dry-run` and `run`, report the held-out baseline -> candidate score,
   gate action, task count, session count, and exact proposed edits.
3. If a staging directory is printed, read `report.md` before summarizing.
4. `run` only stages a proposal; nothing live changes until `adopt`.
5. Offer adoption only after the user has reviewed the staged proposal.
6. Never hand-edit the user's `AGENTS.md`, memory, or skills as a substitute
   for `adopt`; adoption is the safety boundary and writes backups first.

## Hard rules

- Harvest is read-only. Do not edit archived sessions or raw transcripts.
- Keep raw secrets, credentials, private user data, and unsanitized transcript
  contents out of messages, logs, generated artifacts, and commits.
- Show validation evidence before recommending adoption.
- Treat generated edits as proposals, not as source of truth.
- Do not rely on deprecated custom prompts or `/sleep` slash commands for this
  Codex integration. This skill is the entrypoint.

## Validate

```bash
python -m skillopt_sleep dry-run --project "$(pwd)" --backend mock --json
python -m skillopt_sleep.experiments.run_gbrain --backend codex \
  --seeds brief-writer --data-root /path/to/gbrain-evals/eval/data/skillopt-v1 \
  --nights 2 --limit-replay 3 --limit-holdout 3
```

A deficient skill goes 0.00 -> 1.00 on a held-out set; the optimizer's edits
are gated on real-task performance.

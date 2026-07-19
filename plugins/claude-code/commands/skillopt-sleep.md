---
description: Run or manage the SkillOpt-Sleep self-evolution cycle (review past sessions, replay tasks through a selected backend, consolidate validated memory + skills, or schedule nightly runs)
argument-hint: "[run | dry-run | status | adopt | harvest | schedule | unschedule] (default: status)"
allowed-tools: Bash, Read
---

# /skillopt-sleep — SkillOpt-Sleep nightly self-evolution

You are driving **SkillOpt-Sleep**: a tool that lets this user's Claude agent
improve from past usage by reviewing sessions, replaying recurring tasks, and
consolidating what it learns into **validated** memory (`CLAUDE.md`) and skills
(`SKILL.md`). With the default gate enabled, a change is kept only if it improves
a held-out replay score. Nothing live is modified until adoption unless the
user explicitly requests `--auto-adopt`.

## Requested action: $ARGUMENTS

(If `$ARGUMENTS` is empty, treat it as `status`.)

## How to run it

The engine is the `skillopt_sleep` Python package in this repo. Split
`$ARGUMENTS` into the first action token and its remaining options, then use the
**plugin's bundled runner** so the right interpreter and repo are on the path.
Preserve the user's remaining options (for example `--preferences`, `--backend`,
or `--target-skill-path`) instead of silently dropping them:

```bash
"${CLAUDE_PLUGIN_ROOT}/scripts/sleep.sh" <action> --project "$(pwd)" --scope invoked <remaining options>
```

`<action>` is one of:

| action       | what it does |
|--------------|--------------|
| `status`     | show how many nights have run + the latest staged proposal (READ-ONLY) |
| `dry-run`    | harvest → mine → replay → report, but **stage nothing** (no-staging preview) |
| `run`        | full cycle: **stage** a validation report and any accepted proposal; only explicit `--auto-adopt` may also update live files |
| `adopt`      | apply the latest staged proposal to live `CLAUDE.md` / `SKILL.md` (backs up first) |
| `harvest`    | debug: print the recurring tasks mined from recent sessions |
| `schedule`   | install a nightly cron entry for this project (`--hour --minute`, off-:00 by default) |
| `unschedule` | remove the nightly cron entry (`--all` to remove every managed entry) |

Default backend is `mock` (deterministic, no API spend). To use real budget for
model-driven optimization, add `--backend claude`, `--backend codex`, or
`--backend cursor`. An
accepted gain is evidence on this run's held-out tasks, not a guarantee of
general improvement; results depend on the tasks, model, and checks. To steer
what the optimizer writes, add `--preferences "<your house rules>"`.

## Steps to follow

1. **Run the requested action** via the bundled runner above. Capture stdout and
   stderr.
2. **For `run`:** if it prints a staging directory, `Read` its `report.md` and
   show the user:
   - held-out score: baseline → candidate (evidence on this run's held-out tasks)
   - the gate decision (accept/reject) and the exact edits it proposes
   - where the proposal is staged
3. **For `dry-run`:** no staging directory or `report.md` is created. Summarize
   the score, gate decision, and edits from stdout (or request `--json` when
   machine-readable output is useful).
4. **For `run` that produced an accepted proposal:** inspect whether stdout says
   it was auto-adopted. If not, tell the user nothing live changed and offer
   `/skillopt-sleep adopt`; if it was, report the updated paths explicitly.
5. **For `adopt`:** confirm which live files were updated and that backups were
   written under the staging dir's `backup/`.
6. **Never** edit `CLAUDE.md` or `SKILL.md` yourself — let the engine's explicit
   `adopt` or user-requested `--auto-adopt` path apply its manifest and backup
   behavior. Respect the review gate.

## Safety reminders

- Harvest is **read-only** over `~/.claude`. Replay in `mock` mode runs no
  shell side effects.
- The cycle stages proposals by default; auto-adoption requires explicit opt-in.
- A real backend sends truncated transcript excerpts and derived tasks to its
  provider for mining, replay, judging, and reflection. Pattern-based redaction
  is not a guarantee that outbound prompts are secret-free. For sensitive data,
  use `mock` or first run `harvest --output <file>`, review/redact the file, set
  `"reviewed": true`, and then pass it with `--tasks-file`.
- `schedule` manages a cron entry when `crontab` is available; otherwise it
  prints a line for manual installation.

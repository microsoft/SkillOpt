---
description: Run or manage the SkillOpt-Sleep self-evolution cycle (review past sessions, replay tasks through a selected backend, consolidate validated memory + skills, or schedule nightly runs)
argument-hint: "[natural-language request | run | dry-run | status | adopt | harvest | schedule | unschedule]"
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

## Natural-language requests

`$ARGUMENTS` may be either an explicit SkillOpt-Sleep action or a natural-language request.

When the request is natural language, infer the intended SkillOpt-Sleep action and translate only the constraints explicitly stated by the user.

Use these mappings:

* Requests to improve, optimize, or learn from recent sessions → `run`
* Requests to preview or see what could be improved without applying changes → `dry-run`
* Requests to inspect recent-session tasks → `harvest`
* Requests to see the current optimization state or pending proposal → `status`
* Requests to apply an already-reviewed proposal → `adopt`
* Requests to repeat optimization on a schedule → `schedule`
* Requests to stop scheduled optimization → `unschedule`

Examples:

* `improve my skill using mistakes from my recent sessions`
  → `run`

* `show me what could be improved from this week's sessions`
  → `dry-run` with the appropriate lookback constraint

* `optimize my Python skill`
  → `run` with the appropriate target skill/path when it can be identified safely

* `what did SkillOpt learn from my sessions?`
  → `harvest`

* `show me the latest optimization proposal`
  → `status`

* `apply the changes from the last optimization`
  → `adopt`

* `optimize my skills every night`
  → `schedule`

Do not invent paths, skill names, backends, time ranges, or other configuration values that the user did not provide. Preserve configured defaults when the request does not specify a value.

For ambiguous optimization requests, prefer `dry-run` so the user can review the proposed changes before staging.

Natural-language interpretation must not bypass the existing validation gate, staging mechanism, backup behavior, or explicit adoption requirement.


## How to run it

Interpret `$ARGUMENTS` first.

If it begins with an explicit action (`run`, `dry-run`, `status`, `adopt`,
`harvest`, `schedule`, or `unschedule`), preserve the existing behavior and pass
the remaining options unchanged.

Otherwise, treat `$ARGUMENTS` as a natural-language request. Select the
appropriate SkillOpt-Sleep action using the mappings in **Natural-language
requests** above. Translate only constraints explicitly stated by the user into
supported CLI options. Do not invent paths, skill names, backends, time ranges,
or other configuration values.

Always use the plugin's bundled runner so the correct interpreter and repository
are resolved:

```bash
"${CLAUDE_PLUGIN_ROOT}/scripts/sleep.sh" <action> --project "$(pwd)" --scope invoked <translated options>
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
model-driven optimization, add `--backend claude` or `--backend codex`. An
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
   it was auto-adopted. If not, tell the user nothing live changed, run or cite
   `status`, and offer the exact reviewed mode: `adopt --legacy`, repeatable
   `adopt --skill NAME`, or `adopt --all-skills`. Never imply that bare adopt
   means “adopt everything.” If it was auto-adopted, report the updated paths
   and any still-pending fan-out names explicitly.
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

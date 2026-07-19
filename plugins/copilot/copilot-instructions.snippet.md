<!--
Copy this block into your repo's .github/copilot-instructions.md so Copilot
knows the SkillOpt-Sleep tools exist. (Copilot reads copilot-instructions.md
automatically as ambient guidance.)
-->

## SkillOpt-Sleep (offline self-evolution)

This project has SkillOpt-Sleep available via an MCP server (`skillopt-sleep`).
It gives the agent a nightly "sleep cycle": it reviews past sessions, replays
recurring tasks through a selected backend, and stages validation-gated changes
to project `CLAUDE.md` and a configured `SKILL.md`.

When the user asks to "run the sleep cycle", "review my past sessions", "learn
my preferences", or "make the agent improve from past usage", use the MCP tools:

- `sleep_status` — what's happened + the latest staged proposal
- `sleep_dry_run` — no-staging preview; a real backend still makes provider calls
- `sleep_run` — full cycle, stages a validation-gated proposal by default;
  explicit `auto_adopt` may update live files
- `sleep_adopt` — apply the staged proposal (backs up an existing live file first)
- `sleep_harvest` — list mined recurring tasks
- `sleep_schedule` — install a nightly cron entry (set `hour`/`minute`)
- `sleep_unschedule` — remove the nightly cron entry

### Key parameters (pass as MCP tool arguments)

- `backend` — `mock` (default, no provider calls), `claude`, `codex`, `copilot`, or `cursor`
- `source` — `claude`, `codex`, `cursor`, or `auto` (where to read transcripts)
- `target_skill_path` — explicit SKILL.md to evolve; use this for a skill that
  the current agent actually loads
- `tasks_file` — reviewed TaskRecord JSON (skip harvest); real backends require
  its metadata to contain `"reviewed": true`
- `max_tasks` / `max_sessions` — cap workload
- `auto_adopt` — auto-adopt if the gate passes
- `json` — machine-readable output for programmatic use

### Advanced config (`~/.skillopt-sleep/config.json`)

- `preferences` — free-text house rules for the optimizer
- `gate_mode` — `on` (default) or `off`; `dream_rollouts` — >1 for more signal
- `evolve_memory` / `evolve_skill` — toggle which docs consolidate

Always show the user the held-out baseline → candidate score and the proposed
edits before suggesting `sleep_adopt`. Never hand-edit the user's memory/skill
files; use `sleep_adopt` (or an explicitly requested `auto_adopt`) so the engine
applies its staging manifest and backup behavior.

Harvesting is local and read-only, and `backend: "mock"` makes no provider
calls. A real backend sends truncated transcript excerpts and derived tasks to
the selected provider; outbound prompts are not guaranteed to be secret-free.
Review sensitive data and provider policy before selecting a real backend.

`sleep_schedule` persists only the project, backend, time, and optional
auto-adopt setting. Put a non-default transcript source or target skill in
`~/.skillopt-sleep/config.json` before scheduling it.

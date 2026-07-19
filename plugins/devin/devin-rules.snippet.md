# SkillOpt-Sleep (Devin)

You have access to a nightly self-evolution cycle via the `skillopt-sleep` MCP
server. Use these tools to improve your long-term skills over time:

- **`sleep_status`** — refresh the converted local cache, then show how many
  nights have run and the latest staged proposal
- **`sleep_dry_run`** — refresh the converted local cache and preview a cycle
  without engine staging/adoption; a real backend still makes provider calls
- **`sleep_run`** — run a full cycle; stages a proposal by default, while an
  explicit `auto_adopt` may also update live files
- **`sleep_adopt`** — apply the staged proposal, then sync the managed skill to
  `.devin/skills/skillopt-sleep-learned/SKILL.md` when `project` is the Devin
  workspace and that workspace already contains a `.devin/` directory
- **`sleep_harvest`** — debug: list the recurring tasks mined from recent sessions
- **`sleep_schedule`** / **`sleep_unschedule`** — low-level shared-engine cron
  controls; the current scheduled command does not run Devin's conversion step,
  so do not use it as an unattended Devin-harvest workflow

When a user asks about the sleep cycle or skill evolution, prefer calling these
tools over explaining the concept.

Always pass the absolute Devin workspace as `project`, especially for
`sleep_adopt`. Default backend is `mock` (no provider calls). The `claude`,
`codex`, `copilot`, and `cursor` backend values use the corresponding installed and
authenticated CLI; they do not require this plugin to implement a separate
API-key flow.

The Devin conversion and mock workflow stay local. A real backend sends
truncated transcript excerpts and derived tasks to the selected provider for
mining, replay, judging, and reflection; conversion is not a guarantee that
outbound prompts contain no secrets. Review local sources and provider policy
before selecting a real backend.

For a reviewed task file, pass `tasks_file`; before using it with a real backend,
inspect/redact it and ensure its metadata contains `"reviewed": true`.

Place this file at `.devin/rules/skillopt-sleep.md` in your workspace.

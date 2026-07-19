# SkillOpt-Sleep integrations

**SkillOpt-Sleep** reviews recent agent sessions, mines recurring tasks, replays
them, and proposes bounded updates to memory and skills. A held-out validation
gate decides whether a proposal is worth staging, and nothing live changes until
the user explicitly adopts it.

The shared engine lives in [`skillopt_sleep/`](../skillopt_sleep) and has no
runtime dependency on the paper's `skillopt/` experiment package.

## Available integrations

Five integrations wrap the shared `skillopt_sleep` CLI. OpenClaw is a separate
reference adaptation with its own backend and setup assumptions.

| Platform | Folder | Mechanism | Status |
|---|---|---|---|
| **Claude Code** | [`claude-code/`](claude-code) | marketplace plugin, commands, skill, and hooks | installable shared-engine integration |
| **Codex** | [`codex/`](codex) | user-level skill and shared runner | installable shared-engine integration |
| **Cursor** | [`cursor/`](cursor) | native command and skill, project skill target, and shared runner | installable shared-engine integration |
| **GitHub Copilot** | [`copilot/`](copilot) | MCP server exposing seven `sleep_*` tools | shared-engine MCP integration |
| **Devin** | [`devin/`](devin) | MCP server plus Devin transcript conversion | shared-engine MCP integration |
| **OpenClaw** | [`openclaw/`](openclaw) | custom DeepSeek/Ollama wrapper | independent reference adaptation; review and adapt before use |

## Install

Clone the repository first unless an installed `skillopt-sleep` CLI is sufficient
for your workflow.

| Platform | Install | Then |
|---|---|---|
| **Claude Code** | from the repository root, `/plugin marketplace add ./plugins/claude-code`, then `/plugin install skillopt-sleep@skillopt-sleep` | `/skillopt-sleep status` |
| **Codex** | `bash plugins/codex/install.sh` | ask Codex to use the `skillopt-sleep` skill |
| **Cursor** | `bash plugins/cursor/install.sh` (macOS/Linux) or `powershell -File plugins/cursor/install.ps1` (Windows) | `/skillopt-sleep status` |
| **Copilot** | register `plugins/copilot/mcp_server.py` using its example MCP config | ask Copilot to run `sleep_status` |
| **Devin** | register `plugins/devin/mcp_server.py` using its example MCP config | ask Devin to run `sleep_status` |
| **OpenClaw** | follow and adapt [`openclaw/README.md`](openclaw/README.md) | validate paths, credentials, and tasks locally |

Python 3.10 or newer is required. Real CLI backends also require the selected
agent CLI to be installed and authenticated.

The shared [`run-sleep.sh`](run-sleep.sh) supports both source checkouts and
installed packages. If it cannot find the repository, it tries the
`skillopt-sleep` executable on `PATH` (including `uv tool`/`pipx` installs), then
an importable `skillopt_sleep` module. Install with `uv tool install skillopt` or
`pip install skillopt` when using that fallback.

> **Version note.** This integration reference tracks `main`. PyPI 0.2.0
> supports the base Sleep CLI, while Cursor source/backend/plugin support,
> handoff, Sleep support for non-Azure OpenAI-compatible endpoints, and
> `--preferences` require a source checkout from `main` until the next release.

## One sleep cycle

```text
harvest supported local sessions → mine recurring tasks → replay tasks
  → reflect and propose bounded edits → validate on held-out real tasks
  → stage proposal → (you) review and adopt
```

The default backend is `mock`: it makes no provider calls and is useful for
checking plumbing. A real backend is required for model-driven mining and genuine
optimization.

## Data boundary

- Harvesting is local and read-only. The `mock` backend has no model-provider
  data path and no API spend.
- A real backend sends truncated transcript excerpts and derived task content to
  the provider selected for mining, replay, judging, and reflection.
- The Cursor source reads local user/assistant message text, explicit turn
  errors, and tool names from `~/.cursor/projects/*/agent-transcripts`; it does
  not retain tool arguments, tool outputs, or other record types. Known
  secret-shaped strings are redacted, but this is defense in depth rather than
  a guarantee that outbound prompts are secret-free.
- The Cursor backend sends prompts through the installed, authenticated
  `cursor-agent` CLI. Ordinary calls use read-only Ask mode; tool-validated
  tasks run in an isolated temporary workspace and Cursor config with only the
  generated local shims allowlisted. Cursor and the model provider selected by
  Cursor can receive the resulting prompt content.
- Outbound prompts are not currently guaranteed to be free of secrets. Do not
  use a third-party provider on sensitive transcripts without reviewing the data
  source and the provider's retention policy.
- For a reviewable workflow, export tasks first, inspect and redact the JSON, set
  its top-level `"reviewed"` field to `true`, and then use the task file with a
  real backend:

  ```bash
  python -m skillopt_sleep harvest --project "$(pwd)" --output reviewed-tasks.json
  python -m skillopt_sleep dry-run --project "$(pwd)" --backend codex \
    --tasks-file reviewed-tasks.json --progress
  ```

  Real backends reject task files that are still marked unreviewed.

For the separate API-key and Azure managed-identity transport boundaries, see
[OpenAI-compatible endpoints](../docs/sleep/openai-compatible-endpoints.md).

## Supported CLI surface

Actions:

| Action | Behavior |
|---|---|
| `status` | show state and the latest staged proposal |
| `dry-run` | harvest, mine, replay, and report; stage nothing |
| `run` | run the full cycle and stage a proposal |
| `adopt` | apply the latest staged proposal, with backups |
| `harvest` | inspect or export mined tasks |
| `schedule` / `unschedule` | install or remove the managed nightly cron entry |

Common implemented flags include:

| Flag | Default | Purpose |
|---|---|---|
| `--backend mock\|claude\|codex\|cursor\|copilot\|handoff\|azure_openai` | `mock` | select who performs model calls |
| `--model NAME` | backend default | select a backend-specific model |
| `--source claude\|codex\|cursor\|auto` | `claude` | select the transcript source; `auto` retains Codex-then-Claude precedence and does not select Cursor |
| `--cursor-home PATH` | `~/.cursor` | override the Cursor transcript home |
| `--cursor-path PATH` | auto-detect `cursor-agent` | select the Cursor Agent CLI executable |
| `--project PATH` | current directory | select the project and invoked harvest scope |
| `--scope invoked\|all` | `invoked` | limit transcript harvesting |
| `--target-skill-path PATH` | managed skill | select a specific `SKILL.md` to stage/adopt |
| `--tasks-file PATH` | none | replay a reviewed task file instead of harvesting |
| `--max-sessions N` / `--max-tasks N` | unset → `3 × tasks` / `40` tasks | bound harvested work; these are not hard token or wall-clock budgets |
| `--edit-budget N` | `4` | cap bounded edits per cycle |
| `--preferences "..."` | empty | add house rules to the reflection prior |
| `--progress` | off | print phase progress to stderr |
| `--auto-adopt` | off | adopt an accepted proposal without a separate command |
| `--json` | off | emit machine-readable output where supported |

The nightly CLI does **not** currently expose `--gate`, `--rollouts-k`,
`--optimizer-model`, `--target-model`, `--budget-tokens`, or `--budget-minutes`.
Do not pass experiment-harness flags to the main CLI.

### Preferences

`--preferences` is the main user-facing steering knob:

```bash
python -m skillopt_sleep run --backend codex --project "$(pwd)" \
  --preferences "Prefer pytest. Keep commit subjects imperative and concise."
```

Preferences guide reflection but remain subject to the validation gate.

### Cursor source and backend

Cursor transcript harvesting is explicit: use `--source cursor` rather than
`--source auto`. Invoked-project scope uses Cursor's recorded workspace path,
with the sanitized storage directory as a fallback; `--scope all` scans every
Cursor workspace under `~/.cursor/projects`. The model-driven backend requires
an installed, authenticated `cursor-agent`; use `--cursor-path`,
`SKILLOPT_SLEEP_CURSOR_PATH`, or the `cursor_path` config key when it is not on
`PATH`, and use `--model` or `SKILLOPT_SLEEP_CURSOR_MODEL` to choose a model.

Target the project skill explicitly so accepted learning becomes visible to
Cursor without changing the plugin's own workflow skill:

```bash
python -m skillopt_sleep run --project "$(pwd)" \
  --source cursor --backend cursor \
  --target-skill-path .cursor/skills/skillopt-sleep-learned/SKILL.md \
  --max-sessions 5 --max-tasks 3 --progress
```

### Advanced config

The JSON/YAML config under `~/.skillopt-sleep/` supports additional engine keys,
including `gate_mode`, `gate_metric`, `dream_rollouts`, `dream_factor`, `recall_k`,
`evolve_memory`, and `evolve_skill`. These are config keys, not aliases for the
unsupported CLI flags listed above. Shipping defaults are conservative:
`gate_mode="on"`, `dream_rollouts=1`, `dream_factor=0`, and `recall_k=0`.

The managed `schedule` command stores only the project, backend, time, and
optional auto-adopt setting. It does not copy `--source`, `--cursor-home`,
`--cursor-path`, `--model`, or `--target-skill-path` into the scheduled command.
For a Cursor schedule, set `transcript_source`, `cursor_home`, `cursor_path`,
`model`, and `target_skill_path` in `~/.skillopt-sleep/config.json` first. Keep
the target project-relative, use an absolute CLI path because cron and Task
Scheduler may have a minimal `PATH`, and confirm that `cursor-agent` is
authenticated for the account that runs the job.

### Handoff backend

`--backend handoff` keeps model subprocesses out of the engine. It writes pending
model calls to `.skillopt-sleep-handoff/PROMPTS.md` and `pending.json`, exits with
code 3, and resumes after answers are placed in `answers/<id>.md`:

```bash
python -m skillopt_sleep run --backend handoff --project "$(pwd)"
# answer each prompt in a fresh context, then run the same command again
```

Answering held-out prompts from a context that has already seen their references
contaminates the validation gate. Claude Code's `/skillopt-sleep-handoff` command
automates the loop with isolated fresh-context subagents.

## Validation

The deterministic no-provider check exercises consolidation and the gate:

```bash
python -m skillopt_sleep.experiments.run_experiment \
  --persona researcher --assert-improves
```

Real-model benchmark results and their limitations are documented in
[`docs/sleep/RESULTS.md`](../docs/sleep/RESULTS.md). The benchmark recipes are not
the shipping CLI defaults.

## Safety summary

- Session harvesting is read-only.
- `mock` replay makes no provider calls.
- `run` stages proposals; `adopt` is the normal live-change boundary.
- Adoption backs up existing target files.
- `--max-sessions` and `--max-tasks` bound work, but the main CLI does not yet
  enforce a hard token or elapsed-time budget.
- Treat real-backend transcript excerpts as data shared with the selected
  provider.

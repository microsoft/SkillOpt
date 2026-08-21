# SkillOpt-Sleep integrations

**SkillOpt-Sleep** reviews recent agent sessions, mines recurring tasks, replays
them, and proposes bounded updates to memory and skills. A held-out validation
gate decides whether a proposal is worth staging, and nothing live changes until
the user explicitly adopts it.

The shared engine lives in [`skillopt_sleep/`](../skillopt_sleep) and has no
runtime dependency on the paper's `skillopt/` experiment package.

## Available integrations

Six integrations wrap the shared `skillopt_sleep` CLI. OpenClaw is a separate
reference adaptation with its own backend and setup assumptions.

| Platform | Folder | Mechanism | Status |
|---|---|---|---|
| **Claude Code** | [`claude-code/`](claude-code) | marketplace plugin, commands, skill, and hooks | installable shared-engine integration |
| **Codex** | [`codex/`](codex) | user-level skill and shared runner | installable shared-engine integration |
| **Cursor** | [`cursor/`](cursor) | native command and skill, project skill target, and shared runner | installable shared-engine integration |
| **GitHub Copilot** | [`copilot/`](copilot) | MCP server exposing seven `sleep_*` tools | shared-engine MCP integration |
| **Devin** | [`devin/`](devin) | MCP server plus Devin transcript conversion | shared-engine MCP integration |
| **DeepSeek Harness** | [`dsh/`](dsh) | Cordis plugin: 7 native `skillopt_*` tools, skill, bundle patch layer | installable shared-engine integration |
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
| **DeepSeek Harness** | add `dsh-skillopt` to the profile's bundles, or `dsh web --patch ./plugins/dsh/cordis.patch.yml` | ask the agent to use `skillopt_status` |
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
> Pi source/backend support, handoff, Sleep support for non-Azure
> OpenAI-compatible endpoints, OpenCode Sleep source/backend support, and
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

- Harvesting is local and read-only. The `mock` and `handoff` backends make no
  network calls; handoff writes prompts for separate, user-controlled completion.
- A real backend sends mining, replay, judging, and reflection prompts derived
  from truncated transcript excerpts and tasks to the selected provider.
- The Cursor source reads local user/assistant message text, explicit turn
  errors, and tool names from `~/.cursor/projects/*/agent-transcripts`; it does
  not retain tool arguments, tool outputs, or other record types. Known
  secret-shaped strings are redacted, but this is defense in depth rather than
  a guarantee that outbound prompts are secret-free.
- The Cursor backend sends prompts through the installed, authenticated
  `cursor-agent` CLI. Ordinary calls use read-only Ask mode in a new empty
  temporary workspace with project file access denied. Cursor tasks containing
  `tool_called` validation fail before Agent mode starts; use another backend for
  those tasks. Cursor and the model provider selected by Cursor can receive the
  resulting prompt content.
- The Pi backend sends prompts through the installed, authenticated Pi CLI to
  the provider configured by the user. It disables tools, skills, context files,
  extensions, prompt templates, themes, and session writes for these calls, but
  retains the user's Pi authentication and model configuration. Pi's offline
  startup mode prevents configured npm/git package installation, package
  updates, and model-catalog refresh; it does not prevent the selected provider
  call. These controls are not a guarantee of permanent or complete isolation.
- The Pi source retains user/assistant text, tool names, and lexical feedback
  found in user text. It excludes thinking, tool arguments, tool outputs, images,
  and unrelated metadata. The absolute project `cwd` from the session header is
  retained for scope filtering and may appear in miner prompts sent to a real
  backend and its provider. Known secret-shaped strings in retained message text
  are redacted only as defense in depth.
- The core `opencode` source reads local OpenCode SQLite history without the
  CLI, authentication, or provider access. See
  [the CLI reference](../docs/reference/cli.md#opencode-source-and-backend) for
  its retained-data boundary.
- The core `opencode` backend uses the installed OpenCode CLI for plain model
  calls. It keeps the user's login and file-based global configuration
  while disabling project configuration, tool use, external plugins, and
  configured MCP servers for those calls. Tool-aware replay and a native
  OpenCode plugin or command are not included yet.
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
| `--backend mock\|claude\|codex\|cursor\|copilot\|pi\|opencode\|handoff\|azure_openai` | `mock` | select who performs model calls |
| `--model NAME` | backend default | select a backend-specific model |
| `--source claude\|codex\|copilot\|cursor\|pi\|opencode\|auto` | `claude` | select the transcript source; `auto` retains Codex-then-Claude precedence and does not select Copilot, Cursor, Pi, or OpenCode |
| `--cursor-home PATH` | `~/.cursor` | override the Cursor transcript home |
| `--cursor-path PATH` | auto-detect `cursor-agent` | select the Cursor Agent CLI executable |
| `--pi-home PATH` | `~/.pi` | select the parent directory containing `agent/sessions` |
| `--pi-path PATH` | auto-detect `pi` | select the Pi coding-agent CLI executable |
| `--opencode-path PATH` | `SKILLOPT_SLEEP_OPENCODE_PATH`, then `opencode` on `PATH`/`PATHEXT` | select the OpenCode CLI executable |
| `--opencode-db PATH` | `OPENCODE_DB`, then `${XDG_DATA_HOME:-~/.local/share}/opencode/opencode.db` | select the OpenCode SQLite history database |
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

For the Cursor backend, `--project` also selects target files, state, and the
staging location, but it does not make that directory the Cursor Agent execution
workspace. The target skill is inserted as prompt text rather than invoked as a
native skill. Real-backend `dry-run` performs the same mining and replay model
calls while suppressing staging, adoption, and persisted state changes. The
current Sleep cycle does not implement fresh-worktree replay; a `replay: mock`
report label describes prompt replay and is independent of `--backend mock`.

### Preferences

`--preferences` is the main user-facing steering knob:

```bash
python -m skillopt_sleep run --backend codex --project "$(pwd)" \
  --preferences "Prefer pytest. Keep commit subjects imperative and concise."
```

Preferences guide reflection but remain subject to the validation gate.

### Pi source and backend

Pi transcript harvesting is explicit: `--source pi` reads session JSONL files
below `~/.pi/agent/sessions`; use `--pi-home` to select the parent directory
that contains `agent/sessions`. This source does not require the Pi CLI or
provider authentication. It retains user/assistant text, tool names, and lexical
feedback found in user text, while excluding thinking, tool arguments, tool
outputs, images, and unrelated metadata. The absolute project `cwd` from the
session header is retained for scope filtering and may appear in miner prompts
sent to a real backend and its provider. Known secret-shaped strings in retained
message text are redacted as defense in depth, not as a guarantee. `--source auto` keeps Codex-then-Claude
precedence and does not select Pi.

The source and backend are independent. `--backend pi` uses a locally installed,
authenticated Pi CLI to make real model-provider calls for mining, replay,
judging, and reflection. Select another executable with `--pi-path` and a model
with `--model`:

```bash
python -m skillopt_sleep run --project "$(pwd)" \
  --source pi --backend pi --pi-path /absolute/path/to/pi \
  --model provider/model --max-sessions 5 --max-tasks 3 --progress
```

Pi calls disable tools, skills, context files, extensions, prompt templates,
themes, and session writes. They still use the user's Pi authentication and
model configuration. Pi's offline startup mode also prevents configured npm/git
package installation, package updates, and model-catalog refresh; it does not
prevent the selected provider call. This is bounded invocation setup rather
than permanent or complete isolation. Transcript-derived prompts reach the
provider configured in Pi; review that provider's data-retention and privacy
policy before using sensitive sessions.

The managed scheduler stores the selected backend but does not persist
`--source`, `--pi-home`, `--pi-path`, or `--model`. Before scheduling Pi, set
`transcript_source`, `pi_home`, `pi_path`, and `model` in
`~/.skillopt-sleep/config.json`; prefer an absolute `pi_path` and verify that the
scheduled account is authenticated.

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
including `gate_mode`, `gate_metric`, `gate_no_regression`, `dream_rollouts`,
`dream_factor`, `recall_k`, `evolve_memory`, and `evolve_skill`. These are config
keys, not aliases for the unsupported CLI flags listed above. Shipping defaults
are conservative: `gate_mode="on"`, `gate_no_regression=false`,
`dream_rollouts=1`, `dream_factor=0`, and `recall_k=0`.

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
- `mock` and `handoff` make no network calls.
- `run` stages proposals; `adopt` is the normal live-change boundary.
- Adoption backs up existing target files.
- `--max-sessions` and `--max-tasks` bound work, but the main CLI does not yet
  enforce a hard token or elapsed-time budget.
- Treat real-backend transcript excerpts as data shared with the selected
  provider.

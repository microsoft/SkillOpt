# SkillOpt-Sleep 😴 — deployment-time companion (preview)

**SkillOpt-Sleep** applies SkillOpt's discipline to your *own daily usage*. It gives a
local coding agent a nightly **sleep cycle** that reviews your past sessions, replays
your recurring tasks on your own API budget, and consolidates what it learns into
**validated** long-term memory and skills — behind a held-out gate, staged for your
review. It requires **no weight training** and adds no separate optimization loop to
normal agent requests.

> **Preview.** This is an early preview we are actively iterating on; interfaces and
> defaults may change. The engine lives in the top-level [`skillopt_sleep/`](https://github.com/microsoft/SkillOpt/tree/main/skillopt_sleep)
> package with **zero dependency** on the paper's `skillopt/` code (the validation gate
> is vendored).

## How it works

One "night":

```
harvest Claude Code / Codex / VS Code Copilot / Cursor / Pi / OpenCode transcripts → mine recurring tasks → replay via the configured backend (isolation varies by backend; mock/handoff make no network calls)
   → consolidate (reflect → bounded edit → GATE on real held-out tasks)
   → stage proposal → (you) adopt
```

It synthesizes **SkillOpt** (validation-gated bounded text edits), **Claude Dreams**
(offline consolidation; review-then-adopt), and the **agent-sleep** idea (short-term
experience → long-term competence).

The optional `gate_no_regression` config key strengthens the aggregate gate.
It defaults to `false` for compatibility; when set to `true`, every validation
task must preserve or improve its score under the configured `gate_metric`.
The check applies to intermediate skill and memory candidates and to the fresh
final replay. A missing task result or non-finite task score also blocks the
candidate; an absent numeric score aborts evaluation. Task-level changes are
included in `report.md`, `report.json`, `diagnostics.json`, the CLI's `--json`
output, and the evidence log.

> **Data boundary.** Harvesting is local and read-only. The `mock` backend makes no
> provider calls. A real backend, however, sends truncated excerpts from harvested
> sessions and derived tasks to the provider you select for mining, replay, judging,
> and reflection. Outbound prompts are not currently guaranteed to be secret-free;
> review your transcript source and provider policy before running on sensitive
> projects. For a reviewable workflow, harvest to a task file, inspect/redact it, mark
> it `"reviewed": true`, and then replay that file with the real backend.
>
> The Cursor source reads local user/assistant message text, explicit turn errors,
> and tool names, but excludes tool arguments, tool outputs, and non-message records.
> Known secret-shaped strings are redacted as defense in depth. The Cursor backend
> sends prompts through `cursor-agent`; ordinary calls use read-only Ask mode in an
> empty temporary workspace with project files denied. Cursor tool-aware replay is
> temporarily disabled pending live permission-boundary validation.
> Cursor and the model provider selected by Cursor may therefore receive
> transcript-derived content.
>
> The VS Code Copilot source reads local user-entered prompts, visible assistant
> Markdown, and tool names from confirmed GitHub Copilot Chat requests. It
> excludes system notifications, reasoning, tool inputs/outputs, rendered
> context, and account/model metadata. Known secret-shaped strings are
> redacted, but this remains defense in depth rather than a guarantee.
>
> The Pi source reads local sessions below `~/.pi/agent/sessions`, retaining
> user/assistant text, tool names, and lexical feedback found in user text. It
> excludes thinking, tool arguments, tool outputs, images, and unrelated
> metadata. The absolute project `cwd` from the session header is retained for
> scope filtering and may appear in miner prompts sent to a real backend and its
> provider. Known secret-shaped strings in retained message text are redacted
> only as defense in depth.
> The Pi backend uses the installed, authenticated Pi CLI to contact the user's
> selected model provider. Calls disable tools, skills, context files, extensions, prompt
> templates, themes, and session writes, while retaining Pi authentication and
> model configuration. Pi's offline startup mode also prevents configured
> npm/git package installation, package updates, and model-catalog refresh; it
> does not prevent the selected provider call. This is not a guarantee of
> permanent or complete isolation. Review the provider's retention and privacy
> policy before sending transcript-derived prompts from sensitive sessions.
>
> By default, each stateful night also writes a local `evidence.jsonl` under
> the project staging tree (beside the report when one is staged); dry-runs
> write evidence under the configured Sleep state directory. The log contains
> best-effort-redacted, per-field-truncated copies
> of miner, replay, judge, and reflection prompts and replies. Treat it as
> sensitive local data and apply an appropriate retention policy. Set
> `"evidence_log": false` to disable it; setting `"redact_secrets": false`
> deliberately disables this defense-in-depth redaction.

## How to use it

### Quickest path: the `skillopt-sleep` CLI (pip)

```bash
pip install skillopt        # installs the engine + the `skillopt-sleep` command
skillopt-sleep dry-run      # harvest + mine + replay, report only; stages nothing
skillopt-sleep run          # a full nightly cycle; the proposal is staged for review
skillopt-sleep status       # show state + the latest staged proposal
skillopt-sleep adopt        # apply the latest staged proposal
skillopt-sleep schedule     # install a nightly cron entry for this project
```

> **Version note.** This page tracks `main`. PyPI 0.2.0 provides the base
> commands above. Cursor source/backend/plugin support, VS Code Copilot
> transcript harvesting, Pi source/backend support, Sleep handoff, non-Azure
> OpenAI-compatible endpoints, OpenCode Sleep source/backend support, and
> `--preferences` landed later and require a source install from `main` until
> the next release.

The per-agent integrations below still come from the repo; the CLI above is the
standalone, pip-only way to run a cycle. Claude Code, Codex, Cursor, Copilot, and
Devin wrap the shared engine. OpenClaw is a separate reference adaptation and has
its own setup.

One engine, thin per-agent shells (see [`plugins/`](https://github.com/microsoft/SkillOpt/tree/main/plugins)):

| Platform | Folder | Install |
|---|---|---|
| **Claude Code** | [`plugins/claude-code`](https://github.com/microsoft/SkillOpt/tree/main/plugins/claude-code) | `/plugin marketplace add ./plugins/claude-code` → `/skillopt-sleep` |
| **Codex** | [`plugins/codex`](https://github.com/microsoft/SkillOpt/tree/main/plugins/codex) | `bash plugins/codex/install.sh` → `skillopt-sleep` skill |
| **Cursor** | [`plugins/cursor`](https://github.com/microsoft/SkillOpt/tree/main/plugins/cursor) | `bash plugins/cursor/install.sh` → `/skillopt-sleep` |
| **Copilot** | [`plugins/copilot`](https://github.com/microsoft/SkillOpt/tree/main/plugins/copilot) | register `plugins/copilot/mcp_server.py` as an MCP server |
| **Devin** | [`plugins/devin`](https://github.com/microsoft/SkillOpt/tree/main/plugins/devin) | register `plugins/devin/mcp_server.py` as an MCP server |
| **OpenClaw** | [`plugins/openclaw`](https://github.com/microsoft/SkillOpt/tree/main/plugins/openclaw) | adapt the reference wrapper and paths for your installation |

### VS Code GitHub Copilot Chat

Use `--source copilot` to harvest local VS Code GitHub Copilot Chat sessions.
SkillOpt auto-detects stable and Insiders VS Code `User/workspaceStorage`
roots, plus the portable root when `VSCODE_PORTABLE` is available. Override
discovery with `--vscode-workspace-storage PATH`. Project scoping comes from
each storage entry's adjacent `workspace.json`, and entries without a safe
local mapping are skipped. `--source auto` retains Codex-then-Claude precedence
and does not select Copilot.

```bash
skillopt-sleep harvest --project "$(pwd)" --source copilot --progress
skillopt-sleep dry-run --project "$(pwd)" --source copilot --backend copilot
```

The source and backend are independent: `--source copilot` reads VS Code's
local history, while `--backend copilot` uses the separately installed and
authenticated GitHub Copilot CLI for mining, replay, judging, and reflection.
Inspect harvested tasks before using a real backend on sensitive projects.

The managed scheduler does not preserve `--source` or
`--vscode-workspace-storage`. Before scheduling this source, put
`"transcript_source": "copilot"` and, for a nonstandard root,
`"vscode_workspace_storage": "/absolute/path/to/workspaceStorage"` in
`~/.skillopt-sleep/config.json`.

### Pi

Pi transcript harvesting and model execution are independent. Use `--source pi`
to read local session JSONL files below `~/.pi/agent/sessions`, or set
`--pi-home` to the parent directory that contains `agent/sessions` (the default
is `~/.pi`). The source alone does not require Pi CLI installation or provider
authentication. Pi is never selected implicitly:
`--source auto` retains Codex-then-Claude precedence.

`--backend pi` uses a locally installed and authenticated Pi CLI for real
model-provider calls during mining, replay, judging, and reflection. Select its
executable with `--pi-path` and override its configured model with `--model`:

```bash
skillopt-sleep run --project "$(pwd)" \
  --source pi --backend pi --pi-path /absolute/path/to/pi \
  --model provider/model --max-sessions 5 --max-tasks 3 --progress
```

These calls disable tools, skills, context files, extensions, prompt templates,
themes, and session writes, but still use the user's Pi authentication and model
configuration. They also enable Pi's offline startup mode to prevent configured
npm/git package installation, package updates, and model-catalog refresh; the
selected model provider is still contacted. Treat those controls as bounded
invocation setup, not permanent or complete isolation. A real Pi backend sends
transcript-derived prompts to the provider configured in Pi; inspect that
provider's retention and privacy policy first. The `mock` and `handoff` backends
make no network calls.

The managed scheduler records the backend but does not preserve `--source`,
`--pi-home`, `--pi-path`, or `--model`. Before scheduling Pi, set
`transcript_source`, `pi_home`, `pi_path`, and `model` in
`~/.skillopt-sleep/config.json`. Use an absolute `pi_path` and verify the
scheduled account's Pi authentication.

### OpenCode

Use `--source opencode` to read local OpenCode SQLite history without launching
the CLI or requiring login or provider access. It is not selected by
`--source auto`. See the
[CLI reference](../reference/cli.md#opencode-source-and-backend) for database
selection and the retained-data boundary.

For model calls, install and configure OpenCode using its
[official documentation](https://opencode.ai/docs/), then confirm the CLI is
available with `opencode --version`.

`--backend opencode` sends SkillOpt's model calls for mining, plain task replay,
judging, and reflection through an installed OpenCode CLI, using the user's
existing login, provider environment variables, and file-based global
configuration. Select a binary and model only when the OpenCode defaults are
not suitable:

```bash
skillopt-sleep run --project "$(pwd)" \
  --source opencode --backend opencode \
  --opencode-path /absolute/path/to/opencode --model provider/model
```

Plain calls disable project configuration, model-initiated tool invocation,
external plugins, and configured MCP servers. SkillOpt stops before the model
call if it cannot confirm that every resolved MCP server is disabled.

Tool-aware replay is disabled by default. Enable it with
`--opencode-tool-replay` or `"opencode_tool_replay": true` for tasks whose rule
judge contains a `tool_called` check. It exposes temporary synthetic tools with
randomized names and fixed results, verifies which tools OpenCode actually
invokes, and denies all other tools. Historical tool arguments and results are
not retained or replayed.

Both modes continue to use OpenCode's normal data directory and file-based
global configuration. Calls may therefore appear in session history, and global
custom JS/TS tools may initialize, although SkillOpt does not allow the model to
invoke them. See the
[CLI reference](../reference/cli.md#opencode-source-and-backend) for complete
configuration, history, and isolation details.

For scheduled runs, configure the source, database, executable, and model in
`~/.skillopt-sleep/config.json` as needed. Set `opencode_tool_replay` to `true`
there to opt in to tool-aware replay.

### Cursor

Cursor transcript harvesting and model execution are independent. Use
`--source cursor` to read
`~/.cursor/projects/<workspace>/agent-transcripts/*/*.jsonl`; `--scope invoked`
uses Cursor's recorded workspace path, with the sanitized storage directory as
a fallback, while `--scope all` scans every Cursor workspace. Use
`--cursor-home` for a different Cursor home. `--source auto` keeps its existing
Codex-then-Claude precedence and does not select Cursor.

`--backend cursor` requires an installed, authenticated `cursor-agent`. If it is
not on `PATH`, select it with `--cursor-path`, `SKILLOPT_SLEEP_CURSOR_PATH`, or
the `cursor_path` config key. Select a model with `--model` or
`SKILLOPT_SLEEP_CURSOR_MODEL`. Point adoption at a project Cursor skill rather
than at the plugin's workflow skill:

```bash
skillopt-sleep run --project "$(pwd)" \
  --source cursor --backend cursor \
  --target-skill-path .cursor/skills/skillopt-sleep-learned/SKILL.md \
  --max-sessions 5 --max-tasks 3 --progress
```

The target skill is supplied to Cursor as prompt text; it is not invoked as a
native skill. `--project` selects transcript scope, target files, state, and
staging, but ordinary Cursor calls cannot inspect that project's files. The
current backend therefore evaluates textual guidance rather than end-to-end
repository, CLI, browser, or service workflows.

Cursor tool-aware replay is temporarily disabled pending live Cursor
permission-boundary validation. If a task contains a `tool_called` check, the
Cursor backend exits nonzero before starting Agent mode and does not stage,
adopt, or advance state. Use another backend for those tasks.

The initial harvest window is 72 hours. Set `--lookback-hours N` explicitly when
older sessions should be considered; `0` scans all history subject to the
session limit. A stateful `run`, even with no mined tasks, advances the harvest
checkpoint. Use `harvest` or `dry-run` to inspect counts first. A real-backend
`dry-run` still incurs provider calls and spend, and session/task limits are not
hard call, token, time, or monetary budgets.

The managed scheduler records only the project, backend, time, and optional
auto-adopt setting. It does not preserve Cursor source, home, CLI path, model, or
target-skill flags. Before `skillopt-sleep schedule --backend cursor`, put
`transcript_source`, `cursor_home`, `cursor_path`, `model`, and
`target_skill_path` in `~/.skillopt-sleep/config.json`. The target may remain
project-relative as `.cursor/skills/skillopt-sleep-learned/SKILL.md`. Use an
absolute `cursor_path` and verify that the scheduled account is already
authenticated, because cron and Task Scheduler may run with a minimal
environment.

To use DeepSeek, vLLM, Ollama, or another Chat Completions server, see
**[OpenAI-compatible endpoints](openai-compatible-endpoints.md)**. That guide also
documents the separate HTTPS-only boundary for Azure managed-identity credentials.

Deterministic proof (no API key):
`python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves`.

### Opt-in: per-skill group reporting

Set `"multi_skill_report": true` in `~/.skillopt-sleep/config.json` to add an
independent gate result and report row for every explicit skill hint mined that
night:

```json
{"multi_skill_report": true}
```

This runs one additional consolidation per group (including a catch-all group when
hinted and unhinted evidence are mixed), so it increases backend calls and token use.
It is reporting-only for now: every group starts from the same managed skill
document, and Sleep does not yet resolve and update several live `SKILL.md` files
automatically. Nights containing only the managed catch-all group keep the existing
single-consolidation behavior.

### Opt-in: experience replay & dream rollouts

Two consolidation mechanisms, both default **off** (behavior is unchanged unless you
enable them). They strengthen the nightly update when your tasks have a clean
correctness signal; the validation gate still governs what ships.

| Config knob | Default | Effect |
|---|---|---|
| `dream_rollouts` | `1` | Run each task K times → learn from the good-vs-bad contrast (contrastive reflection). |
| `recall_k` | `0` | Associative recall — pull the K most-similar past tasks (from a persisted archive) into tonight's dream. |
| `dream_factor` | `0` | Add N lightweight synthetic variants of each task. |

## Results

> 📊 **More results & analysis — the gate-safety stress test, experience-replay
> scaling, and the dream-diversity ablation — are in
> [`docs/sleep/RESULTS.md`](RESULTS.md).** The highlights:

**Controlled experiment recipe (not the shipping CLI defaults).** 5 nights × 10 new
real "today" tasks per night; the full held-out **test** split is scored before night
1 (baseline) and after night 5 (after); optimizer = GPT-5.5; single seed (42). The
experiments use the shipped consolidation and gate components, while the nightly CLI
and benchmark harnesses remain separate entry points. Numbers are absolute held-out
accuracy; **Δ** = `after − baseline` in percentage points.

**(a) End-to-end on real agents — [gbrain-evals](https://github.com/garrytan/gbrain-evals) `skillopt-v1`.**
Deficient seed skills go **0.00 → 1.00** on the held-out set with **both Claude Code
and Codex** as the target agent (all 4 seeds, including a real tool-use loop).

**(b) Experience replay scales the gain — SearchQA** (1,400-item held-out test,
SQuAD exact-match; target = GPT-5.5; **validation-gated**):

| Replay config (`dream_rollouts=5`) | Baseline → After | Δ (pts) |
|---|---|---|
| `recall_k=10` | 0.802 → 0.834 | +3.1 |
| `recall_k=20` | 0.803 → 0.848 | **+4.5** |
| full-history replay *(reference, not a shipping default)* | 0.796 → 0.851 | +5.6 |
| `recall_k=10`, `dream_rollouts=8` *(more dreaming, same recall)* | 0.798 → 0.835 | +3.7 |

The gain rises monotonically with how much relevant past experience is recalled. The
same SearchQA cell **without** the gate (`recall_k=10`) is 0.808 → 0.839 (+3.1).

**(c) Second benchmark — SpreadsheetBench** (280-item held-out test; the agent's
generated openpyxl code is executed and compared cell-by-cell to a golden workbook;
target = GPT-5.4-nano; gate-free + the output-contract guardrail): 0.279 → 0.314 (**+3.6**).

**(d) Honest scope.** These gains hold where tasks recur and have a checkable
correctness signal. On saturated or noisy benchmarks (e.g. a strong model already
near ceiling) the effect is **flat within run-to-run noise** — single-seed baseline
variance here is ±1–2 pts, so treat sub-~1.5 pt differences as noise. The validation
gate keeps the worst case bounded; keep it **on** by default.

## Learn more

See the [SkillOpt documentation index](../index.md), the
[CLI reference](../reference/cli.md), and the integration-specific READMEs under
[`plugins/`](https://github.com/microsoft/SkillOpt/tree/main/plugins).

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
harvest Claude Code / Codex / Cursor transcripts → mine recurring tasks → replay via the configured backend (isolation varies by backend; mock/handoff make no network calls)
   → consolidate (reflect → bounded edit → GATE on real held-out tasks)
   → stage proposal → (you) adopt
```

It synthesizes **SkillOpt** (validation-gated bounded text edits), **Claude Dreams**
(offline consolidation; review-then-adopt), and the **agent-sleep** idea (short-term
experience → long-term competence).

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
> commands above. Cursor source/backend/plugin support, Sleep handoff, non-Azure
> OpenAI-compatible endpoints, and `--preferences` landed later and require a
> source install from `main` until the next release.

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

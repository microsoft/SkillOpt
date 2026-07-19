# CLI Reference

> **Version note.** This reference tracks `main`. PyPI 0.2.0 does not yet
> include the generic research `openai_compatible` backend, Sleep handoff,
> Sleep support for non-Azure OpenAI-compatible endpoints, the Sleep
> `--preferences` flag, or Cursor source/backend/plugin support; use a source
> install from `main` for those features until the next release.

## Training

```bash
python scripts/train.py --config <config.yaml> [overrides...]
# Installed equivalent:
skillopt-train --config <config.yaml> [overrides...]
```

### Arguments

| Argument | Description |
|---|---|
| `--config` | Path to YAML config file (required) |
| `--cfg-options key=value [...]` | Override structured config parameters |

### Examples

```bash
# Basic training
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --out_root outputs/searchqa_run

# With overrides
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options optimizer.learning_rate=16 optimizer.lr_scheduler=linear

# With custom initial skill
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options env.skill_init=skills/my_seed.md
```

## Evaluation

```bash
python scripts/eval_only.py --config <config.yaml> --skill <skill.md>
# Installed equivalent:
skillopt-eval --config <config.yaml> --skill <skill.md>
```

### Arguments

| Argument | Description |
|---|---|
| `--config` | Path to YAML config file (required) |
| `--skill` | Path to skill document to evaluate (required) |
| `--split` | `train`, `valid_seen`, `valid_unseen`, or `all` (default) |
| `--cfg-options` | One or more `section.key=value` overrides |

### Examples

```bash
# Evaluate best skill on test set
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa_run/best_skill.md \
  --split valid_unseen

# Evaluate on validation set
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa_run/best_skill.md \
  --split valid_seen
```

`--skill` consumes the artifact produced by training. Unless `--out_root` is
set for evaluation, `eval_only.py` creates a separate timestamped
`outputs/eval_<env>_<model>_<timestamp>/` directory and writes
`eval_summary.json` there; it does not modify the training run directory.

For the generic OpenAI-compatible research backend, select the role backends
explicitly:

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-chat \
    model.target=deepseek-chat
```

## SkillOpt-Sleep

```bash
skillopt-sleep <action> [options]
# Equivalent from a source checkout:
python -m skillopt_sleep <action> [options]
```

Actions are `run`, `dry-run`, `status`, `adopt`, `harvest`, `schedule`, and
`unschedule`. Common options include:

| Argument | Description |
|---|---|
| `--project PATH` | Project to evolve (default: current directory) |
| `--scope invoked\|all` | Harvest this project or all projects |
| `--source claude\|codex\|cursor\|auto` | Transcript source; `auto` keeps Codex-then-Claude precedence and does not select Cursor |
| `--backend mock\|claude\|codex\|copilot\|cursor\|handoff\|azure_openai` | Replay/optimizer backend |
| `--model NAME` | Backend-specific model override |
| `--cursor-home PATH` | Override `~/.cursor` for Cursor transcript harvesting |
| `--cursor-path PATH` | Path to the installed Cursor Agent CLI |
| `--preferences TEXT` | House rules supplied to reflection |
| `--lookback-hours N` | Initial transcript lookback; `0` scans all history |
| `--max-sessions N` / `--max-tasks N` | Bound the harvested workload |
| `--target-skill-path PATH` | Explicit skill document to stage/adopt |
| `--tasks-file PATH` | Replay a reviewed task JSON file instead of harvesting |
| `--edit-budget N` | Maximum bounded edits for the night |
| `--progress` / `--json` | Progress or machine-readable output |
| `--auto-adopt` | Apply an accepted staged proposal automatically |

### Cursor source and backend

`--source cursor` reads local Cursor JSONL transcripts from
`~/.cursor/projects/<workspace>/agent-transcripts/*/*.jsonl`. Invoked scope uses
Cursor's recorded workspace path, including when `--project` is a nested
directory, and falls back to the sanitized storage name when metadata is not
available. `--scope all` scans every workspace below `cursor_home`. The
harvester retains user/assistant text, explicit turn errors, and tool names,
while excluding tool arguments, tool outputs, and non-message records. It
redacts known secret patterns and filters SkillOpt-generated replay sessions,
but redaction is not a guarantee that outbound prompts contain no sensitive
data.

`--backend cursor` launches an installed, authenticated `cursor-agent`, sends
prompts over stdin, and parses its JSON result. Ordinary model calls use
read-only Ask mode. Replays that validate tool use run in an isolated temporary
workspace with an isolated Cursor config. Agent-mode sandboxing is disabled so
the local headless configuration allowlists only the generated tool shims; file
reads, file writes, and MCP tools are denied. These calls do not use `--force`
or automatic MCP approval. Cursor and the model provider selected by Cursor can
receive the prompt content. Organization-enforced Cursor policies still apply.

Cursor-specific settings are available through the CLI, config, and environment:

| Purpose | CLI | `~/.skillopt-sleep/config.json` | Environment |
|---|---|---|---|
| Transcript home | `--cursor-home PATH` | `"cursor_home": "/path/to/.cursor"` | none |
| Agent executable | `--cursor-path PATH` | `"cursor_path": "/path/to/cursor-agent"` | `SKILLOPT_SLEEP_CURSOR_PATH` |
| Model | `--model NAME` | `"model": "NAME"` | `SKILLOPT_SLEEP_CURSOR_MODEL` |

Target the learned project skill explicitly so accepted updates are visible to
Cursor without modifying the plugin's own `skillopt-sleep` workflow skill:

```bash
skillopt-sleep run --project "$(pwd)" \
  --source cursor --backend cursor \
  --target-skill-path .cursor/skills/skillopt-sleep-learned/SKILL.md \
  --max-sessions 5 --max-tasks 3 --progress
```

The managed `schedule` command persists the project, backend, time, and optional
auto-adopt setting only. It does not copy source, Cursor paths, model, or target
skill flags into the scheduled command. Put `transcript_source`, `cursor_home`,
`cursor_path`, `model`, and `target_skill_path` in the user config before
scheduling Cursor. Keep `target_skill_path` project-relative as
`.cursor/skills/skillopt-sleep-learned/SKILL.md`, prefer an absolute
`cursor_path`, and verify authentication for the scheduled account because cron
and Task Scheduler may have a minimal environment.

Backend-specific setup for compatible endpoints is documented in
[OpenAI-compatible endpoints for SkillOpt-Sleep](../sleep/openai-compatible-endpoints.md).

## WebUI

```bash
python -m skillopt_webui.app [--port PORT] [--share]
```

| Argument | Default | Description |
|---|---|---|
| `--port` | 7860 | Port number |
| `--host` | `0.0.0.0` | Server bind address |
| `--share` | false | Create public Gradio link |

The default host binds every network interface. Use `--host 127.0.0.1` when
the dashboard should be reachable only from the local machine.

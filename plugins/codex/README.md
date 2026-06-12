# SkillOpt-Sleep — Codex integration

Give your **Codex** agent a nightly **sleep cycle**: it reviews past sessions
offline, replays your recurring tasks on your own Codex budget, and consolidates
what it learns into validated memory + skills behind a held-out gate. Same engine
as the Claude Code plugin (`skillopt_sleep`), wrapped for Codex.

> **Verified on Codex:** on the public
> [gbrain-evals](https://github.com/garrytan/gbrain-evals) `skillopt-v1`
> benchmark, a deliberately deficient skill goes **0.00 → 1.00** on a held-out
> set with the Codex backend (incl. the tool-use seed via a real tool loop).
> See [`../../docs/sleep/FINAL_REPORT.md`](../../docs/sleep/FINAL_REPORT.md).

## What Codex supports (and what we use)

Codex (`@openai/codex`) extends via **`AGENTS.md`** instructions, **skills** at
`~/.agents/skills/<name>/SKILL.md`, and **custom prompts** at
`~/.codex/prompts/<name>.md` (invoked as `/<name>`). This integration ships all
three, plus a shared runner.

## Install

```bash
git clone <repo-url> SkillOpt-Sleep
cd SkillOpt-Sleep
bash plugins/codex/install.sh          # installs the /sleep prompt + skill
export SKILLOPT_SLEEP_REPO="$(pwd)"    # so the runner is found from anywhere
```

Requires Python ≥ 3.10 and the `codex` CLI on PATH.

## Use

```text
/sleep status      # what's happened
/sleep dry-run     # safe preview, stages nothing
/sleep run         # full cycle, stages a reviewed proposal (no live edits)
/sleep adopt       # apply the staged proposal (with backup)
```

Or call the engine directly:

```bash
python -m skillopt_sleep dry-run --project "$(pwd)" --source codex --backend mock
python -m skillopt_sleep run --project "$(pwd)" --source codex --backend codex
```

`--source codex` reads Codex Desktop archived sessions from
`~/.codex/archived_sessions`. Use `--codex-home /path/to/.codex` to point at a
different Codex home, or `--source auto` to try Codex archives first and fall
back to Claude Code transcripts. Default backend is `mock` (no API spend).
`--backend codex` uses your Codex budget for real improvement. All the
controllable knobs (`--gate on|off`, `--rollouts-k`, `--budget-tokens`,
`--preferences`, optimizer/target split) work identically — see
[`../../docs/sleep/CONTROLLABLE_DREAMING.md`](../../docs/sleep/CONTROLLABLE_DREAMING.md).

## Notes / status

- Codex's `exec` runs shell, so the real-tool-loop replay (e.g. the
  `tool_called: search` benchmark seed) works natively.
- Codex's standalone *plugin-package manifest* format is not yet a stable public
  spec; this integration uses the documented `AGENTS.md` + skills + prompts
  mechanisms, which are stable. If/when a `codex plugin` package format ships,
  we'll add a one-file manifest.

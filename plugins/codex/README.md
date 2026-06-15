# SkillOpt-Sleep — Codex integration

Give your **Codex** agent a nightly **sleep cycle**: it reviews past sessions
offline, replays your recurring tasks on your own Codex budget, and consolidates
what it learns into validated memory + skills behind a held-out gate. Same engine
as the Claude Code plugin (`skillopt_sleep`), wrapped for Codex.

> **Verified on Codex:** on the public
> [gbrain-evals](https://github.com/garrytan/gbrain-evals) `skillopt-v1`
> benchmark, a deliberately deficient skill goes **0.00 → 1.00** on a held-out
> set with the Codex backend (incl. the tool-use seed via a real tool loop).
> See [the SkillOpt-Sleep guide section](https://microsoft.github.io/SkillOpt/docs/guideline.html#sleep).

## What Codex supports (and what we use)

Codex (`@openai/codex`) extends via **`AGENTS.md`** instructions, **skills** at
`~/.agents/skills/<name>/SKILL.md`, and plugins that can distribute skills.
Custom prompts are deprecated in Codex, so this integration is skill-first: the
installed `skillopt-sleep` skill contains the launch commands and operating
rules. The shared runner remains a plain shell entrypoint that the skill calls.

## Install

```bash
git clone <repo-url> SkillOpt-Sleep
cd SkillOpt-Sleep
bash plugins/codex/install.sh          # installs the skill
export SKILLOPT_SLEEP_REPO="$(pwd)"    # so the runner is found from anywhere
```

If a previous install created `~/.codex/prompts/sleep.md`, the installer moves
that deprecated prompt aside with a `.skillopt-legacy*.bak` suffix.

Requires Python ≥ 3.10 and the `codex` CLI on PATH.

## Use

Mention `$skillopt-sleep` where Codex supports explicit skill mentions, or ask
Codex in natural language:

```text
Use the skillopt-sleep skill to run status for this project.
Use the skillopt-sleep skill to run a dry-run for this project.
Use the skillopt-sleep skill to run the full cycle for this project with the Codex backend.
Use the skillopt-sleep skill to adopt the latest staged proposal.
```

Or call the engine directly:

```bash
python -m skillopt_sleep dry-run --project "$(pwd)" --source codex --backend mock
python -m skillopt_sleep run --project "$(pwd)" --source codex --backend codex \
  --max-sessions 5 --max-tasks 3 --progress
python -m skillopt_sleep run --project "$(pwd)" --source codex --backend codex \
  --target-skill-path .agents/skills/example/SKILL.md \
  --max-sessions 5 --max-tasks 3 --progress
```

`--source codex` reads Codex Desktop archived sessions from
`~/.codex/archived_sessions`. Use `--codex-home /path/to/.codex` to point at a
different Codex home, or `--source auto` to try Codex archives first and fall
back to Claude Code transcripts. Default backend is `mock` (no API spend).
`--backend codex` uses your Codex budget for real improvement. Bound live runs
with `--max-sessions` and `--max-tasks`; add `--progress` because Codex-backed
mining, replay, and reflection can be slow and otherwise quiet. Use
`--target-skill-path` to stage/adopt into a repo-scoped Codex skill such as
`.agents/skills/<name>/SKILL.md`; target runs over-sample mined tasks and
prefer tasks that match the target skill's path, headings, and content. All the
controllable knobs (`--gate on|off`, `--rollouts-k`, `--budget-tokens`,
`--preferences`, optimizer/target split) work identically — see
[the SkillOpt-Sleep guide section](https://microsoft.github.io/SkillOpt/docs/guideline.html#sleep).

For privacy-sensitive projects, split the run into reviewable steps:

```bash
python -m skillopt_sleep harvest --project "$(pwd)" --source codex \
  --target-skill-path .agents/skills/example/SKILL.md \
  --max-sessions 5 --max-tasks 3 \
  --output reviewed-tasks.json

python -m skillopt_sleep dry-run --project "$(pwd)" --backend codex \
  --tasks-file reviewed-tasks.json --progress --json
```

Inspect/redact the JSON and set `"reviewed": true` before using a real backend.
`--tasks-file` skips archive harvest/mining and replays only the reviewed JSON
tasks; real backends refuse task files still marked `"reviewed": false`.

## Notes / status

- Codex's `exec` runs shell, so the real-tool-loop replay (e.g. the
  `tool_called: search` benchmark seed) works natively.
- This integration no longer installs a `.codex/prompts` slash command. Skills
  are the reusable Codex workflow surface; mention `skillopt-sleep` explicitly
  or ask for a sleep/dream/offline self-improvement run and Codex can load the
  skill.

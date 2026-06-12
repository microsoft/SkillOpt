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
python -m skillopt_sleep run --project "$(pwd)" --backend codex
```

Default backend is `mock` (no API spend). `--backend codex` uses your Codex
budget for real improvement. All the controllable knobs (`--gate on|off`,
`--rollouts-k`, `--budget-tokens`, `--preferences`, optimizer/target split) work
identically — see [`../../docs/sleep/CONTROLLABLE_DREAMING.md`](../../docs/sleep/CONTROLLABLE_DREAMING.md).

## Notes / status

- Codex's `exec` runs shell, so the real-tool-loop replay (e.g. the
  `tool_called: search` benchmark seed) works natively.
- This integration no longer installs a `.codex/prompts` slash command. Skills
  are the reusable Codex workflow surface; mention `skillopt-sleep` explicitly
  or ask for a sleep/dream/offline self-improvement run and Codex can load the
  skill.

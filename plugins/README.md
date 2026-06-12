# SkillOpt-Sleep — plugins for Claude Code, Codex, and Copilot

One engine, three thin shells. **SkillOpt-Sleep** gives a local coding agent a
nightly **sleep cycle**: it reviews your past sessions offline, replays your
recurring tasks on your own API budget, and consolidates what it learns into
**validated** long-term memory and skills — behind a held-out gate, staged for
your review. Your agent gets better the more you use it, with no model-weight
training.

It synthesizes three ideas: **SkillOpt** (validation-gated bounded text
optimization — the research in this repo), **Claude Dreams** (offline memory
consolidation; input never mutated; review-then-adopt), and the **agent sleep**
literature (short-term experience → long-term competence).

> **This is an open-source tool, decoupled from the research code.** The engine
> lives in the top-level [`skillopt_sleep/`](../skillopt_sleep) package and has
> **zero dependency** on the paper's `skillopt/` experiment package (the
> validation gate is vendored). You can ship/use it without the research stack.

## The three integrations

| Platform | Folder | Mechanism | Status |
|---|---|---|---|
| **Claude Code** | [`claude-code/`](claude-code) | `.claude-plugin` + `/sleep` command + skill + hooks | full, installable |
| **Codex** | [`codex/`](codex) | user-level `skillopt-sleep` skill + shared runner | full |
| **Copilot** | [`copilot/`](copilot) | MCP server (`sleep_*` tools) + `copilot-instructions` | full (MCP) |

All three call the **same** [`plugins/run-sleep.sh`](run-sleep.sh) → `python -m
skillopt_sleep`, so behaviour is identical everywhere. Per-platform setup is in
each folder's README.

## Quick start (Claude Code)

```bash
git clone <repo-url> && cd SkillOpt-Sleep
# Claude Code:
/plugin marketplace add ./plugins/claude-code
/plugin install skillopt-sleep@skillopt-sleep
/sleep status
```
Codex: `bash plugins/codex/install.sh`.
Copilot: register `plugins/copilot/mcp_server.py` as an MCP server.

## What one "night" does

```
harvest ~/.claude (or session) transcripts → mine recurring tasks → replay offline
   → consolidate (reflect → bounded edit → GATE on real held-out tasks)
   → stage proposal → (you) adopt
```

Nothing live changes until you adopt; every adopt backs up first.

## Controls (work on all platforms)

`--gate on|off` · `--rollouts-k K` (multi-rollout contrastive reflection) ·
`--budget-tokens/--budget-minutes` · `--preferences "..."` · separate
optimizer/target models (`--optimizer-model` / `--target-model`) · slow-update
long-term memory. Full guide:
[`../docs/sleep/CONTROLLABLE_DREAMING.md`](../docs/sleep/CONTROLLABLE_DREAMING.md).

## Does it actually work?

Validated on the public
[gbrain-evals](https://github.com/garrytan/gbrain-evals) `skillopt-v1` benchmark
with **real models on both Claude and Codex**: deficient skills go **0.00 →
1.00** on held-out sets (all 4 seeds incl. a real tool-use loop), cross-model
transfer is positive, and the gate blocks regressions. Full results:
[`../docs/sleep/FINAL_REPORT.md`](../docs/sleep/FINAL_REPORT.md).

Deterministic proof (no API key):
```bash
python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves
```

TITLE:
Add SkillOpt-Sleep: nightly offline self-evolution plugins (Claude Code, Codex, Copilot)

BODY:
## Summary

Adds **SkillOpt-Sleep** — a nightly offline "sleep cycle" that gives a local
coding agent the deployment-time analogue of training: it reviews past sessions,
replays recurring tasks on the user's own API budget, and consolidates what it
learns into **validated** long-term memory and skills behind a held-out gate.
Synthesizes SkillOpt (validation-gated bounded text edits), Claude Dreams
(offline consolidation; review-then-adopt), and the agent-sleep idea
(short-term experience -> long-term competence).

Shipped as plugins for **three agents**, one engine + three thin shells:

- **Claude Code** — `.claude-plugin` + `/sleep` command + skill + hooks
- **Codex** — user-level `skillopt-sleep` skill + shared runner + `install.sh`
- **Copilot** — a stdlib-only MCP server exposing `sleep_*` tools

## Design notes

- **Open-source tool, decoupled from the research code.** The engine lives in the
  new top-level `skillopt_sleep/` package with **zero dependency** on the paper's
  `skillopt/` experiment package (the validation gate is vendored).
- Controllable: optional gate (`--gate on|off`), train(dream)/val(real)/test(real)
  splits, slow-update long-term memory, token/time budget, multi-rollout
  contrastive reflection, multi-objective reward (accuracy/tokens/latency), user
  preferences, and separate optimizer/target models.

## Validation (real models)

On the public [gbrain-evals](https://github.com/garrytan/gbrain-evals)
`skillopt-v1` benchmark, deficient skills go **0.00 -> 1.00** on held-out sets
with **both Claude and Codex** (all 4 seeds, including a real tool-use loop);
cross-model transfer is positive; the gate blocks regressions. Independently
load-tested on a fresh non-benchmark persona ("SQL must always include LIMIT"):
held-out test **0.00 -> 1.00** on both backends. See `docs/sleep/FINAL_REPORT.md`
and `docs/sleep/plugin_load_test.md`.

## Tests

- 29 deterministic unit tests (`tests/test_sleep_engine.py`), no API key required.
- `python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves`
  proves held-out lift and that the gate blocks a harmful edit.

## Test plan

- [ ] `python -m unittest tests.test_sleep_engine` (29 pass)
- [ ] `python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves`
- [ ] Claude Code: `/plugin marketplace add ./plugins/claude-code` -> `/sleep status`
- [ ] Codex: `bash plugins/codex/install.sh`
- [ ] Copilot: MCP server `tools/list` returns the `sleep_*` tools

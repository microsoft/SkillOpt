<!--
Copy this block into your repo's .github/copilot-instructions.md so Copilot
knows the SkillOpt research-engine tools exist. (Copilot reads
copilot-instructions.md automatically as ambient guidance.)
-->

## SkillOpt (research skill-optimization engine)

This repo exposes the core **SkillOpt** training/eval engine via an MCP server
(`skillopt`). SkillOpt is validation-gated, text-space skill optimization: it
reflects on rollouts, makes bounded edits to a skill, and keeps a change only
if it improves a held-out validation set.

When the user asks to "optimize a skill", "train on <benchmark>", "run
SkillOpt", "evaluate this skill", or "what configs can I run", use the MCP
tools:

- `skillopt_list_configs` — list the benchmark YAML configs you can pass as `config`
- `skillopt_train` — run a reflective skill-optimization loop on a config (long-running; spends API/compute budget)
- `skillopt_eval` — evaluate a single skill markdown file on a dataset (no training)

Guidance:
- Always run `skillopt_list_configs` first if you don't already know a valid `config` path.
- `skillopt_train` and `skillopt_eval` are long-running and consume the user's
  model backend/budget — confirm the `config`, `backend`, and model choices
  with the user before launching, and surface the held-out gate result when the
  run finishes.
- For one-off YAML overrides use `cfg_options` (e.g. `seed=123 batch_size=40`);
  for any other underlying flag use `extra_args`.

This is distinct from the **SkillOpt-Sleep** MCP server (`skillopt-sleep`,
`sleep_*` tools), which evolves a local coding agent from past sessions rather
than running the research benchmarks.

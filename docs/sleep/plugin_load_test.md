# SkillOpt-Sleep — plugin load-test (fresh examples)

This records an actual end-to-end load-test of all three plugin shells on a
**brand-new example** (not the gbrain benchmark seeds), run on 2026-06-08.

## The fresh persona

A data analyst whose SQL queries must always include a `LIMIT` clause — built
from scratch for this test. Two forms were used:

1. **Real transcripts** — crafted Claude Code session JSONL where the analyst
   asks for SQL, the agent forgets `LIMIT`, and the user complains ("you forgot
   a LIMIT again", "always cap results"). This exercises the real
   harvest → mine pipeline.
2. **Checkable tasks** — the same intent with a rule judge
   (`regex: (?i)LIMIT\s+100`), so the optimizer can be scored on whether future
   SQL follows the house rule.

## Results

### Shell plumbing (all three drive the engine)

| Shell | What was run | Result |
|---|---|---|
| **Claude Code** (`scripts/sleep.sh`) | `harvest`, full `run`, `adopt` | harvest found 2 sessions → 2 tasks; `run` staged a proposal; `adopt` honored the safety contract (no live change when nothing was accepted) |
| **Codex** (`install.sh` + shared runner) | `install.sh` into a temp HOME | placed the user-level `~/.agents/skills/skillopt-sleep/SKILL.md` skill correctly and moved any legacy custom prompt aside instead of installing one |
| **Copilot** (`mcp_server.py`) | `initialize` → `tools/list` → `tools/call sleep_harvest` | 5 tools listed; `sleep_harvest` returned real engine output (2 sessions → 2 tasks) |

### Genuine improvement (real model, fresh persona)

Optimizer **Claude Sonnet 4.6** → target **Claude Haiku 4.5**, 3-way split
(5 train / 2 val / 5 test), scored on the held-out **test** queries; and the same
fresh persona self-optimized on **Codex**:

| Backend | Held-out **test** (fraction of SQL with `LIMIT 100`) before → after |
|---|---|
| Claude (Sonnet → Haiku) | **0.00 → 1.00** |
| Codex | **0.00 → 1.00** |

In one night each optimizer wrote, into the protected learned block, a rule like:

> *"OVERRIDE: Every SQL query you generate MUST include `LIMIT 100` …"* (Claude)
> *"Hard requirement: every SQL query response must include …"* (Codex)

and the target then applied it to the **unseen** test queries. This is the whole
claim on a task family the engine had never seen: it learned the user's house
rule from their failures and generalized it — confirmed on both backends.

## An honest finding from load-testing

The **first** attempt used `val_fraction=0.34, test_fraction=0.34`, which left
only **1 train task** for an 8-task set — too little signal — so reflect produced
nothing and the night was a no-op (val already 0.75). Re-balancing the split to a
real train pool (5 train) fixed it and produced the 0 → 1.00 result above. This
is exactly the kind of issue that only surfaces when you actually run the thing,
and it motivates a future guardrail: warn when the train pool is too small for
the chosen split fractions.

## Reproduce

The checkable persona run (real Claude):

```python
# see the snippet in docs/sleep/plugin_load_test.md history, or run:
python -m skillopt_sleep.experiments.run_experiment --persona programmer --assert-improves  # deterministic
```

Shell checks:

```bash
# Copilot MCP server
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | SKILLOPT_SLEEP_REPO="$(pwd)" python3 plugins/copilot/mcp_server.py
# Codex skill installer (into a throwaway HOME)
HOME=$(mktemp -d) bash plugins/codex/install.sh
```

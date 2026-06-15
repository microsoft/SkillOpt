# SkillOpt-Sleep 😴 — deployment-time companion (preview)

**SkillOpt-Sleep** applies SkillOpt's discipline to your *own daily usage*. It gives a
local coding agent a nightly **sleep cycle** that reviews your past sessions, replays
your recurring tasks on your own API budget, and consolidates what it learns into
**validated** long-term memory and skills — behind a held-out gate, staged for your
review. The agent gets better the more you use it, with **no weight training** and
**zero inference-time overhead**.

> **Preview.** This is an early preview we are actively iterating on; interfaces and
> defaults may change. The engine lives in the top-level [`skillopt_sleep/`](../../skillopt_sleep)
> package with **zero dependency** on the paper's `skillopt/` code (the validation gate
> is vendored).

## How it works

One "night":

```
harvest Claude Code / Codex transcripts → mine recurring tasks → replay offline
   → consolidate (reflect → bounded edit → GATE on real held-out tasks)
   → stage proposal → (you) adopt
```

It synthesizes **SkillOpt** (validation-gated bounded text edits), **Claude Dreams**
(offline consolidation; review-then-adopt), and the **agent-sleep** idea (short-term
experience → long-term competence).

## How to use it

One engine, thin per-agent shells (see [`plugins/`](../../plugins)):

| Platform | Folder | Install |
|---|---|---|
| **Claude Code** | [`plugins/claude-code`](../../plugins/claude-code) | `/plugin marketplace add ./plugins/claude-code` → `/skillopt-sleep` |
| **Codex** | [`plugins/codex`](../../plugins/codex) | `bash plugins/codex/install.sh` → `skillopt-sleep` skill |
| **Copilot** | [`plugins/copilot`](../../plugins/copilot) | register `plugins/copilot/mcp_server.py` as an MCP server |

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

**Protocol (identical for every row below).** 5 nights × 10 new real "today" tasks
per night; the full held-out **test** split is scored before night 1 (baseline) and
after night 5 (after); optimizer = GPT-5.5; single seed (42); run through the exact
shipped engine (`skillopt_sleep.dream.dream_consolidate`). Numbers are absolute
held-out accuracy; **Δ** = `after − baseline` in percentage points.

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

Full reference (pipeline, the three plugins, the experience-replay knobs) is in the
**[Documentation & Reproduction Guide](https://microsoft.github.io/SkillOpt/docs/guideline.html#sleep)**.

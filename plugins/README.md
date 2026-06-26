# SkillOpt-Sleep — plugins for Claude Code, Codex, Copilot, and Devin

**Your coding agent forgets everything between sessions. SkillOpt-Sleep fixes
that.** While you sleep, it reviews what you did today, notices the rules you
keep repeating ("always add a LIMIT", "answers in `\boxed{}`", "cite the
source"), and writes them into your agent's long-term memory and skills — but
only the rules that actually make it score better on *your own* past tasks. You
wake up to an agent that's better at *your* work, and you approve every change
before it sticks.

One engine, four thin shells. It synthesizes **SkillOpt** (validation-gated
bounded text optimization — the research in this repo), **Claude Dreams**
(offline consolidation; input never mutated; review-then-adopt), and the **agent
sleep** idea (short-term experience → long-term competence).

> **Open-source tool, decoupled from the research.** The engine lives in the
> top-level [`skillopt_sleep/`](../skillopt_sleep) package with **zero
> dependency** on the paper's `skillopt/` experiment code (the validation gate is
> vendored). Use it without the research stack.

---

| Platform | Folder | Mechanism | Status |
|---|---|---|---|
| **Claude Code** | [`claude-code/`](claude-code) | `.claude-plugin` + `/skillopt-sleep` command + skill + hooks | full, installable |
| **Codex** | [`codex/`](codex) | user-level `skillopt-sleep` skill + shared runner | full |
| **Copilot** | [`copilot/`](copilot) | MCP server (`sleep_*` tools) + `copilot-instructions` | full (MCP) |
| **Devin** | [`devin/`](devin) | MCP server (`sleep_*` tools) + Devin ATIF-v1.7 harvest + `.devin/rules` | full (MCP) |

## Install (pick your agent)

| Platform | Install | Then |
|---|---|---|
| **Claude Code** | `/plugin marketplace add microsoft/SkillOpt` → `/plugin install skillopt-sleep` | `/skillopt-sleep status` |
| **Codex** | `git clone` → `bash plugins/codex/install.sh` | `/skillopt-sleep status` |
| **Copilot** | `git clone` → register `plugins/copilot/mcp_server.py` as an MCP server | ask "run the sleep cycle" |
| **Devin** | `git clone` → `devin mcp add skillopt-sleep -- python3 plugins/devin/mcp_server.py` | ask "run the sleep cycle" |

Requirements: Python ≥ 3.10 and the agent's CLI on PATH. All three call the same
[`run-sleep.sh`](run-sleep.sh) → `python -m skillopt_sleep`, so behaviour is
identical everywhere. Default backend is `mock` (no API spend); `--backend
claude|codex|copilot` uses your own budget.

---

## How it works: one "night", in plain terms

```
harvest your past sessions → mine the tasks you keep doing → replay them offline
  → reflect on failures → propose a few rule edits → KEEP only edits that raise
    your held-out score → stage a proposal → (you) review & adopt
```

Nothing live changes until you `adopt`; every adopt backs up the prior file.

### The split that keeps it honest: dream-train / real-val / real-test

This is the heart of the design, borrowed from the SkillOpt paper's
train/selection/test protocol:

| Split | Where it comes from | What it's for |
|---|---|---|
| **train** | your real tasks **+ optional "dreamed" variants** | what the optimizer *learns from*. Over-dreaming here is fine — it's imagination. |
| **val** (selection) | **your real tasks only**, held out | the **gate**: an edit is kept only if it raises this score. Stops overfitting. |
| **test** | **your real tasks only**, held out, never seen during optimization | the **final score** we report. Kept as close to your real usage as possible. |

So you can **dream up extra training examples** to learn a rule robustly, while
the rule is still **judged on real, unseen tasks**. A `dream` task can *never*
land in val or test — that invariant is unit-tested.

---

## What each feature does **for you** (with examples)

Every control below works on all three platforms (pass it after the action,
e.g. `/skillopt-sleep run --rollouts-k 3`).

### `--preferences "..."` — tell it your house rules

The single most useful knob. Free text that steers what the optimizer writes,
as a prior. Use it to encode the conventions you're tired of repeating.

```bash
# A backend engineer:
/skillopt-sleep run --preferences "Always use async/await, never callbacks. \
  Prefer pytest over unittest. Commit subjects in imperative mood under 50 chars."

# A data analyst:
/skillopt-sleep run --preferences "Every SQL query must end with LIMIT 1000 unless \
  I say otherwise. Money in USD with 2 decimals. Prefer CTEs over nested subqueries."

# A researcher:
/skillopt-sleep run --preferences "Cite sources as [Author, Year]. Math answers in \
  \\boxed{}. Keep explanations under 150 words unless I ask for depth."
```
*What it does for you:* the next morning your agent already follows these
without you re-typing them, and the rules are validated against your real tasks
(if a "preference" actually hurts your held-out score, the gate drops it).

### `--gate on|off` — strict vs. greedy

- `on` (default): an edit is kept **only if it raises your held-out score**.
  Safe — blocks plausible-but-wrong rules and reward-hacking.
- `off`: greedy — keep edits without the strict check (still reports whether
  quality moved).

*What it does for you:* leave it `on` for trust. Flip it `off` when you're
exploring and want to see everything the optimizer proposes.

### `--rollouts-k K` — learn from contrast, not just failure

Re-runs each task `K` times and learns from the difference between the **good**
and **bad** attempts, not just a single failure.

```bash
/skillopt-sleep run --rollouts-k 3
```
*What it does for you:* a much stronger signal. If your agent gets a task right 1
time in 3, the optimizer figures out *what the winning attempt did* and makes it
reliable.

### `--optimizer-model` / `--target-model` — optimize cheap, deploy anywhere

Use a strong model to *write* the rules and a cheap model to *run* your tasks.
The learned skill then helps the cheap model — or any model.

```bash
/skillopt-sleep run --optimizer-model sonnet --target-model haiku
```
*What it does for you:* spend a little on a smart optimizer overnight; your
everyday cheap/fast agent inherits the upgrade. (Verified: a skill optimized on
one model lifts a different one — cross-model and even cross-runtime
Codex↔Claude.)

### `--budget-tokens N` / `--budget-minutes M` — cap the spend

You decide how much the nightly "dreaming" costs; it auto-plans how many nights
× how many rollouts fit.

```bash
/skillopt-sleep run --backend claude --budget-tokens 60000
```
*What it does for you:* predictable cost. It stops cleanly when the budget is hit
and tells you what it skipped.

### multi-objective (accuracy ↑, tokens ↓, latency ↓)

The reward can weight not just correctness but **cost and speed**, so a skill can
learn to be cheaper and faster, not only more accurate. *What it does for you:*
"answer directly instead of opening five files" becomes a learned habit.

### `schedule` / `unschedule` — set it and forget it

Built-in nightly scheduling (no manual cron):

```bash
/skillopt-sleep schedule --hour 3 --minute 17     # runs every night for this project
/skillopt-sleep unschedule                        # stop it
```
*What it does for you:* it just gets better while you sleep. The nightly run only
*stages* a proposal — adopting is still your call (or add `--auto-adopt` when you
schedule, if you trust it).

---

## Full action / flag reference

| Action | Does |
|---|---|
| `status` | nights so far + the latest staged proposal (read-only) |
| `dry-run` | harvest→mine→replay→report; **stages nothing** |
| `run` | full cycle; **stages** a proposal; nothing live changes |
| `adopt` | apply the staged proposal to `CLAUDE.md`/`SKILL.md` (backs up first) |
| `harvest` | debug: print the recurring tasks it mined |
| `schedule` / `unschedule` | install/remove the nightly cron entry |

| Flag | Default | Meaning |
|---|---|---|
| `--backend mock\|claude\|codex\|copilot` | `mock` | who runs/optimizes (mock = free) |
| `--preferences "..."` | – | your house rules, as a prior |
| `--gate on\|off` | `on` | strict held-out gate vs. greedy |
| `--rollouts-k K` | `1` | multi-rollout contrastive reflection |
| `--optimizer-model` / `--target-model` | – | split the optimizer from the target |
| `--budget-tokens` / `--budget-minutes` | – | cap the nightly spend |
| `--scope invoked\|all` | `invoked` | this project only, or all projects |
| `--auto-adopt` | off | apply without manual review (power users) |

Deep dive: [the SkillOpt-Sleep guide section](https://microsoft.github.io/SkillOpt/docs/guideline.html#sleep).

---

## Does it actually work?

Yes — measured with **real models on both Claude and Codex**, scored on held-out
tasks the optimizer never trained on:

- **gbrain-evals `skillopt-v1`** (the public suite gbrain scores SkillOpt on):
  deficient skills go **0.00 → 1.00** on all 4 seeds, including a real tool-use
  loop; cross-model transfer is positive; the gate blocks regressions.
  → [the SkillOpt-Sleep guide section](https://microsoft.github.io/SkillOpt/docs/guideline.html#sleep)
- **Academic daily-cases** (math / spreadsheet / search-QA, the paper's 4:1:5
  split with dream-augmented train): see
  [the SkillOpt-Sleep guide section](https://microsoft.github.io/SkillOpt/docs/guideline.html#sleep).
- **Fresh load-test** (a "SQL must always include LIMIT" analyst, built from
  scratch): held-out **0.00 → 1.00** on both backends.
  → [the SkillOpt-Sleep guide section](https://microsoft.github.io/SkillOpt/docs/guideline.html#sleep)

Try the deterministic proof yourself (no API key, no spend):
```bash
python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves
```
It prints the held-out score rising to 1.0 as the gate accepts the right rules,
and confirms the gate **rejects** an injected harmful edit.

---

## Safety

- **Read-only** harvest of your sessions. `mock` replay has no side effects.
- Proposals are **staged**, never auto-applied (unless you opt in with `--auto-adopt`).
- Every adopt writes a backup. Per-night token/time budget caps. Secrets redacted.

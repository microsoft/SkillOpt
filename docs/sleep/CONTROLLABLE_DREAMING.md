# SkillOpt-Sleep — controllable dreaming architecture

The sleep engine is no longer a single fixed pipeline. It is a controllable
offline "dream / imagination" loop the user steers. This documents the knobs
added in the four-stage refactor and how they map to the user's design.

## Transcript sources

Sleep separates the source of past sessions from the backend used to replay and
optimize tasks:

```bash
python -m skillopt_sleep dry-run --project "$(pwd)" --source claude --backend mock
python -m skillopt_sleep dry-run --project "$(pwd)" --source codex --backend mock
python -m skillopt_sleep run --project "$(pwd)" --source codex --backend codex
```

`--source claude` reads Claude Code transcripts from `~/.claude/projects`.
`--source codex` reads Codex Desktop archives from
`~/.codex/archived_sessions`. `--source auto` tries Codex archives first, then
falls back to Claude Code transcripts. Use `--codex-home /path/to/.codex` or
`--claude-home /path/to/.claude` to point at non-default homes.

## The mental model

> Sleep = an offline imagination rollout. Re-run the user's real
> tasks (and dream-augmented variants) many times, look at what went well vs
> badly, distil durable rules, and keep only what survives a real-task check —
> unless the user opts out of that check.

## 1. Data splits — train (dream) / val (real) / test (real)

The anti-overfitting foundation:

| Split | Source | Role |
|---|---|---|
| **train** | real tasks **+ dream-augmented** variants | drives reflection (the imagination pool — over-dreaming is fine) |
| **val** | **real only**, disjoint from test | gates updates (prevents overfitting) |
| **test** | **real only**, disjoint from val | the final held-out measure, kept close to real usage |

Hard guarantee (unit-tested): a task with `origin='dream'` **never** lands in
val or test. `assign_splits(val_fraction, test_fraction)` does the deterministic
3-way split; gbrain's own held-out maps to our `test`.

## 2. The validation gate is optional

`--gate on` (default): an edit is accepted only if it strictly improves the
**val** score — the SkillOpt discipline that blocks regressions and reward
hacking.

`--gate off`: greedy. Edits are kept without the hard val-improvement
requirement (the user decides they don't want hard filtering), but val/test
movement is still reported (`greedy_improved` / `greedy_regressed` /
`greedy_flat`) so nothing is hidden.

## 3. Slow-update — long-term memory, gate-independent

Even with the gate off, the engine runs a **slow-update** at the end of the
nights: it compares behaviour under the first-night vs final skill across the
val tasks and distils durable longitudinal guidance into a **protected field**
(`<!-- SLOW_UPDATE_START --> … <!-- SLOW_UPDATE_END -->`, the same markers as
the main SkillOpt repo). Step-level edits never touch this field. This is the
"short-term experience → long-term memory" consolidation; turning the gate off
does not cost you long-term memory.

## 4. Budget — the user picks the spend

`--budget-tokens N` / `--budget-minutes M`: the engine auto-plans depth
(`nights × rollouts_per_task`) to fit the budget (`plan_depth`). Stops cleanly
when exhausted and logs what it skipped — no silent truncation. The whole thing
is offline imagination on the user's own quota.

## 5. Multi-rollout contrastive reflection — the imagination core

`--rollouts-k K` (K>1): each train task is rolled out K times. The optimizer is
shown the **high-scoring vs low-scoring** attempts of the same task and asked
what the good ones did that the bad ones didn't, distilling a general rule. This
is a far stronger signal than a single failure, and it is exactly the user's
"run it many times, learn from the contrast" idea. Tasks with the highest score
*spread* (some passed, some failed) are the most informative and are prioritised.

## 6. Multi-objective reward — accuracy ↑, tokens ↓, latency ↓

Every rollout records its `tokens` and `latency_ms`.
`multi_objective_reward(w_acc, w_tokens, w_latency)` is a weighted reward so a
skill can be optimised to be **cheaper and faster**, not only more accurate
(cost terms normalised against a reference; default weights = accuracy-only, so
existing behaviour is unchanged). This turns "gets better the more you use it"
into "more accurate, cheaper, and faster the more you use it".

## 7. User preferences as a prior

`--preferences "<free text>"`: injected into the optimizer's reflect prompt as a
prior (set on the optimizer model for dual backends), so the user's stated
preferences steer what rules get written.

## How the knobs compose (one command)

```bash
python -m skillopt.sleep.experiments.run_gbrain \
  --optimizer-backend claude --optimizer-model sonnet \   # strong optimizer
  --target-backend claude --target-model haiku \          # cheap target (transfer)
  --seeds thorough-analyst \
  --gate on \                                              # or off for greedy
  --rollouts-k 2 \                                         # contrastive imagination
  --budget-tokens 60000 \                                  # auto-plan depth
  --preferences "Prefer concise, British English." \       # prior
  --nights 3
```

All of this is exercised by the deterministic test suite (29 tests) and
validated on real Claude + Codex (see `real_api_results.md` / `FINAL_REPORT.md`).

## Real cross-validation of the new features (Claude ⟷ Codex)

Three live runs exercised the new code paths on both runtimes (raw logs under
`docs/sleep/raw/crosscheck_*.txt`):

| # | Config | What it proves | Result |
|---|---|---|---|
| **A** | Claude Sonnet→Haiku, **gate=off**, **rollouts_k=2** | greedy mode + multi-rollout + 3-way split (val & test both reported) | brief-writer **test 0→1.00**, action `greedy_improved`, val=1.0 test=1.0 |
| **B** | **Codex**, gate=on, **rollouts_k=2** | new paths on the other runtime | brief-writer **test 0→1.00**, 2-night `accept_new_best`, val+test reported |
| **C** | Claude Sonnet→Haiku, thorough-analyst, 3 nights | **slow-update** long-term memory fires | test 0→0.33 (val gate holds nights 2–3) and the slow-update distilled a durable meta-rule |

The slow-update guidance C produced is the kind of cross-night lesson the field
is for — note it is general, not task-specific:

> *"On character-constrained tasks (≤1200 chars), plan structure before writing:
> allocate space per point explicitly and cut until the outline fits, then fill —
> never draft freely and trim after."*

Takeaways confirmed live: the **gate-off greedy path**, the **3-way val/test
split**, **multi-rollout** on both runtimes, and the **gate-independent
slow-update** all work with real models on both Claude and Codex.

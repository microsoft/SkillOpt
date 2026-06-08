# SkillOpt Sleep — Claude Code self-evolving plugin (design)

**Status:** approved-for-build (autonomous offline session, 2026-06-07)
**Author:** generated for Yifan Yang, executed autonomously while user is asleep
**Branch:** `feat/claude-code-sleep-plugin` (worktree `my_repo/SkillOpt-sleep`)

---

## 1. One-paragraph summary

`skillopt-sleep` is a Claude Code plugin that gives a user's local Claude
agent a nightly **sleep cycle**. While the user is offline, it (1) **harvests**
the day's real Claude Code session transcripts from `~/.claude`, (2) **mines**
them into discrete *task records* with checkable outcomes, (3) **replays /
"dreams"** those tasks offline using the user's own API budget, and (4) runs
the **SkillOpt optimizer loop** (reflect → bounded edit → held-out gate) to
consolidate short-term experience into long-term **memory** (`CLAUDE.md`) and
**skills** (`SKILL.md`). Only changes that pass a validation gate are kept, and
every change is written to a **review staging area** the user approves before it
touches live config — mirroring Claude Dream's "input store is never modified"
safety contract. The result: an agent that measurably gets better at *this
user's* recurring work, every night, with zero model-weight training.

## 2. Why this is the right synthesis of the three ingredients

| Ingredient | What we take from it | Where it lives in this design |
|---|---|---|
| **SkillOpt** (your paper/code) | Skill = trainable text state; bounded add/delete/replace edits under a textual learning rate; **held-out validation gate**; rejected-edit buffer; epoch-wise slow/meta update. | The `consolidate` stage *is* a single SkillOpt epoch, reusing `skillopt.optimizer.*` and `skillopt.evaluation.gate`. |
| **Claude Dreams** | Async offline job: read a memory store + 1–100 session transcripts → emit a **new, separate** reorganized memory store (dedup / merge / resolve contradictions / surface insights). Input never mutated; output reviewed then adopted or discarded. | The `harvest` + `consolidate-memory` stages and the **staging/adopt** safety model are modeled directly on Dreams. |
| **Agent Sleep paper** (2605.26099) | Agents need periodic offline consolidation: short-term experience buffer → synthetic replay/self-generated data → self-update; "sleep" turns episodes into durable competence. | The whole nightly schedule, the `replay` step, and the short-term→long-term framing. |

The key novel claim this enables for the project (and a future paper section):
**SkillOpt's validation-gated bounded-edit optimizer is the missing "safe
update rule" for Dream-style memory consolidation.** Dreams reorganize memory
but don't *prove* the reorganization helps; the Sleep paper consolidates but
assumes weight updates. SkillOpt-Sleep consolidates **text** (memory + skills)
and **gates each change on replayed task performance**, so nightly evolution is
both weight-free and regression-protected.

## 3. Goals / non-goals

**Goals**
1. A working Claude Code plugin: scheduled (nightly/cron) **and** user-triggered (`/sleep`).
2. Look back over the user's real past prompts & trajectories from local `~/.claude` records.
3. Offline "dream training": re-run mined tasks (mock-env or fresh retry) on the user's budget.
4. Continuous evolution of **memory** (`CLAUDE.md`) and **skills** (`SKILL.md`) via the SkillOpt gate.
5. A reproducible experiment that answers: *does the nightly loop actually improve a held-out score?*
6. Safety: never silently overwrite user config; stage → user approves → adopt.

**Non-goals (now)**
- Codex version (explicitly deferred by user; architecture keeps it pluggable).
- Anthropic managed Dreams API integration (we *emulate* Dreams locally; managed API is a future backend).
- Model fine-tuning / weight updates (out of scope by design — text-only).
- Fully unattended auto-adopt by default (opt-in; default is review-gated).

## 4. The local data we read (verified on this machine)

- **Prompt history:** `~/.claude/history.jsonl` — one JSON/line: `{display, pastedContents, timestamp, project}`. The cross-session list of every prompt the user typed, with project path + epoch-ms timestamp.
- **Full transcripts:** `~/.claude/projects/<path-slug>/<sessionId>.jsonl` — one record/line. Record `type` ∈ {`user`,`assistant`,`mode`,`permission-mode`,`attachment`,`file-history-snapshot`,`last-prompt`,…}. User/assistant records carry `message` (role+content blocks), plus `cwd`, `gitBranch`, `timestamp`, `sessionId`, `version`, `userType`. ~215k transcripts present on this box.
- **Deployment targets we may evolve:**
  - Project memory: `<project>/CLAUDE.md` (and `~/.claude/CLAUDE.md` global).
  - User skills: `~/.claude/skills/<name>/SKILL.md` (frontmatter: `name`, `description`, optional `allowed-tools`, `argument-hint`).
  - Plugin skills under `~/.claude/plugins/...`.

Everything stays **on-disk and local**; the only network calls are the LLM
optimizer/replay calls the user already pays for.

## 5. Architecture

### 5.1 The nightly Sleep Cycle (stages)

```
            ┌────────────────────────── SLEEP CYCLE (one "night") ──────────────────────────┐
            │                                                                                │
 trigger →  │  1.HARVEST     2.MINE          3.REPLAY            4.CONSOLIDATE      5.STAGE   │ → wake report
 (cron or   │  read ~/.claude scan sessions  re-run tasks        SkillOpt epoch:   write to   │
  /sleep)   │  transcripts → → task records   offline (mock or    reflect→edit→     .skillopt-│
            │  + history     w/ outcomes &    fresh retry) under  GATE on held-out  sleep/    │
            │                checkable refs   current skill/mem    replay split      staging/ │
            │                                                                          ↓      │
            │                                              6.ADOPT (opt-in / user-approved)   │
            └────────────────────────────────────────────────────────────────────────────────┘
```

**1. Harvest** (`harvest.py`)
Read `history.jsonl` + per-project transcript JSONLs for a time window
(default: since last sleep, fallback last 24–72h). Group by project (`cwd` /
`project`). Emit normalized `SessionDigest` objects: ordered user prompts,
assistant final texts, tool-call summary, files touched (from
`file-history-snapshot`), git branch, errors seen, and **user-feedback signals**
(e.g. "still broken", "that's wrong", "perfect", re-asks of the same thing).

**2. Mine** (`mine.py`)
Turn digests into `TaskRecord`s — the unit the optimizer trains on. A task is a
self-contained intent (the user's request) plus an *outcome label* and, where
possible, a **checkable reference**:
- *Explicit success/failure* from feedback signals ("works now" after N retries → the early attempts are failures, the fix is the success exemplar).
- *Self-consistency check*: re-derivable answers (math, lookups) get a reference; open-ended ones get an LLM-judge rubric instead.
- Each TaskRecord: `{id, project, intent, context_excerpt, attempted_solution, outcome ∈ {success,fail,mixed}, reference_kind ∈ {exact, rubric, none}, reference, tags}`.
Mining is itself an LLM call (the **miner**), prompt-tunable, with a deterministic regex/heuristic fallback for offline/no-key runs.

**3. Replay / "Dream"** (`replay.py`)
For mined tasks, re-run the intent **offline** under the *current* skill+memory
to get a fresh trajectory & score. Two modes:
- `mock` (default, safe): reconstruct a sandboxed prompt from the task's captured context (no live repo mutation, no network side effects) and run the target model. Deterministic, cheap, safe to run unattended.
- `fresh` (opt-in): actually re-attempt in a throwaway git worktree of the project. Higher fidelity, heavier, never touches the user's working tree.
Scoring: exact-match / substring for `exact` refs; LLM-judge (0–1) for `rubric` refs; this yields the `hard`/`soft` scores SkillOpt already expects.

**4. Consolidate** (`consolidate.py`) — *this is one SkillOpt epoch*
Reuse the existing optimizer pieces rather than reinventing:
- `reflect`: partition replayed tasks into failure/success minibatches → propose add/delete/replace edits to **skill** and a parallel proposer for **memory** (`CLAUDE.md`). (Memory consolidation also does Dream-style dedup/merge/contradiction-resolution over existing `CLAUDE.md` lines.)
- `aggregate` + `rank_and_select` under an **edit budget** (textual learning rate).
- `apply_patch_with_report` → candidate skill / candidate memory.
- **GATE** (`skillopt.evaluation.gate.evaluate_gate`): replay a *held-out* slice of tasks with the candidate; accept only if it strictly beats current. Rejected edits go to the rejected-edit buffer (negative feedback) exactly as in the paper.
- A **slow/meta** pass across nights (not just within one night) carries durable, cross-session lessons — the literal "short-term experience → long-term knowledge" of the Sleep paper. Per-night state persists in `~/.skillopt-sleep/state.json`.

**5. Stage** (`staging/`)
Write `proposed_CLAUDE.md`, `proposed_SKILL.md`, a unified diff, and a
`sleep_report.md` (what changed, why, gate deltas, token cost) into
`<project>/.skillopt-sleep/staging/<date>/`. **Nothing live is modified.**

**6. Adopt**
`/sleep adopt` (or `auto_adopt: true` in config for power users) copies staged
files over the live `CLAUDE.md` / `SKILL.md`, after a `git`-style backup. This
is the only stage that mutates user-facing config, and it is explicit by default
— the Dreams "review the output, then adopt or discard" contract.

### 5.2 Components & boundaries (each independently testable)

```
skillopt/sleep/
  __init__.py
  types.py         # SessionDigest, TaskRecord, ReplayResult, SleepConfig, SleepReport (dataclasses)
  harvest.py       # ~/.claude transcripts + history.jsonl  ->  list[SessionDigest]
  mine.py          # list[SessionDigest]  ->  list[TaskRecord]   (LLM miner + heuristic fallback)
  replay.py        # TaskRecord + skill + memory  ->  ReplayResult (hard/soft)   (mock | fresh)
  consolidate.py   # ReplayResults -> candidate skill+memory -> GATE -> accepted artifacts
  memory.py        # CLAUDE.md read/merge/dedup/diff (Dream-style) + protected-region markers
  state.py         # ~/.skillopt-sleep/state.json: last_sleep, night counter, slow/meta memory
  staging.py       # write/adopt staging dir, backups
  cli.py           # `python -m skillopt.sleep {run|status|adopt|harvest|dry-run}`
  config.py        # SleepConfig load/merge (defaults + ~/.skillopt-sleep/config.yaml)
  optimizer_backend.py  # thin: route reflect/judge to a chosen backend; mock backend for tests

skillopt-sleep-plugin/            # the Claude Code plugin surface
  .claude-plugin/plugin.json
  commands/sleep.md               # /sleep [run|status|adopt|dry-run]
  commands/sleep-status.md
  skills/skillopt-sleep/SKILL.md  # so Claude knows how to drive the engine
  hooks/hooks.json                # optional: schedule + on-session-end harvest
  scripts/*                       # shims that call `python -m skillopt.sleep ...`
```

**Reuse, don't fork:** `consolidate.py` calls into existing
`skillopt.optimizer.clip.rank_and_select`, `skillopt.gradient.aggregate.merge_patches`,
`skillopt.optimizer.skill.apply_patch_with_report`, and
`skillopt.evaluation.gate.evaluate_gate`. The sleep layer is an **EnvAdapter-shaped
shim** over the user's own life, not a new optimizer.

### 5.3 Data flow (one task, end to end)

```
history.jsonl + <session>.jsonl
   └─harvest→ SessionDigest{prompts, finals, tools, feedback}
        └─mine→ TaskRecord{intent, attempted, outcome, reference}
             └─replay(current skill+mem)→ ReplayResult{hard, soft, trajectory}
                  └─reflect→ edits(skill), edits(memory)
                       └─rank/clip(edit_budget)→ candidate
                            └─GATE(replay held-out)→ accept? → staging/  → (adopt) live CLAUDE.md/SKILL.md
```

## 6. Scheduling & triggering

- **Cron/scheduled:** documented `crontab` line + an optional Claude Code hook; default `0 3 * * *` (3am local; pick an off-:00 minute in practice). The engine is a plain CLI so it works under cron, systemd-timer, or the Claude Code scheduler.
- **User-triggered:** `/sleep run` (full cycle), `/sleep dry-run` (harvest+mine+replay, no edits), `/sleep status`, `/sleep adopt`.
- **On-session-end harvest (optional hook):** cheaply append the just-finished session to the night's buffer so the 3am run has fresh data without a full rescan.

## 7. Safety model (hard requirements)

1. **Never mutate live `CLAUDE.md`/`SKILL.md` except via explicit `adopt`** (or opt-in `auto_adopt`). Default = staged + reviewed (Dreams contract).
2. **Backups:** every adopt snapshots the prior file to `staging/<date>/backup/`.
3. **Read-only harvest:** transcripts are read, never written.
4. **`fresh` replay runs only in throwaway worktrees**, never the user's checkout; no `rm -rf`, no force-push, network off unless `replay.network: true`.
5. **Budget cap:** `max_tokens_per_night` + `max_tasks_per_night`; stop early when hit, log what was skipped (no silent truncation).
6. **Secret hygiene:** redact obvious secrets from digests before they enter prompts (reuse `_redact_*` ideas from trainer).
7. **PII/scope:** only harvest projects on an allowlist (default: the project the plugin is invoked in) or `projects: all` opt-in.

## 8. Validation experiment — "does it actually improve?"

A self-contained, **deterministic-by-default** experiment lives in
`skillopt/sleep/experiments/` and is the acceptance test for the whole idea.

**Setup:** a synthetic "user persona" (e.g. *researcher who keeps asking for
arXiv-id extraction in a fixed format*, or *programmer who keeps mis-formatting
git commit messages*). We ship 12–20 tiny tasks with **exact checkable
references**, split into `replay` (train) and `holdout` (test).

**Procedure:**
1. Score the holdout with an **empty** skill+memory → `baseline`.
2. Run `N` sleep nights (each: replay train slice → reflect → gated edit).
3. Score holdout with the evolved skill+memory → `after`.
4. Report `after − baseline`, accept/reject counts, edit count, tokens.

**Two backends:**
- `mock` (default, **no API key, fully deterministic**): a scripted optimizer that proposes the known-good rule on failure and a scripted judge. Proves the *plumbing* (harvest→mine→replay→gate→adopt) monotonically improves the score and the gate blocks regressions. This is the CI-able acceptance test.
- `anthropic` (opt-in, uses `ANTHROPIC_API_KEY`): the real optimizer/judge, to demonstrate genuine lift on the persona tasks.

**Success criteria:**
- Mock: `after > baseline`, gate rejects an injected harmful edit, adopt+backup works, re-run is reproducible. (Hard gate in CI.)
- Anthropic (when run): `after ≥ baseline` on holdout with ≥1 accepted, human-readable edit; documented in the wake-up report.

## 9. Personas (the user's framing) → concrete recurring-task families

- **Programmer:** commit-message conventions, repo-specific build/test commands, "always run X before Y", framework gotchas → consolidated into project `CLAUDE.md` + a `repo-workflow` skill.
- **Researcher:** citation/format preferences, experiment-logging habits, paper-section style, dataset-path memory → `research-prefs` skill + memory.
- **Finance/analyst:** report formatting, recurring data-pull recipes, terminology → `report-style` skill + memory.
The engine is domain-agnostic; the persona only changes which tasks get mined.

## 10. Phased delivery

- **Phase 0 — scaffold + types + harvest** (read-only, no API). Provable on this box's real `~/.claude`.
- **Phase 1 — mine + replay(mock) + consolidate + gate + staging**, with the **mock** optimizer backend and the deterministic experiment green. *(primary deliverable of the offline session)*
- **Phase 2 — plugin surface** (`/sleep`, skill, hooks, plugin.json) wired to the CLI.
- **Phase 3 — real Anthropic backend** for miner/reflect/judge + `fresh` replay in worktrees.
- **Phase 4 — slow/meta cross-night memory**, adopt automation, multi-project, polish + docs.

This session targets **Phase 0 + Phase 1 fully**, **Phase 2 scaffolded**, and the
**deterministic experiment passing**, all committed (not pushed) for review.

## 11. Open questions for the user (answer when awake)

1. **Adopt policy:** keep default *review-gated*, or do you want `auto_adopt` for your own machine?
2. **Scope:** harvest only the invoked project, or all projects in `~/.claude/projects`?
3. **Real-API demo:** want me to spend live `ANTHROPIC_API_KEY` budget on the persona demo, or keep everything mock until you say go?
4. **Skill target:** evolve a *new* dedicated `skillopt-sleep`-managed skill, or also edit your existing hand-written skills in `~/.claude/skills`?
5. **Paper:** should this become a section/figure in the SkillOpt arXiv (Dream+Sleep framing as "deployment-time continual skill optimization")?
```

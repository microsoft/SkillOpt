# AIForAI Tri-Agent SkillOpt-Sleep Optimization Design

**Status:** approved-for-design by user, 2026-06-09
**Target skill repo:** `/Users/zhangjinouwen/Github/skill/AIForAI`
**Target skill:** `ai-model-rd-protocol/SKILL.md`
**Host repo:** `/Users/zhangjinouwen/Github/microsoft/SkillOpt`

---

## 1. Summary

Build a weekly/monthly optimization loop for the active AIForAI skill
`ai-model-rd-protocol`. The loop reads real trajectories from three local agent
surfaces, mines recurring AI model R&D tasks, replays checkable tasks under the
current and candidate skill, and stages only candidate changes that pass a
held-out validation gate plus AIForAI's repository validators.

The first version must support all three agent sources:

| Source | Local evidence path | First-version role |
|---|---|---|
| Codex | `~/.codex/state_5.sqlite`, `~/.codex/sessions/**/*.jsonl` | Structured thread metadata plus rollout JSONL. |
| Claude Code | `~/.claude/projects/**/*.jsonl`, `~/.claude/history.jsonl` | Reuse existing SkillOpt-Sleep transcript parser. |
| CodeWhale / DeepSeek TUI | `~/.codewhale/tasks/runtime/*`, `~/.deepseek/sessions/*.json`, `~/.deepseek/logs/tui-*.log`, audit logs | New adapter for CodeWhale task runtime and DeepSeek session/log files. |

The optimizer must not directly overwrite the live skill. It writes a staged
proposal under the AIForAI repo, including a report, diff, mined task manifest,
per-source metrics, and validation logs. Adoption remains explicit.

## 2. Goals

1. Harvest Codex, Claude Code, and CodeWhale trajectories from local disk in a
   read-only way.
2. Normalize all sources into a common trajectory/session schema with source
   provenance.
3. Mine only checkable AI model R&D tasks; drop or quarantine tasks that cannot
   be validated.
4. Optimize `ai-model-rd-protocol/SKILL.md` using SkillOpt-Sleep's bounded edit
   and gate discipline.
5. Keep the first version safe by writing only into a protected learned block in
   `SKILL.md`, not rewriting the hand-authored doctrine or references.
6. Gate candidate changes on source-stratified held-out tasks, an AIForAI
   curated regression suite, and the existing AIForAI validators.
7. Produce a human-reviewable staging bundle for every run.

## 3. Non-Goals

- No automatic adoption in the first version.
- No rewriting `references/*.md` in the first version.
- No full benchmark materialization or large data/model downloads.
- No cluster submission as part of the optimization loop.
- No claim that the optimized skill improves all future AI R&D tasks without
  slice-level evidence from the staged report.
- No use of test split tasks for proposing edits.

## 4. Existing Context

SkillOpt already contains `skillopt_sleep`, which implements the deployment-time
loop closest to this use case:

```text
harvest session transcripts -> mine recurring tasks -> replay offline
  -> consolidate (reflect -> bounded edit -> validation gate)
  -> stage proposal -> adopt
```

AIForAI is an independent Codex skill repo. It currently provides:

- The live skill at `ai-model-rd-protocol/SKILL.md`.
- Reference guidance under `ai-model-rd-protocol/references/`.
- Repository validators:
  - `python3 scripts/quick_validate.py ai-model-rd-protocol`
  - `python3 -m unittest discover -s tests -v`

Those validators check package structure and documentation consistency. They do
not verify behavioral quality. The new loop supplies the missing behavior gate.

## 5. Architecture

### 5.1 New Package Boundary

Add an AIForAI-specific extension layer under `skillopt_sleep` rather than
changing the research `skillopt` trainer:

```text
skillopt_sleep/
  aiforai/
    __init__.py
    cli.py
    config.py
    harvesters/
      __init__.py
      base.py
      codex.py
      claude.py
      codewhale.py
    normalize.py
    mine.py
    regression_suite.py
    skill_adapter.py
    replay.py
    report.py
    run.py
```

Responsibilities:

- `harvesters/*`: source-specific local parsers. No writes and no network.
- `normalize.py`: maps source-specific records into shared session and event
  dataclasses.
- `mine.py`: converts normalized sessions into checkable AIForAI task records.
- `regression_suite.py`: fixed curated behavioral checks for AIForAI.
- `skill_adapter.py`: reads and stages the target skill, manages protected
  learned block, runs AIForAI validators.
- `replay.py`: runs text replay in version one; leaves an interface for future
  exec replay.
- `run.py`: orchestrates audit and optimization commands.
- `report.py`: writes staged reports and source coverage summaries.

### 5.2 CLI Shape

Expose a dedicated command so this flow is explicit and does not alter generic
SkillOpt-Sleep defaults:

```bash
python -m skillopt_sleep.aiforai audit \
  --target-skill-repo /Users/zhangjinouwen/Github/skill/AIForAI \
  --sources codex,claude,codewhale \
  --lookback-days 30
```

```bash
python -m skillopt_sleep.aiforai run \
  --target-skill-repo /Users/zhangjinouwen/Github/skill/AIForAI \
  --sources codex,claude,codewhale \
  --lookback-days 7 \
  --gate on \
  --max-tasks-per-source 40 \
  --backend codex
```

```bash
python -m skillopt_sleep.aiforai adopt \
  --target-skill-repo /Users/zhangjinouwen/Github/skill/AIForAI
```

`audit` is the first implementation target. It proves source parsing and task
mining before any optimization is allowed.

## 6. Data Model

### 6.1 Normalized Session

Every harvester emits `AiforaiSessionDigest`:

```python
@dataclass(slots=True)
class AiforaiSessionDigest:
    source_agent: Literal["codex", "claude", "codewhale"]
    session_id: str
    raw_path: str
    cwd: str
    git_branch: str = ""
    started_at: str = ""
    ended_at: str = ""
    user_prompts: list[str] = field(default_factory=list)
    assistant_finals: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    feedback_signals: list[str] = field(default_factory=list)
    skill_mentions: list[str] = field(default_factory=list)
    event_count: int = 0
    parse_warnings: list[str] = field(default_factory=list)
```

`skill_mentions` records evidence such as `ai-model-rd-protocol`,
`AI Model R&D Protocol`, or loaded skill paths. It is a filter feature, not a
hard requirement, because some agents may use the skill indirectly through
global instructions.

### 6.2 Checkable Task

The miner emits `AiforaiTaskRecord`:

```python
@dataclass(slots=True)
class AiforaiTaskRecord:
    id: str
    source_agent: str
    source_sessions: list[str]
    project: str
    intent: str
    context_excerpt: str
    task_family: str
    outcome: Literal["success", "fail", "mixed", "unknown"]
    split: Literal["train", "val", "test"] = "train"
    reference_kind: Literal["rule", "rubric"] = "rule"
    judge: dict = field(default_factory=dict)
    origin: Literal["real", "curated"] = "real"
```

Task families for the first version:

- `training_contract`
- `evaluation_contract`
- `data_acquisition`
- `cluster_preflight`
- `dirty_worktree_gate`
- `claim_integrity`
- `rag_agent_diagnosis`
- `experiment_record`
- `handoff_report`

Tasks without a concrete judge are not used for gate. They may appear in the
audit report under `uncheckable_candidates`.

## 7. Harvester Design

### 7.1 Codex Harvester

Read `~/.codex/state_5.sqlite`:

- Query `threads` for `rollout_path`, `cwd`, `title`, `created_at_ms`,
  `updated_at_ms`, `model`, `reasoning_effort`, `agent_role`, and branch
  fields.
- Filter by lookback and optional project allowlist.
- Parse each `rollout_path` JSONL from `~/.codex/sessions/**`.
- Extract user events, assistant final messages, tool calls, tool outputs,
  file references, and feedback-like user follow-ups.

Do not scan `~/.codex/logs_2.sqlite` in the first version. It is large and
low-level; `state_5.sqlite` plus rollout JSONL is enough for MVP.

### 7.2 Claude Harvester

Reuse and lightly adapt existing `skillopt_sleep.harvest`:

- Input directories:
  - `~/.claude/projects`
  - `~/.claude/history.jsonl`
- Preserve existing parsing for user prompts, assistant finals, tools, files,
  feedback signals, cwd, branch, and raw path.
- Add `source_agent="claude"`.

### 7.3 CodeWhale Harvester

Use a layered parser because CodeWhale/DeepSeek stores multiple artifacts:

Priority 1, structured runtime:

- `~/.codewhale/tasks/runtime/threads/*.json`
- `~/.codewhale/tasks/runtime/items/*.json`
- `~/.codewhale/tasks/runtime/turns/*.json`
- `~/.codewhale/tasks/runtime/events/*.jsonl`
- `~/.codewhale/tasks/queue.json`

Priority 2, DeepSeek sessions:

- `~/.deepseek/sessions/*.json`

Priority 3, logs for metadata and warnings:

- `~/.deepseek/logs/tui-*.log`
- `~/.codewhale/audit.log`
- `~/.deepseek/audit.log`

The first implementation should support schema discovery defensively: parse
known keys, preserve unknown keys only as counts or warnings, and never fail the
whole run because one session has unexpected shape.

## 8. Mining And Filtering

The miner uses a two-stage filter:

1. **AIForAI relevance filter**
   Keep sessions where at least one is true:
   - the skill is mentioned or loaded;
   - prompt/final/tool trace contains AI model R&D terms;
   - task family classifier maps it to one of the first-version task families.

2. **Checkability filter**
   Emit a task only when it can attach a local rule judge or a compact rubric
   judge. Preferred first-version judges are local rule checks because they make
   the validation gate cheap and reproducible.

Examples:

```json
{
  "task_family": "claim_integrity",
  "judge": {
    "kind": "rule",
    "checks": [
      {"op": "contains", "arg": "Delivered artifact"},
      {"op": "contains", "arg": "Verified evidence"},
      {"op": "contains", "arg": "Unverified boundary"},
      {"op": "contains", "arg": "Next deliverable"}
    ]
  }
}
```

```json
{
  "task_family": "training_contract",
  "judge": {
    "kind": "rule",
    "checks": [
      {"op": "contains", "arg": "training contract"},
      {"op": "contains", "arg": "evaluation contract"},
      {"op": "contains", "arg": "stop criteria"},
      {"op": "contains", "arg": "artifact paths"}
    ]
  }
}
```

## 9. Split Policy

The split must prevent source imbalance and leakage:

- Split by stable hash of `source_agent + task_family + normalized intent`.
- Stratify by `source_agent` and `task_family`.
- Use only real mined tasks for `val` and `test`.
- Curated regression tasks have `origin="curated"` and are evaluated separately;
  they do not drive reflection.
- Weekly default: `val_fraction=0.25`, `test_fraction=0.0`.
- Monthly default: `val_fraction=0.2`, `test_fraction=0.2`.
- Cap each source independently with `max_tasks_per_source`.

Gate must report:

- aggregate score;
- per-source score;
- per-task-family score;
- number of held-out examples per slice.

If a source has too few val tasks, it is reported but not used for a hard
per-source no-regression threshold.

## 10. Replay And Gate

### 10.1 Text Replay For MVP

Prompt the target backend with:

- current or candidate `SKILL.md`;
- relevant excerpts from references when needed;
- one mined task intent and context;
- instruction to answer as an AI model R&D protocol agent.

Score with local rule judge when possible. Use LLM rubric judge only when a task
has no reliable local rule checks and is explicitly marked as such.

### 10.2 Candidate Edit Scope

The first version appends or replaces only inside this protected block:

```markdown
<!-- SKILLOPT-AIFORAI:LEARNED START -->
## Learned AIForAI Rules

...
<!-- SKILLOPT-AIFORAI:LEARNED END -->
```

Rules outside the block remain hand-authored. If a learned rule is valuable
after several accepted runs, a human can promote it into the main skill or a
reference file in a separate change.

### 10.3 Acceptance Criteria

A candidate is accepted for staging only if all are true:

1. Aggregate val score strictly improves.
2. Curated regression suite passes.
3. Existing AIForAI quick validator passes.
4. Existing AIForAI unit tests pass.
5. No eligible source slice regresses beyond the configured tolerance.
6. Candidate diff touches only allowed files and protected regions.

The first-version source regression tolerance should be strict when a source has
at least five val tasks: no decrease in hard score. When fewer than five tasks
exist, report the slice but do not block solely on that slice.

## 11. Curated Regression Suite

Ship a fixed suite of AIForAI behavior checks. These prevent the optimizer from
overfitting to recent user sessions.

Initial cases:

1. **Training contract gate**
   A user asks to start or resume training. The answer must require a written
   training contract and evaluation contract with code version, dataset/split,
   metrics, artifact paths, stop criteria, and failure criteria.

2. **Data acquisition hygiene**
   A user asks to download a full or size-unknown dataset locally. The answer
   must classify scope, avoid local full download by default, and route to the
   approved data/platform/shared-storage path.

3. **Dirty worktree boundary**
   A user asks to run a formal experiment in a dirty repo. The answer must
   inspect/report workspace state and block or downgrade the run unless the user
   explicitly authorizes a non-formal diagnostic.

4. **Claim integrity**
   A user asks whether a system is done. The answer must separate delivered
   artifact, verified evidence, unverified boundary, and next deliverable.

5. **RAG/agent diagnosis**
   A user reports a RAG or agent failure. The answer must inspect retrieval or
   tool trajectories before blaming generation or reasoning.

6. **Cluster preflight**
   A user asks to submit a cluster job. The answer must require image,
   dependencies, data access, artifact writing, logging, and a minimal smoke
   validation before the expensive job.

7. **Test-set hygiene**
   A user asks to tune against final test results. The answer must reject tuning
   on test and suggest train/val controlled experiments.

8. **Human-reviewable deliverable**
   A user asks for a completed deliverable. The answer must require a clean
   commit SHA or explicitly label the work as exploratory/diagnostic.

## 12. Staging Output

Every non-dry `run` writes:

```text
<AIForAI>/.skillopt-sleep/staging/YYYYMMDD-HHMMSS/
  manifest.json
  report.md
  report.json
  proposed_SKILL.md
  diff.patch
  task_manifest.jsonl
  uncheckable_candidates.jsonl
  source_coverage.json
  baseline_results.jsonl
  candidate_results.jsonl
  curated_regression_results.jsonl
  validation.log
  backup/
```

`report.md` must include:

- source coverage by Codex, Claude, and CodeWhale;
- accepted/rejected edits;
- gate metric movement;
- per-source and per-family score movement;
- validator command output summary;
- explicit unverified boundaries;
- adopt instructions.

`adopt` copies only staged `proposed_SKILL.md` over the live AIForAI skill after
backing up the previous file into the staging `backup/` directory.

## 13. Error Handling

- Missing source directory: warn and continue; fail only if all requested
  sources are missing.
- Malformed session file: record parse warning and continue.
- No checkable tasks: write an audit report and do not run optimization.
- Validator failure: stage candidate as rejected with logs, do not mark accepted.
- Gate rejection: write report with rejected edits and negative evidence, do not
  propose adoption.
- Dirty AIForAI repo before run: allow `audit`; block `run` and `adopt` unless
  the user explicitly passes an override flag.
- Secrets: redact obvious API keys, tokens, and auth blobs before any mined
  context enters prompts or reports.

## 14. Testing Strategy

### Unit Tests

- Codex harvester parses a synthetic `state_5.sqlite` plus rollout JSONL.
- Claude harvester preserves existing behavior on synthetic transcript JSONL.
- CodeWhale harvester parses synthetic runtime thread/item/turn/event files and
  DeepSeek session JSON.
- Normalizer deduplicates sessions and preserves `source_agent`.
- Miner emits checkable task records for each task family and drops
  uncheckable prompts.
- Splitter preserves source/task-family stratification and keeps curated tasks
  out of train-driven reflection.
- Skill adapter only edits the protected learned block.
- Staging writes the full expected manifest and report files.
- Gate blocks a candidate that improves one source but regresses another
  eligible source.

### Integration Tests

- `audit` over fixture Codex, Claude, and CodeWhale directories produces a
  source coverage report with all three sources.
- `run` with mock backend produces an accepted candidate when a fixture task
  requires a missing learned rule.
- `run` rejects a harmful edit and still writes a report.
- `adopt` backs up the old skill and updates only the live skill path from the
  staging manifest.
- AIForAI validators are invoked and their failure blocks acceptance.

### Manual Smoke

Run against the real local machine in dry audit mode:

```bash
python -m skillopt_sleep.aiforai audit \
  --target-skill-repo /Users/zhangjinouwen/Github/skill/AIForAI \
  --sources codex,claude,codewhale \
  --lookback-days 30
```

Expected result: report shows nonzero discovered sessions for at least one
source, parse warnings are bounded, and no files are modified.

## 15. Phased Delivery

### Phase 1: Read-only audit

Deliver the tri-agent harvesters, normalizer, miner, and audit report. This
phase makes no edits and spends no optimizer budget unless LLM mining is
explicitly enabled.

Acceptance:

- Codex, Claude, and CodeWhale fixture tests pass.
- Real-machine audit command writes a report without modifying AIForAI.
- Report distinguishes checkable tasks from uncheckable candidates.

### Phase 2: Mock gated run

Add protected-block skill adapter, curated regression suite, mock replay, gate,
and staging.

Acceptance:

- Fixture run accepts a helpful learned rule.
- Fixture run rejects a harmful rule.
- AIForAI validators are run and captured in `validation.log`.
- No live skill mutation without `adopt`.

### Phase 3: Real backend weekly run

Enable `--backend codex` or another configured backend for replay/reflect.
Keep `auto_adopt=false`.

Acceptance:

- Weekly staged report includes real source coverage and gate movement.
- Candidate diff is limited to protected learned block.
- User can review and adopt.

### Phase 4: Monthly deeper evaluation

Add monthly test split reporting, more task volume per source, and optional exec
replay for higher fidelity.

Acceptance:

- Monthly report includes held-out test metrics not used for optimization.
- Exec replay remains isolated in temporary workspaces.

## 16. Open Decisions

1. Whether CodeWhale should be treated as `source_agent="codewhale"` even when
   data is stored under `~/.deepseek`, or whether reports should display
   `codewhale/deepseek`. Recommendation: normalize to `codewhale` and include
   raw path provenance.
2. Whether weekly runs should use LLM mining by default. Recommendation: no for
   Phase 1 audit, yes for Phase 3 real backend runs with a strict task cap.
3. Whether adopted learned rules should ever be auto-promoted from the protected
   block into references. Recommendation: not in this feature; use separate
   human-reviewed maintenance changes.
4. Which backend should be the default real optimizer. Recommendation: keep
   configurable; start with Codex because the local SkillOpt-Sleep Codex path
   already exists.

## 17. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| CodeWhale schema drifts or has sparse files. | Defensive parser, parse warnings, fixture tests for known schemas, audit before run. |
| Recent sessions overfit one agent's behavior. | Source-stratified caps and per-source gate slices. |
| Mined tasks are not objectively checkable. | Drop from gate; report as uncheckable candidates. |
| Learned block grows noisy. | Edit budget, dedup, rejected-edit buffer, and monthly human review. |
| Candidate weakens core doctrine while passing recent tasks. | Curated regression suite plus no direct rewrite outside protected block. |
| Sensitive data leaks into prompts or reports. | Redaction before mining/replay; raw paths recorded, raw content minimized. |
| User assumes accepted means globally better. | Report must state verified evidence, unverified boundaries, and source/test coverage. |

## 18. Approval Gate

This design is ready to become an implementation plan after user review. The
first implementation plan should cover only Phase 1 and Phase 2, because those
phases produce a safe, testable MVP without relying on expensive real backend
runs.

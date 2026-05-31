# Gitmoot-SkillOpt MVP

Implement the plan task by task. Each task must be developed, reviewed, opened
as its own pull request, merged, and verified before moving on, unless tasks are
explicitly safe to run in parallel.

Build the first Gitmoot-specific SkillOpt optimizer path. The intended outcome
is a local-first optimizer repo that consumes Gitmoot training packages, runs
the existing SkillOpt training loop through a Gitmoot adapter, emits a
Gitmoot-compatible pending candidate package, and hands optimizer-produced
artifacts back to Gitmoot through a verified artifact manifest. Gitmoot remains
the registry, artifact-store owner, feedback collector, and human
promotion/rejection layer.

This goal touches two repositories:

- Primary repo: `/root/gitmoot-skillopt`
- Supporting contract repo: `/root/gitmoot`

Do not implement live pairwise evaluation in this goal. That remains tracked in
Gitmoot issue #77. Do not make `gitmoot-skillopt` write directly into
`~/.gitmoot`, Gitmoot SQLite, or Gitmoot artifact blobs.

External context checked before writing this goal:

- Gitmoot issue #67 defines the approved architecture: Gitmoot exports training
  packages, `gitmoot-skillopt` optimizes, Gitmoot imports pending candidates,
  and humans promote or reject.
- Gitmoot issue #77 tracks future live pairwise evaluation and is out of scope
  for this MVP.
- Artifact-system patterns favor immutable output packages, content hashes,
  provenance metadata, and import-time verification by the owning registry.
- Prompt-optimization systems need examples plus metrics/evaluators; prompt
  text alone is not enough for reproducible optimization.

## Core Rules

- Work one task at a time in the listed order by default.
- If tasks are independent, have disjoint file ownership, and do not depend on
  each other's results, they may be done in parallel on separate branches.
- Do not start dependent work until the prerequisite task has passed checks,
  passed `codex exec review --uncommitted`, been pushed, opened as a PR, merged,
  and verified on the target branch.
- Do not commit generated data, reports, caches, build artifacts, secrets,
  credentials, session archives, cloned helper repos, local plugin build output,
  optimizer output directories, or large outputs unless the plan explicitly says
  they are intended tracked fixtures/artifacts.
- Preserve existing behavior unless the current task explicitly changes it.
- Keep changes clean, scoped, and organized. Avoid broad rewrites.
- Avoid code duplication. When repeated logic appears, extract small reusable
  helpers that match existing repo patterns.
- If implementation depends on external APIs, docs, CLIs, data formats,
  generated scripts, installers, service launchers, subprocess calls, env vars,
  config formats, or third-party libraries, verify the real contract with local
  commands and/or official sources before editing.
- For multi-repo tasks, identify every changed repository and run the relevant
  checks and `codex exec review --uncommitted` in each changed repo.

## Before Starting

1. Inspect current repo state in both repos:
   - `cd /root/gitmoot-skillopt && git status --short && git branch --show-current && git remote -v`
   - `cd /root/gitmoot && git status --short && git branch --show-current && git remote -v`
2. If either target branch is unclear, a remote looks wrong, or a worktree has
   unrelated existing changes that make task commits ambiguous, stop and ask
   before continuing.
3. Confirm the target base branch from each repo. If unspecified, use the
   current branch as the base.
4. Inspect relevant existing patterns before editing.
5. Verify PR tooling is available before the first PR:
   - `gh auth status`
   - each repo remote resolves to the expected GitHub repository.

## Per-Task Branch Workflow

1. Confirm the current task's scope.
2. Create a task branch from the latest target base branch in the repo or repos
   touched by the task.
3. Implement only that task.
4. Add or update focused tests/checks appropriate to the task.
5. Run focused tests for touched modules.
6. Run broader checks when the task touches shared behavior, CLI/API surfaces,
   data/model/evaluation logic, generated scripts, installers, service
   launchers, docs build systems, or user-facing workflows.
7. For wrapper, installer, CLI, subprocess, generated-script, env propagation,
   service-launcher, or deployment changes, include an operational smoke test or
   direct contract check. Syntax checks alone are not enough.
8. Identify every repository where files changed. In each changed repo, run:
   `codex exec review --uncommitted`
9. Preserve the exact raw review output per repo.

## Review-Fix Loop

1. If review finds issues, do not only patch the literal line.
2. Identify the underlying invariant/class of bug.
3. Audit nearby and sibling paths for the same issue.
4. Write a concise fix plan using:

   ```text
   Review found these issues: <<PASTE RAW REVIEW RESULTS BY REPO>>.
   For each issue, identify the underlying invariant/class of bug, audit sibling
   paths for the same issue, and plan the smallest safe fix. Verify external
   assumptions with local commands and/or official sources. Preserve repo
   patterns, avoid unnecessary refactors, and list tests/checks per repo.
   ```

5. Execute the fix plan.
6. Re-run focused tests/checks and `codex exec review --uncommitted` in every
   repo with uncommitted changes.
7. Repeat until the final raw review output contains no findings, or stop if
   blocked or if a finding is incorrect after verification.

## Commit Gate

1. Before committing, run `git diff --check` and inspect the final diff.
2. Commit only the current task's intended tracked changes.
3. Use the commit message specified by the plan. If the plan does not specify
   one, use a concise conventional message that describes only the current task.
4. Push the task branch.
5. Verify the task branch worktree is clean after push, except for intentionally
   ignored generated files.

## Pull Request Gate

1. Create one PR for the current task in every changed repo. If a task changes
   both repos, create linked PRs and merge them in dependency order.
2. The PR title must describe only the current task.
3. The PR body must include:
   - WHAT: what was changed
   - WHY: why the task was needed
   - CHANGES: concrete implementation changes
   - RESULTS: tests/checks/review results
   - RISK: skipped checks, blockers, or residual risk
4. Include the exact raw final `codex exec review --uncommitted` output for each
   changed repo in the PR body.
5. If CI or required checks exist, wait for them and fix failures before merge.
6. Merge the PR using the repository's configured/preferred merge method. If no
   preference is discoverable, use squash merge for a clean task-level history.
7. After merge, update the local target base branch and verify the worktree is
   clean.
8. Record the PR number, PR URL, branch name, and merged commit hash.
9. Delete the task branch after merge only if the repository normally does so or
   the merge command supports safe branch deletion.

## Parallel Task Rules

- Parallelize only when tasks are independent, have disjoint file ownership, and
  can be reviewed and merged without order-dependent assumptions.
- Use a separate branch per task.
- Clearly assign each branch a task number and file ownership.
- Do not duplicate work across branches.
- If parallel branches conflict after one PR merges, rebase or update the
  remaining branch on the latest target base and re-run its checks/review.
- If a task becomes dependent on another task, stop treating it as parallel and
  merge the dependency first.

## Final Response After All Tasks

- List completed tasks.
- For each task, list branch, PR URL, merge status, and merged commit hash.
- List tests/checks run.
- Include exact final raw `codex exec review --uncommitted` output for the last
  task/repo.
- Mention skipped checks, blockers, or residual risk.
- Do not claim interactive `/review` is clean. Say:
  `codex exec review is clean; ready for manual /review.`

## Implementation Tasks

### Task 1: Rebrand The Fork And Add The Gitmoot CLI Entry Point

Scope:

- In `/root/gitmoot-skillopt`, update project identity from upstream SkillOpt to
  Gitmoot-SkillOpt while preserving upstream attribution and license.
- Add a new console script entry point, preferably:
  `gitmoot-skillopt = gitmoot_skillopt.cli:main`.
- Keep existing `skillopt-train` and `skillopt-eval` commands working unless
  there is a clear conflict.
- Add a minimal `gitmoot_skillopt` package for Gitmoot-specific CLI and glue
  code, leaving upstream `skillopt` modules intact where possible.
- Add top-level README notes explaining that this fork is the Gitmoot optimizer
  layer and does not own Gitmoot promotion state.

Acceptance criteria:

- `python -m gitmoot_skillopt --help` works.
- `gitmoot-skillopt --help` works after editable install.
- Existing `python scripts/train.py --help` still works.
- README clearly explains the boundary:
  `gitmoot-skillopt` proposes candidates; Gitmoot imports, reviews, and
  promotes/rejects.

Tests/checks:

- `python -m compileall gitmoot_skillopt skillopt scripts`
- `python -m gitmoot_skillopt --help`
- `python scripts/train.py --help`
- If dev deps are installed: `ruff check .`

Suggested commit message:

- `feat: add gitmoot-skillopt cli scaffold`

### Task 2: Add Gitmoot Package Models And Artifact Resolution

Scope:

- Add Gitmoot contract models in `/root/gitmoot-skillopt`, matching Gitmoot's
  v1 JSON contract:
  - training package
  - template snapshot
  - eval run
  - eval items
  - artifact refs
  - feedback events
  - candidate package
  - candidate artifact manifest entries
- Add strict validation for:
  - expected `kind`
  - `contract_version == 1`
  - template id and candidate metadata consistency
  - non-empty template content
  - artifact ids, paths, hashes, media types, and drivers
- Add a Gitmoot artifact resolver that can read Gitmoot SHA256 blobs from
  `--artifact-root`, using the Gitmoot layout:
  `sha256/<first-2-hex>/<full-hex-hash>`.
- Add safe local output artifact helpers that write optimizer-produced artifacts
  under `--out-root/artifacts`, compute SHA256, and produce manifest entries.
- Do not write to Gitmoot SQLite or `~/.gitmoot`.

Acceptance criteria:

- Valid fixture training packages load and round-trip.
- Invalid package kind/version/hash/path data fails with actionable errors.
- Artifact resolution refuses missing blobs and invalid hashes.
- Local optimizer artifacts cannot escape the configured artifact output dir.

Tests/checks:

- Add pytest tests for package load/validation, Gitmoot blob resolution, and
  optimizer artifact manifest generation.
- `python -m pytest tests/test_gitmoot_package.py tests/test_gitmoot_artifacts.py`
- `python -m compileall gitmoot_skillopt skillopt scripts`

Suggested commit message:

- `feat: add gitmoot package contract models`

### Task 3: Add Gitmoot DataLoader And EnvAdapter

Scope:

- Add `skillopt/envs/gitmoot/` with:
  - `package.py` or imports from `gitmoot_skillopt` contract code
  - `dataloader.py`
  - `adapter.py`
  - `rollout.py`
  - `evaluator.py`
- Register the `gitmoot` adapter in the training CLI registry without breaking
  existing built-in adapters.
- Convert Gitmoot training package items into SkillOpt train/val/test batches.
- Use package item metadata and artifact refs to build task prompts.
- Start with text/Markdown artifacts. Other drivers must fail with a clear
  "driver not supported yet" error.
- Preserve YAML frontmatter from the template snapshot. SkillOpt should optimize
  the Markdown body while candidate output remains a full agent-template
  Markdown document.
- Implement the MVP evaluator:
  - use explicit evaluator config when supported
  - otherwise use an LLM judge comparing source/baseline/candidate output
  - mark judge-derived scores clearly in result metadata
- Include existing feedback events as optimizer context, not as automatic
  promotion.

Acceptance criteria:

- A fixture Gitmoot training package can be loaded into a Gitmoot adapter.
- The adapter can build train and eval batches from package items.
- Unsupported artifact drivers fail clearly.
- Rollout returns SkillOpt-compatible result dicts with `id`, `hard`, `soft`,
  `response`, `fail_reason`, and useful metadata.
- The adapter does not mutate Gitmoot local state.

Tests/checks:

- Unit tests for `GitmootDataLoader` split/batch behavior.
- Unit tests for text artifact prompt construction and unsupported drivers.
- Unit tests for evaluator behavior with deterministic fixture evaluator config.
- `python -m pytest tests/test_gitmoot_dataloader.py tests/test_gitmoot_adapter.py`
- `python -m compileall gitmoot_skillopt skillopt scripts`

Suggested commit message:

- `feat: add gitmoot skillopt adapter`

### Task 4: Implement `gitmoot-skillopt optimize`

Scope:

- Implement:

  ```sh
  gitmoot-skillopt optimize \
    --training-package training.json \
    --artifact-root ~/.gitmoot/evals/blobs \
    --out-root outputs/<run-id> \
    --candidate-output outputs/<run-id>/candidate.json
  ```

- Build a resolved SkillOpt config from the Gitmoot package plus CLI options.
- Use the existing `ReflACTTrainer` instead of duplicating the training loop.
- Persist the template snapshot content as the initial skill for the run.
- Route the Gitmoot package path, artifact root, and output artifact dir through
  config/env fields consumed by the Gitmoot adapter.
- After training, read `best_skill.md` and emit a Gitmoot candidate package.
- Write optimizer artifacts under `--out-root/artifacts`, including at least:
  - `candidate.diff.md`
  - `eval-report.json`
  - optionally `preference-summary.md`
- Candidate package `summary.diff_artifact_id` must reference the generated
  artifact manifest id.
- Candidate package must include `eval_report` and a concise
  `preference_summary`.

Acceptance criteria:

- `gitmoot-skillopt optimize --help` documents all required flags.
- A dry-run or tiny fixture mode can produce a structurally valid
  `candidate.json` without network calls.
- Normal optimize mode runs the existing trainer with the Gitmoot adapter.
- Output artifacts include declared hashes and can be verified independently.
- Candidate content is full agent-template Markdown with valid frontmatter.

Tests/checks:

- CLI tests for missing required flags, invalid package path, and invalid
  artifact root.
- Fixture end-to-end test that produces `candidate.json` and `artifacts/`
  without external model calls.
- `python -m pytest tests/test_gitmoot_optimize_cli.py`
- `python -m compileall gitmoot_skillopt skillopt scripts`

Suggested commit message:

- `feat: add gitmoot optimize command`

### Task 5: Add Gitmoot Candidate Artifact Manifest Import

Scope:

- In `/root/gitmoot`, extend `gitmoot skillopt import` with:

  ```sh
  gitmoot skillopt import --file candidate.json --artifact-dir ./artifacts
  ```

- Extend the Gitmoot candidate package contract with optional artifact manifest
  entries:
  - `id`
  - `path`
  - `hash`
  - `media_type`
  - `driver`
  - optional `size_bytes`
- Import logic must:
  1. Load `candidate.json`.
  2. If artifact entries exist, require `--artifact-dir`.
  3. Resolve artifact paths only inside `--artifact-dir`.
  4. Reject absolute paths and path traversal.
  5. Verify SHA256 hashes.
  6. Store verified blobs in Gitmoot content-addressed storage.
  7. Register artifact metadata in SQLite.
  8. Validate `summary.diff_artifact_id`.
  9. Import the candidate as a pending template version.
- Keep backward compatibility for candidate packages without artifact entries.

Acceptance criteria:

- `gitmoot skillopt import --file candidate.json --artifact-dir artifacts`
  imports valid candidate artifacts and pending candidate metadata.
- Invalid hashes, missing files, absolute paths, and path traversal fail before
  mutating candidate state.
- Packages without artifact entries still import as before.
- `candidate show` can display the imported diff artifact id.

Tests/checks:

- Go unit tests for artifact manifest validation and path safety.
- CLI e2e test for valid artifact import.
- CLI tests for invalid hash, missing artifact dir, and path traversal.
- `/root/temp/ts/tool/go test ./internal/skillopt ./internal/cli ./internal/artifact ./internal/db`
- `/root/temp/ts/tool/go test ./...`

Suggested commit message:

- `feat: import skillopt candidate artifacts`

### Task 6: Add Cross-Repo Smoke Documentation And Examples

Scope:

- Add docs in `/root/gitmoot-skillopt` showing the full flow:
  1. export from Gitmoot
  2. optimize with `gitmoot-skillopt`
  3. import candidate with `--artifact-dir`
  4. inspect candidate
  5. collect Markdown or GitHub feedback
  6. promote or reject
- Add a tiny fixture package and artifact files only if they are intentionally
  tracked test fixtures and small enough for the repo.
- Add a smoke script or documented smoke command that exercises fixture output
  without external model calls.
- Add a short note in `/root/gitmoot` docs linking to the external
  `gitmoot-skillopt` optimizer flow if appropriate.

Acceptance criteria:

- A new user can follow the docs without needing to know internal architecture.
- Docs clearly state that `gitmoot-skillopt` does not auto-promote candidates.
- Docs explain that live pairwise evaluation is out of scope for MVP and tracked
  separately.
- Fixture smoke validates candidate package shape and artifact hashes.

Tests/checks:

- `python -m pytest`
- `python -m compileall gitmoot_skillopt skillopt scripts`
- In `/root/gitmoot`, run docs/static checks if docs are changed.
- In every changed repo, run `git diff --check`.

Suggested commit message:

- `docs: add gitmoot skillopt mvp workflow`

### Task 7: End-To-End Local Contract Verification

Scope:

- Run a full local contract smoke using both repos and fixture data:
  1. create or use a small Gitmoot fixture eval run
  2. export `training.json`
  3. run `gitmoot-skillopt optimize` in fixture/dry-run mode
  4. import `candidate.json` with `--artifact-dir`
  5. show the pending candidate
  6. reject it with a test reason in a temp Gitmoot home
- Do not mutate the user's real `~/.gitmoot` during smoke. Use a temp
  `--home`/test home where supported.
- If a gap in Gitmoot's fixture setup makes this impossible, add the smallest
  deterministic test helper needed rather than using production state.

Acceptance criteria:

- The smoke proves the package boundary end to end without external model calls.
- The smoke proves artifact hashes are verified by Gitmoot import.
- The smoke proves the candidate remains pending until explicit promote/reject.
- The smoke output is documented in the final PR body or final task summary.

Tests/checks:

- Full smoke command sequence in temp directories.
- `python -m pytest`
- `python -m compileall gitmoot_skillopt skillopt scripts`
- `/root/temp/ts/tool/go test ./...` in `/root/gitmoot` if changed.
- `codex exec review --uncommitted` in every changed repo.

Suggested commit message:

- `test: add gitmoot skillopt contract smoke`

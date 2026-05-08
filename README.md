# ReflACT: Reflective Agent Tuning

ReflACT is a framework for optimizing an external skill document through iterative rollout, reflection, editing, and gated validation.

It does **not** fine-tune model weights. Instead, it treats the skill document as the optimization target:

- the **student** model executes tasks with the current skill
- the **teacher** model analyzes trajectories and proposes edits
- the framework merges, ranks, applies, and validates those edits
- only validated skill updates are kept

This branch implements a full training loop with step-level skill optimization and optional epoch-level memory mechanisms (`slow_update`, `meta_skill`, `meta_reflect`).

## Method Overview

### Optimization Target

Each run maintains a mutable markdown skill document. The framework repeatedly improves that document instead of changing model parameters.

This gives a training-style loop for prompt / policy optimization:

1. Roll out the current skill on a batch of tasks.
2. Reflect on failures and successes.
3. Merge patch proposals into a coherent candidate update.
4. Rank and select a bounded number of edits.
5. Apply those edits to produce a candidate skill.
6. Validate the candidate skill on a held-out selection split.
7. Keep the update only if the gate accepts it.

### Per-Step Pipeline

Every training step executes the following pipeline in `skillopt/engine/trainer.py`:

1. **Rollout**
   The student model runs a batch of tasks using the current skill.

2. **Reflect**
   The teacher analyzes minibatches of trajectories and emits raw patches.
   Failure-driven and success-driven patches are tracked separately.

3. **Aggregate**
   Raw patches are merged hierarchically. Metadata such as `support_count` and `source_type` is carried into the merged patch so later ranking can use it.

4. **Select**
   The teacher ranks the merged edit pool and keeps up to `edit_budget` edits.

5. **Update**
   The selected edits are applied to the skill document. The framework records an `edit_apply_report.json` so you can see which edits actually landed, which were skipped, and why.

6. **Evaluate / Gate**
   The candidate skill is evaluated on the selection split. Gate validation is mandatory in this branch. A candidate update is accepted only if it improves over the current selection score; a new global best is tracked separately.

### Within-Epoch Memory

Inside an epoch, the trainer maintains a step buffer containing:

- compact failure-pattern summaries from previous steps
- rejected edits and their score deltas

That context is fed back into later reflection calls so the teacher can avoid repeating ineffective edits and can focus on unsolved error patterns.

### Epoch-Level Mechanisms

This branch supports three optional epoch-level mechanisms.

#### Slow Update

At the end of each epoch, `slow_update` compares the previous epoch’s terminal skill and current epoch’s terminal skill on a sampled train subset. It then writes longitudinal guidance into a protected slow-update region inside the skill document.

Importantly, this guidance is **not** blindly written through. It is converted into a candidate skill and sent through the same selection gate as step-level updates.

#### Meta Skill

`meta_skill` is teacher-side cross-epoch memory. It does not directly edit the current skill. Instead, it writes a compact memory artifact describing longer-term patterns across adjacent epochs. That memory is loaded into later reflection / merge / ranking calls as extra context.

#### Meta Reflect

`meta_reflect` runs at epoch end over the step history of the current epoch. It looks at accepted and rejected directions from the whole epoch, proposes higher-level patch edits, applies them to a meta candidate, and then sends that candidate through the same selection gate.

## What This Branch Guarantees

The current implementation assumes the following as the mainline method contract:

- gate validation is always on
- the current skill, current score, best skill, and best score stay aligned
- `slow_update` is gated before being committed
- patch provenance (`source_type`, `support_count`) reaches selection
- patch application is observable through per-edit reports
- resume state is restored from `runtime_state.json` rather than inferred only from history
- all benchmark model calls go through the unified backend router

## Model Backends

All model access now goes through the split teacher/student model layer in `skillopt.model`.

Supported teacher backends:

- `openai_chat`
- `claude_chat`

Supported student backends:

- `openai_chat`
- `claude_chat`
- `codex_exec`
- `claude_code_exec`

Recommended config shape:

```yaml
model:
  teacher_backend: openai_chat
  student_backend: codex_exec
  teacher: gpt-5.4
  student: gpt-5.4-codex
  reasoning_effort: medium
```

Legacy `model.backend` and CLI flags like `--backend codex` still work. They are mapped onto the split backend model for backward compatibility.

The same routing is used by:

- training (`scripts/train.py`)
- eval-only runs (`scripts/eval_only.py`)
- SpreadsheetBench standalone prompt eval scripts
- LiveMathematicianBench baseline eval script
- benchmark rollout code inside the main framework

### Azure OpenAI

If you use `openai_chat`, configure either environment variables or config values:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
```

The config supports both the old keys and the new explicit names:

```yaml
model:
  azure_openai_endpoint: "..."
  azure_openai_api_version: "..."
  azure_openai_api_key: ""
  azure_openai_auth_mode: api_key
  azure_openai_ad_scope: "https://cognitiveservices.azure.com/.default"
  azure_openai_managed_identity_client_id: ""
```

`azure_openai_auth_mode` can be used for API-key auth or Azure AD / managed identity flows.

### Exec Harness

`codex_exec` and `claude_code_exec` run the student inside a workspace harness instead of a plain chat call. The harness writes task files, renders a dynamic `SKILL.md`, runs the student CLI, and saves raw execution artifacts such as:

- `codex_raw.txt`
- `codex_trace_summary.txt`
- workspace-local task / skill files

This branch keeps `meta_skill` and `apply_patch_with_report`, while upgrading the student path to the more realistic workspace-exec setup.

### Trace-Aware Deep Reflect

When `student_backend=codex_exec` and `gradient.use_deep_reflect=true`, deep reflection can probe a specific earlier Codex attempt:

- the teacher sees a compact Codex trace summary
- deep probe can target `probe_target_id`
- the follow-up rollout can resume from `probe_after_step`

This is wired for the dataset-backed environments in this branch.

### Rewrite Mode

Skill updates support two modes:

- `optimizer.skill_update_mode=patch`
- `optimizer.skill_update_mode=rewrite_from_suggestions`

`patch` keeps the existing fine-grained edit application path and still records `edit_apply_report.json`.

`rewrite_from_suggestions` asks the teacher to emit higher-level rewrite suggestions, then rewrites the whole skill in one pass. This is useful when patch edits become too fragmented.

## Repository Layout

```text
skillopt/
  engine/
    trainer.py                 main training loop
  gradient/
    reflect.py                 minibatch reflection
    aggregate.py               hierarchical patch merge
    deep_probe.py              diagnostic probing for deep reflect
  optimizer/
    clip.py                    edit ranking / selection
    skill.py                   patch application + apply report
    slow_update.py             epoch-level longitudinal guidance
    meta_skill.py              teacher-side cross-epoch memory
    meta_reflect.py            epoch-level macro editing
  evaluation/
    gate.py                    pure gate decision logic
  model/
    backend_config.py          teacher/student backend routing
    azure_openai.py            Azure backend
    codex_harness.py           workspace exec harness + Codex trace parsing
    claude_backend.py          Claude backend
  envs/
    ...                        environment adapters and rollout logic
scripts/
  train.py                     unified training entry
  eval_only.py                 evaluate one skill without training
configs/
  _base_/default.yaml          shared defaults
  <env>/default.yaml           environment-specific configs
```

## Configuration

Configs use structured YAML with `_base_` inheritance.

The base config is `configs/_base_/default.yaml`. Key defaults in this branch are:

- `model.teacher_backend = openai_chat`
- `model.student_backend = openai_chat`
- `model.reasoning_effort = medium`
- `optimizer.use_slow_update = true`
- `optimizer.use_meta_skill = true`
- `optimizer.use_meta_reflect = false`
- `gradient.use_deep_reflect = false`
- `optimizer.skill_update_mode = patch`

Default setting snapshot:

```yaml
model:
  backend: azure_openai
  teacher: gpt-5.4
  student: gpt-5.4
  teacher_backend: openai_chat
  student_backend: openai_chat
  reasoning_effort: medium
  rewrite_reasoning_effort: ""
  rewrite_max_completion_tokens: 64000
  codex_exec_path: codex
  codex_exec_sandbox: workspace-write
  codex_exec_profile: ""
  codex_exec_full_auto: false
  codex_exec_reasoning_effort: none
  claude_code_exec_path: claude
  claude_code_exec_profile: ""
  codex_trace_to_teacher: true

train:
  num_epochs: 4
  train_size: 0
  batch_size: 80
  accumulation: 1
  seed: 42

gradient:
  minibatch_size: 16
  merge_batch_size: 16
  analyst_workers: 16
  max_analyst_rounds: 3
  failure_only: false
  use_deep_reflect: false
  deep_reflect_failures: 4
  deep_reflect_successes: 2

optimizer:
  learning_rate: 8
  min_learning_rate: 2
  lr_scheduler: cosine
  skill_update_mode: patch
  use_meta_reflect: false
  meta_learning_rate: 8
  use_slow_update: true
  slow_update_samples: 20
  use_meta_skill: true

evaluation:
  use_gate: true
  sel_env_num: 0
  test_env_num: 0
  eval_test: true

env:
  split_mode: ratio
  split_ratio: "2:1:7"
  split_seed: 42
```

For the full source of truth, see [configs/_base_/default.yaml](/home/azureuser/workspace-yqh/skillopt_final/configs/_base_/default.yaml).

Selected fields:

| Section | Key | Meaning |
|---|---|---|
| `model` | `teacher_backend` | teacher backend: `openai_chat` or `claude_chat` |
| `model` | `student_backend` | student backend: chat backend or exec backend |
| `model` | `teacher` | teacher model / deployment |
| `model` | `student` | student model / deployment |
| `model` | `reasoning_effort` | reasoning budget passed to the backend when supported |
| `model` | `codex_trace_to_teacher` | include Codex trace summaries in teacher reflection context |
| `train` | `num_epochs` | number of epochs |
| `train` | `train_size` | expected train split size, or `0` to infer |
| `train` | `batch_size` | tasks per rollout batch |
| `train` | `accumulation` | number of rollout/reflect minibatches per step |
| `gradient` | `minibatch_size` | trajectories per analyst minibatch |
| `gradient` | `merge_batch_size` | patches per aggregate batch |
| `gradient` | `use_deep_reflect` | enable diagnostic probe rollouts |
| `gradient` | `max_analyst_rounds` | teacher reflection retries / refinement budget |
| `optimizer` | `learning_rate` | max edits kept after selection |
| `optimizer` | `lr_scheduler` | edit-budget scheduler |
| `optimizer` | `use_slow_update` | epoch-level longitudinal guidance |
| `optimizer` | `use_meta_skill` | teacher-side epoch memory |
| `optimizer` | `use_meta_reflect` | epoch-level macro editing |
| `optimizer` | `skill_update_mode` | `patch` or `rewrite_from_suggestions` |
| `evaluation` | `sel_env_num` | selection set size (`0` means full split) |
| `evaluation` | `test_env_num` | test set size (`0` means full split) |

### Important Branch Rule

`use_gate=false` is intentionally not supported in this branch. Gate validation is part of the method contract here.

If an old config still contains `evaluation.use_gate: false`, the loader / trainer will raise instead of silently continuing.

## Supported Environments

The main training entry and eval-only entry now register 11 environments:

| Env | Default rollout shape | Current default split / data setting | Branch alignment |
|---|---|---|---|
| `alfworld` | environment-backed episodic rollout | native ALFWorld train/eval splits | in `skillopt_new_zzw` |
| `babyvision` | single-round multimodal QA | `split_mode=ratio` from raw metadata/images, or prepared `split_dir` | in `skillopt_new_zzw` |
| `docvqa` | single-round multimodal QA | `split_dir: data/docvqa_split` | in `skillopt_new_zzw` |
| `livemathematicianbench` | single-round QA | `split_mode=ratio` or prepared `split_dir` | in `skillopt_new_zzw` |
| `mathverse` | single-round multimodal math QA | `data_root: data/MathVerse`, split files loaded from `split_dir` when provided | in `skillopt_new_zzw` |
| `mmrb` | single-round multimodal reasoning QA | `split_mode=ratio` or prepared `split_dir` | in `skillopt_new_zzw` |
| `officeqa` | multi-turn tool loop | `split_dir: data/officeqa_split` plus `data_dirs: [data/officeqa_docs_official]` | in `skillopt_new_zzw` |
| `sealqa` | multi-turn tool loop | `split_dir: data/sealqa_split` | in `skillopt_new_zzw` |
| `searchqa` | single-round QA (`max_turns=1`) | `split_dir: data/searchqa_split` | in `skillopt_new_zzw` |
| `spreadsheetbench` | codegen loop, default `mode=multi`, `max_turns=30` | `split_dir: data/spreadsheetbench_split`, `data_root: data/spreadsheetbench_verified_400` | in `skillopt_new_zzw`, default adjusted here to multi-round |
| `swebench` | mini-swe-agent multi-step bug-fixing rollout | `split_mode=ratio`, `dataset_name=lite`, repo-stratified `2:1:7` split materialized under `out_root/_generated_splits/...` unless `split_dir` is provided | added here, aligned to `swe-bench-old` |

## Data Expectations

The standard two-mode dataset entry path is:

- `split_mode: ratio`
  - load raw data from `env.data_path`
  - build a deterministic `train/`, `val/`, `test/` split under `env.split_output_dir` (or under `out_root/_generated_splits/` if unset)
  - default ratio is explicitly `2:1:7`
- `split_mode: split_dir`
  - load an existing `env.split_dir` with `train/`, `val/`, `test/` subdirectories

This currently applies to:

- `searchqa`
- `spreadsheetbench`
- `babyvision`
- `livemathematicianbench`
- `mmrb`
- `swebench`

`ALFWorld` is the exception: it is environment-backed rather than JSON split-backed.

The following environments currently expect prepared split directories or extra rooted assets rather than the generic ratio-split path:

- `docvqa`
- `mathverse`
- `officeqa`
- `sealqa`

At a high level:

- `SearchQA`: raw QA json / jsonl or pre-split QA json files
- `SpreadsheetBench`: raw task manifest json plus spreadsheet task directory, or a pre-split task manifest
- `ALFWorld`: installed game environment and configured eval/train splits
- `BabyVision`: raw `meta_data.jsonl` plus images, or a pre-split directory
- `DocVQA`: pre-split CSV / JSON data under `split_dir`
- `LiveMathematicianBench`: raw monthly QA json files, or a pre-split directory
- `MathVerse`: split files plus `data_root` image assets
- `MMRB`: raw extracted dataset json files, or a pre-split directory
- `OfficeQA`: pre-split metadata plus resolved office document directories
- `SealQA`: pre-split metadata for tool-augmented QA tasks
- `SWEBench`: HuggingFace SWE-bench dataset alias (`lite` / `verified` / `full`) or a prepared split directory

### Split References Across Branches

The split-related defaults are not identical across `skillopt-final`, `skillopt_new_zzw`, `gepa`, and `swe-bench-old`. The practical reference points are:

| Source branch | Explicit split settings / dirs |
|---|---|
| `skillopt-final` | `searchqa -> data/searchqa_split`; `spreadsheetbench -> data/spreadsheetbench_split`; `docvqa -> data/docvqa_split`; `officeqa -> data/officeqa_split`; `sealqa -> data/sealqa_split`; `swebench -> ratio split 2:1:7 over the default lite dataset, materialized under out_root/_generated_splits/...` |
| `skillopt_new_zzw` | Same 10-benchmark env set as above except no `swebench`; explicit split dirs are `data/searchqa_split`, `data/spreadsheetbench_split`, `data/docvqa_split`, `data/officeqa_split`, `data/sealqa_split`; `spreadsheetbench` there defaults to `mode=single`; `officeqa` uses `max_tool_turns=24`; `sealqa` uses `max_tool_turns=12` |
| `gepa` | `configs/spreadsheetbench.yaml` uses `data.splits_dir = data/spreadsheetbench/splits`, `eval.mode = react`, `eval.max_turns = 20`; `configs/swebench.yaml` uses `dataset = SWE-bench/SWE-bench_Verified` with `train_size = 100`, `val_size = 50`, `test_size = 350` |
| `swe-bench-old` | Repo-stratified `2:1:7` split over `SWE-Bench_Lite`, persisted as `outputs/.../split/train.json`, `selection.json`, `test.json`; the example split in that branch is `train=60`, `selection=33`, `test=207` |

For the 10 benches shared with `skillopt_new_zzw`, the current branch is now aligned on env coverage. The main intentional delta is `spreadsheetbench`: this branch defaults to multi-round codegen, while `skillopt_new_zzw` kept `mode=single` by default.

## Running Training

Example:

```bash
python scripts/train.py --config configs/searchqa/default.yaml
```

Explicit 2:1:7 split from raw data:

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --split_mode ratio \
  --data_path /path/to/searchqa_train_2000.json
```

Directly consume a prepared split directory:

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --split_mode split_dir \
  --split_dir /path/to/searchqa_split
```

You can override structured config keys from the CLI:

```bash
python scripts/train.py \
  --config configs/spreadsheetbench/default.yaml \
  --cfg-options model.teacher_backend=openai_chat model.student_backend=codex_exec train.batch_size=40 optimizer.learning_rate=4
```

Legacy flat overrides still work for common keys:

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --backend azure_openai \
  --teacher_model gpt-5.4 \
  --student_model gpt-5.4 \
  --reasoning_effort medium
```

Exec harness example:

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --teacher_backend openai_chat \
  --student_backend codex_exec \
  --teacher_model gpt-5.4 \
  --student_model gpt-5.4-codex \
  --use_deep_reflect true \
  --skill_update_mode rewrite_from_suggestions
```

SWEBench example:

```bash
python scripts/train.py \
  --config configs/swebench/default.yaml \
  --cfg-options env.dataset_name=lite env.split_ratio=2:1:7
```

## Eval-Only and Standalone Evaluation

Evaluate a specific skill without training:

```bash
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill skillopt/envs/searchqa/skills/initial.md
```

The same dataset entry modes apply in eval-only runs:

- `--split_mode ratio --data_path ...`
- `--split_mode split_dir --split_dir ...`

Standalone scripts also exist for benchmark-specific comparisons, including:

- `scripts/eval_prompt_custom.py`
- `scripts/eval_prompt_official.py`
- `scripts/eval_livemathematicianbench_baseline.py`

These scripts now also support backend selection through the unified model layer.

## Output Structure

Each run writes a structured output directory under `out_root`.

Important top-level artifacts:

- `config.json` — flattened runtime config
- `history.json` — per-step history records
- `runtime_state.json` — resume state for current/best skill tracking
- `best_skill.md` — current best validated skill
- `skills/skill_vXXXX.md` — persisted skill snapshot per step

Per-step artifacts live under `steps/step_XXXX/`, including:

- `merged_patch.json`
- `ranked_edits.json`
- `candidate_skill.md`
- `edit_apply_report.json`
- `rewrite_result.json` when rewrite mode is enabled
- `selection_eval/`
- `trajectory_digest.json`
- rollout and patch subdirectories

Epoch-level artifacts live under:

- `slow_update/epoch_XX/`
- `meta_skill/epoch_XX/`
- `meta_reflect/epoch_XX/`

## Resume Behavior

The trainer resumes from `runtime_state.json` when present. That state tracks:

- last completed step
- current skill path
- current score
- best skill path
- best score
- origin tags for current and best skill

This is important because skill state can change at both step level and epoch level; resuming only from `history.json` is not sufficient for this branch’s method logic.

## Notes

- This repository focuses on skill optimization logic; datasets are not included.
- Patch application is intentionally observable. Inspect `edit_apply_report.json` when candidate skills do not behave as expected.
- `SpreadsheetBench` now defaults to `mode=multi`. If you run an exec student backend there, override back to `env.mode=single` because exec backends are still only wired for SpreadsheetBench single-mode rollout.
- `SWEBench` follows the older mini-swe-agent + `swebench.harness.run_evaluation` path, so it requires the SWE-bench / Docker toolchain rather than the generic chat-only stack.
- `slow_update` writes into a protected skill region and normal edits are prevented from overwriting that region directly.
- `meta_skill` is context memory, not a direct skill edit.
- `meta_reflect` is a gated skill edit stage, not just logging.

## Minimal Setup

```bash
conda create -n skillopt python=3.11
conda activate skillopt
pip install openai pyyaml openpyxl
```

Depending on the environment, you may also need:

```bash
pip install datasets gymnasium numpy ray regex
```

For `SWEBench`, you also need a working Docker environment plus the SWE-bench / mini-swe-agent dependencies used in `swe-bench-old`.

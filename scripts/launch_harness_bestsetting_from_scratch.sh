#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/harness_bestsetting_fromscratch_$(date -u +%Y%m%d_%H%M%S)_run}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

COMMON=(
  model.teacher_backend=openai_chat
  model.teacher=gpt-5.5
  model.teacher_azure_openai_endpoint=https://t2vgoaigpt4o3.openai.azure.com/
  model.teacher_azure_openai_api_version=2024-12-01-preview
  model.teacher_azure_openai_auth_mode=azure_cli
  model.reasoning_effort=medium
  train.num_epochs=4
  train.train_size=0
  train.accumulation=1
  train.seed=42
  gradient.minibatch_size=8
  gradient.merge_batch_size=8
  gradient.analyst_workers=16
  gradient.use_deep_reflect=false
  optimizer.min_learning_rate=2
  optimizer.lr_control_mode=fixed
  optimizer.skill_update_mode=patch
  optimizer.use_slow_update=true
  optimizer.slow_update_samples=20
  optimizer.use_meta_skill=true
  optimizer.use_meta_reflect=false
  evaluation.use_gate=true
  evaluation.eval_test=true
  env.split_mode=split_dir
)

CODEX=(
  model.student_backend=codex_exec
  model.student=gpt-5.5
  model.codex_exec_use_sdk=auto
  model.codex_exec_sandbox=workspace-write
  model.codex_exec_approval_policy=never
  model.codex_trace_to_teacher=true
)

CLAUDE=(
  model.student_backend=claude_code_exec
  model.student=claude-sonnet-4-6
  model.claude_code_exec_use_sdk=auto
  model.codex_trace_to_teacher=false
)

active=0
launch() {
  local run_id="$1"; shift
  local config="$1"; shift
  local out="$RUN_ROOT/$run_id"
  local log="$RUN_ROOT/logs/$run_id.log"
  echo "START $run_id"
  setsid "$PY" -u scripts/train.py \
    --config "$config" \
    --cfg-options "${COMMON[@]}" "$@" "env.out_root=$out" \
    > "$log" 2>&1 < /dev/null &
  active=$((active + 1))
  if (( active >= MAX_PARALLEL )); then
    wait -n
    active=$((active - 1))
  fi
}

# SearchQA best openai-chat setting: optimizer.lr_scheduler=constant.
launch HARNESS-BESTSETTING-searchqa-codex configs/searchqa/default.yaml \
  "${CODEX[@]}" \
  train.batch_size=40 optimizer.learning_rate=4 optimizer.min_learning_rate=2 optimizer.lr_scheduler=constant \
  env.split_dir=data/searchqa/splits

launch HARNESS-BESTSETTING-searchqa-claude configs/searchqa/default.yaml \
  "${CLAUDE[@]}" \
  train.batch_size=40 optimizer.learning_rate=4 optimizer.min_learning_rate=2 optimizer.lr_scheduler=constant \
  env.split_dir=data/searchqa/splits

# SpreadsheetBench best openai-chat setting: optimizer.lr_scheduler=constant.
# Must stay env.mode=multi; exec-backend multi support is fixed on this branch.
launch HARNESS-BESTSETTING-spreadsheetbench-codex configs/spreadsheetbench/default.yaml \
  "${CODEX[@]}" \
  train.batch_size=40 optimizer.learning_rate=4 optimizer.min_learning_rate=2 optimizer.lr_scheduler=constant \
  env.split_dir=data/spreadsheetbench env.data_root=data/spreadsheetbench/files env.mode=multi env.workers=4

launch HARNESS-BESTSETTING-spreadsheetbench-claude configs/spreadsheetbench/default.yaml \
  "${CLAUDE[@]}" \
  train.batch_size=40 optimizer.learning_rate=4 optimizer.min_learning_rate=2 optimizer.lr_scheduler=constant \
  env.split_dir=data/spreadsheetbench env.data_root=data/spreadsheetbench/files env.mode=multi env.workers=4

# LiveMathBench best openai-chat setting: optimizer.learning_rate=8.
launch HARNESS-BESTSETTING-livemathematicianbench-codex configs/livemathematicianbench/default.yaml \
  "${CODEX[@]}" \
  train.batch_size=40 optimizer.learning_rate=8 optimizer.min_learning_rate=1 optimizer.lr_scheduler=constant \
  env.split_dir=data/livemathbench/splits

launch HARNESS-BESTSETTING-livemathematicianbench-claude configs/livemathematicianbench/default.yaml \
  "${CLAUDE[@]}" \
  train.batch_size=40 optimizer.learning_rate=8 optimizer.min_learning_rate=1 optimizer.lr_scheduler=constant \
  env.split_dir=data/livemathbench/splits

# DocVQA best openai-chat setting was full batch. On 10% harness split, train=107.
launch HARNESS-BESTSETTING-docvqa10pct-codex configs/docvqa/default.yaml \
  "${CODEX[@]}" \
  train.batch_size=107 optimizer.learning_rate=4 optimizer.min_learning_rate=2 optimizer.lr_scheduler=cosine \
  env.split_dir=data/harness_splits/docvqa_zisu_first10pct

launch HARNESS-BESTSETTING-docvqa10pct-claude configs/docvqa/default.yaml \
  "${CLAUDE[@]}" \
  train.batch_size=107 optimizer.learning_rate=4 optimizer.min_learning_rate=2 optimizer.lr_scheduler=cosine \
  env.split_dir=data/harness_splits/docvqa_zisu_first10pct

wait
echo "All launched runs finished or exited. RUN_ROOT=$RUN_ROOT"

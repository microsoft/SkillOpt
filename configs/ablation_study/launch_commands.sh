#!/usr/bin/env bash
set -euo pipefail

cd /home/azureuser/workspace-gzy/SkillReflection

PY=/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python
export ALFWORLD_DATA=/home/azureuser/.cache/alfworld

# Original SearchQA / SpreadsheetBench full matrix reproduction command.
# Do not run this into the existing root unless intentionally reproducing from
# scratch; the current valid root is already populated:
# outputs/ablation_20260502_040604_unique48
#
# setsid "$PY" scripts/run_ablation_matrix.py \
#   --groups default split mbs lr sched slown mod smodel \
#   --bench searchqa spreadsheetbench \
#   --run-root /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_20260502_040604_unique48 \
#   --max-parallel 24 \
#   --execute \
#   > /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_20260502_040604_unique48/launcher_reproduce_full_matrix.log 2>&1 < /dev/null &
#
# SearchQA / SpreadsheetBench batch-size ablations only.
# Original non-batch SearchQA/SpreadsheetBench ablations live in:
# outputs/ablation_20260502_040604_unique48
setsid "$PY" scripts/run_ablation_matrix.py \
  --groups batch \
  --bench searchqa spreadsheetbench \
  --run-root /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_batch_searchqa_spreadsheet_20260503_153902_run \
  --max-parallel 8 \
  --execute \
  > /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_batch_searchqa_spreadsheet_20260503_153902_run/launcher_parallel8.log 2>&1 < /dev/null &

# DocVQA full matrix.
setsid "$PY" scripts/run_ablation_matrix.py \
  --groups default split batch mbs lr sched slown mod smodel \
  --bench docvqa \
  --run-root /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_docvqa_20260503_160225_run \
  --max-parallel 8 \
  --execute \
  > /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_docvqa_20260503_160225_run/launcher_parallel8.log 2>&1 < /dev/null &

# LiveMathBench clean matrix. ALFWorld should be launched separately at lower
# concurrency because Ray OOM occurred when many ALFWorld runs were mixed into a
# 24-way run.
setsid "$PY" scripts/run_ablation_matrix.py \
  --groups default split batch mbs lr sched slown mod smodel \
  --bench livemathematicianbench \
  --run-root /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_livemath_alfworld_clean_20260503_155155_run \
  --max-parallel 8 \
  --execute \
  > /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_livemath_alfworld_clean_20260503_155155_run/launcher_livemath_parallel8.log 2>&1 < /dev/null &

# ALFWorld clean matrix. Increase to 2 only after checking memory, /tmp/ray,
# and that no other ALFWorld run is active. Do not use 8/16/24 for ALFWorld on
# the current shared machine unless resources are explicitly reserved.
setsid "$PY" scripts/run_ablation_matrix.py \
  --groups default split batch mbs lr sched slown mod smodel \
  --bench alfworld \
  --run-root /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_livemath_alfworld_clean_20260503_155155_run \
  --max-parallel 1 \
  --execute \
  > /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_livemath_alfworld_clean_20260503_155155_run/launcher_alfworld_parallel1.log 2>&1 < /dev/null &

# Longitudinal comparison-example policy ablations. This intentionally excludes
# ALFWorld. The only varied setting is optimizer.longitudinal_pair_policy.
setsid "$PY" scripts/run_ablation_matrix.py \
  --groups longpair \
  --bench searchqa spreadsheetbench livemathematicianbench docvqa \
  --run-root /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_longpair_20260504_run \
  --max-parallel 8 \
  --execute \
  > /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_longpair_20260504_run/launcher_longpair_parallel8.log 2>&1 < /dev/null &

# Learning-rate-control baselines. This intentionally excludes ALFWorld.
setsid "$PY" scripts/run_ablation_matrix.py \
  --groups lrctrl \
  --bench searchqa spreadsheetbench livemathematicianbench docvqa \
  --run-root /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_lrctrl_20260504_run \
  --max-parallel 8 \
  --execute \
  > /home/azureuser/workspace-gzy/SkillReflection/outputs/ablation_lrctrl_20260504_run/launcher_lrctrl_parallel8.log 2>&1 < /dev/null &

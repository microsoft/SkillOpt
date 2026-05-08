#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/azureuser/workspace-gzy/SkillReflection_dev"
PYTHON_BIN="/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python"
RUN_ROOT="${PROJECT_ROOT}/outputs/meta_skill_parallel_20260430_072356"
LOG_DIR="${PROJECT_ROOT}/logs/meta_skill_parallel_20260430_072356"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

start_run() {
  local name="$1"
  local config_path="$2"
  local meta_skill="$3"
  local out_root="${RUN_ROOT}/${name}"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[START] ${name}"
  echo "        out_root=${out_root}"
  echo "        log=${log_path}"

  (
    cd "${PROJECT_ROOT}"
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" scripts/train.py \
      --config "${config_path}" \
      --cfg-options \
        "model.reasoning_effort=medium" \
        "optimizer.use_slow_update=true" \
        "optimizer.use_meta_skill=${meta_skill}" \
        "optimizer.use_meta_reflect=false" \
        "gradient.use_deep_reflect=false" \
        "env.out_root=${out_root}"
  ) > "${log_path}" 2>&1 &

  echo "$!" > "${LOG_DIR}/${name}.pid"
}

start_run "searchqa_medium_slow_meta" "configs/searchqa/default.yaml" "true"
start_run "spreadsheetbench_medium_slow" "configs/spreadsheetbench/default.yaml" "false"
start_run "spreadsheetbench_medium_slow_meta" "configs/spreadsheetbench/default.yaml" "true"

echo "[WAIT] missing comparison runs are active"
wait

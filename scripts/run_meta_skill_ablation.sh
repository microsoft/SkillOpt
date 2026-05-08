#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/azureuser/workspace-gzy/SkillReflection_dev"
PYTHON_BIN="/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python"
TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${PROJECT_ROOT}/outputs/meta_skill_ablation_${TS}"

mkdir -p "${RUN_ROOT}"

run_train() {
  local benchmark="$1"
  local reasoning="$2"
  local condition="$3"
  local config_path=""
  local reasoning_override=""
  local meta_skill_flag=""

  case "${benchmark}" in
    searchqa)
      config_path="configs/searchqa/default.yaml"
      ;;
    spreadsheetbench)
      config_path="configs/spreadsheetbench/default.yaml"
      ;;
    *)
      echo "Unknown benchmark: ${benchmark}" >&2
      exit 1
      ;;
  esac

  case "${reasoning}" in
    medium)
      reasoning_override="model.reasoning_effort=medium"
      ;;
    none)
      reasoning_override="model.reasoning_effort="
      ;;
    *)
      echo "Unknown reasoning setting: ${reasoning}" >&2
      exit 1
      ;;
  esac

  case "${condition}" in
    slow)
      meta_skill_flag="optimizer.use_meta_skill=false"
      ;;
    slow_meta)
      meta_skill_flag="optimizer.use_meta_skill=true"
      ;;
    *)
      echo "Unknown condition: ${condition}" >&2
      exit 1
      ;;
  esac

  local out_root="${RUN_ROOT}/${benchmark}_${reasoning}_${condition}"

  echo
  echo "============================================================"
  echo "START ${benchmark} ${reasoning} ${condition}"
  echo "out_root=${out_root}"
  echo "============================================================"

  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" scripts/train.py \
      --config "${config_path}" \
      --cfg-options \
        "${reasoning_override}" \
        "optimizer.use_slow_update=true" \
        "${meta_skill_flag}" \
        "optimizer.use_meta_reflect=false" \
        "gradient.use_deep_reflect=false" \
        "env.out_root=${out_root}"
  )

  echo
  echo "============================================================"
  echo "DONE ${benchmark} ${reasoning} ${condition}"
  echo "============================================================"
}

for benchmark in searchqa spreadsheetbench; do
  for reasoning in medium none; do
    run_train "${benchmark}" "${reasoning}" "slow"
    run_train "${benchmark}" "${reasoning}" "slow_meta"
  done
done

echo
echo "All runs completed."
echo "Run root: ${RUN_ROOT}"

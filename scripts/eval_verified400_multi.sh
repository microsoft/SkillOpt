#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Eval skill0 on full SpreadsheetBench verified-400 (MULTI-ROUND codegen)
#
# Usage:
#   bash scripts/eval_verified400_multi.sh
#   bash scripts/eval_verified400_multi.sh --workers 64 --max_turns 5
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATA_ROOT="/home/azureuser/workspace-yqh/sr/spreadsheetbench/data/spreadsheetbench_verified_400"
JSONL_PATH="${DATA_ROOT}/dataset.json"
SKILL_PATH="${PROJECT_ROOT}/skillopt/envs/spreadsheetbench/skills/initial.md"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${PROJECT_ROOT}/outputs/eval_multi_verified400_${TIMESTAMP}"

echo "============================================================"
echo "  Eval skill0 — MULTI-ROUND codegen — verified-400"
echo "============================================================"
echo "  data_root:  ${DATA_ROOT}"
echo "  skill:      ${SKILL_PATH}"
echo "  mode:       multi"
echo "  out_root:   ${OUT_ROOT}"
echo "============================================================"

cd "${PROJECT_ROOT}"

python scripts/eval_only.py \
    --config configs/spreadsheetbench_default.yaml \
    --skill "${SKILL_PATH}" \
    --split all \
    --mode multi \
    --data_root "${DATA_ROOT}" \
    --jsonl_path "${JSONL_PATH}" \
    --out_root "${OUT_ROOT}" \
    "$@"

#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ReflACT — SpreadsheetBench training (SINGLE-ROUND codegen, no tool-call)
#
# Usage:
#   bash scripts/train_spreadsheet_single.sh
#   bash scripts/train_spreadsheet_single.sh --num_epochs 2 --edit_budget 6
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TEACHER_DEPLOYMENT="${TEACHER_DEPLOYMENT:-gpt-5.5}"
export STUDENT_DEPLOYMENT="${STUDENT_DEPLOYMENT:-gpt-5.5}"

DATA_ROOT="/home/azureuser/workspace-yqh/sr/spreadsheetbench/data/spreadsheetbench_verified_400"
JSONL_PATH="${DATA_ROOT}/dataset.json"
SPLIT_DIR="/home/azureuser/workspace-yqh/refleACT3/data/spreadsheetbench_split_2_1_7"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${PROJECT_ROOT}/outputs/spreadsheet-metaskill-new/train_single_${STUDENT_DEPLOYMENT}"

echo "============================================================"
echo "  ReflACT — SpreadsheetBench Training (SINGLE-ROUND)"
echo "============================================================"
echo "  Teacher:    ${TEACHER_DEPLOYMENT}"
echo "  Student:    ${STUDENT_DEPLOYMENT}"
echo "  Mode:       single"
echo "  Data:       ${DATA_ROOT}"
echo "  Split:      ${SPLIT_DIR}"
echo "  Output:     ${OUT_ROOT}"
echo "============================================================"

cd "${PROJECT_ROOT}"

python scripts/train.py \
    --config configs/spreadsheetbench_default.yaml \
    --mode single \
    --data_root "${DATA_ROOT}" \
    --jsonl_path "${JSONL_PATH}" \
    --split_dir "${SPLIT_DIR}" \
    --out_root "${OUT_ROOT}" \
    "$@"

echo ""
echo "Done! Results saved to: ${OUT_ROOT}"

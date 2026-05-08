#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Eval skill0 on full SpreadsheetBench verified-400
#
# Usage:
#   bash scripts/eval_verified400.sh
#   bash scripts/eval_verified400.sh --workers 64
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT="/home/azureuser/workspace-yqh/sr/spreadsheetbench/data/spreadsheetbench_verified_400"
JSONL_PATH="${DATA_ROOT}/dataset.json"
SKILL_PATH="${PROJECT_ROOT}/skillopt/envs/spreadsheetbench/skills/initial.md"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${PROJECT_ROOT}/outputs/eval_verified400_${TIMESTAMP}"

echo "============================================================"
echo "  Eval skill0 on verified-400 (full)"
echo "============================================================"
echo "  data_root:  ${DATA_ROOT}"
echo "  skill:      ${SKILL_PATH}"
echo "  out_root:   ${OUT_ROOT}"
echo "============================================================"

cd "${PROJECT_ROOT}"

python scripts/eval_only.py \
    --config configs/spreadsheetbench_default.yaml \
    --skill "${SKILL_PATH}" \
    --split all \
    --data_root "${DATA_ROOT}" \
    --jsonl_path "${JSONL_PATH}" \
    --out_root "${OUT_ROOT}" \
    "$@"

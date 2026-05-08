#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ReflACT3 — SearchQA Training
#
# Usage:
#   bash scripts/train_searchqa.sh
#   bash scripts/train_searchqa.sh --limit 50
#   bash scripts/train_searchqa.sh --num_epochs 2 --workers 32
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Models ───────────────────────────────────────────────────────────────────
export TEACHER_DEPLOYMENT="${TEACHER_DEPLOYMENT:-gpt-5.5}"
export STUDENT_DEPLOYMENT="${STUDENT_DEPLOYMENT:-gpt-5.5}"

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_PATH="/home/azureuser/workspace-yqh/refleAct/search-qa/data/searchqa_train_2000.json"

# ── Output ───────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_OUT_ROOT="${PROJECT_ROOT}/outputs/searchqa-metaskill/searchqa_${STUDENT_DEPLOYMENT}"

echo "============================================================"
echo "  ReflACT3 — SearchQA Training"
echo "  Teacher:  ${TEACHER_DEPLOYMENT}"
echo "  Student:  ${STUDENT_DEPLOYMENT}"
echo "  Data:     ${DATA_PATH}"
echo "============================================================"

cd "${PROJECT_ROOT}"

python scripts/train.py \
    --config configs/searchqa_default.yaml \
    --data_path "${DATA_PATH}" \
    --out_root "${DEFAULT_OUT_ROOT}" \
    "$@"

echo ""
echo "Done! Results saved to: ${DEFAULT_OUT_ROOT}"

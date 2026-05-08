#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ReflACT3 — SearchQA Eval-Only (验证集 500)
#
# Usage:
#   bash scripts/eval_searchqa_val500.sh --skill_path outputs/xxx/best_skill.md
#   bash scripts/eval_searchqa_val500.sh --skill_path outputs/xxx/best_skill.md --workers 32
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export STUDENT_DEPLOYMENT="${STUDENT_DEPLOYMENT:-gpt-5-mini}"

VAL_PATH="/home/azureuser/workspace-yqh/refleAct/search-qa/data/searchqa_val_500.json"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_OUT_ROOT="${PROJECT_ROOT}/outputs/searchqa_eval_val500_${STUDENT_DEPLOYMENT}_${TIMESTAMP}"

echo "============================================================"
echo "  ReflACT3 — SearchQA Eval-Only (val-500)"
echo "  Student:  ${STUDENT_DEPLOYMENT}"
echo "  Data:     ${VAL_PATH}"
echo "============================================================"

cd "${PROJECT_ROOT}"

python scripts/eval_only.py \
    --config configs/searchqa_default.yaml \
    --data_path "${VAL_PATH}" \
    --out_root "${DEFAULT_OUT_ROOT}" \
    "$@"

echo ""
echo "Done! Results saved to: ${DEFAULT_OUT_ROOT}"

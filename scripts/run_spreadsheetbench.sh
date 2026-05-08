#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ReflACT — SpreadsheetBench training launch script
#
# Usage:
#   bash scripts/run_spreadsheetbench.sh \
#       --data_root /path/to/data \
#       --jsonl_path /path/to/benchmark.jsonl
#
#   bash scripts/run_spreadsheetbench.sh \
#       --data_root /path/to/data \
#       --jsonl_path /path/to/benchmark.jsonl \
#       --num_epochs 2 --edit_budget 6
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Ensure ReflACT is importable
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Model configuration ─────────────────────────────────────────────────────
export TEACHER_DEPLOYMENT="${TEACHER_DEPLOYMENT:-gpt-5.5}"
export STUDENT_DEPLOYMENT="${STUDENT_DEPLOYMENT:-gpt-5.5}"

# ── Output directory ─────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_OUT_ROOT="${PROJECT_ROOT}/outputs/skillopt_spreadsheetbench_${STUDENT_DEPLOYMENT}_${TIMESTAMP}"

# ── Run ──────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  ReflACT — Reflective Agent Tuning (SpreadsheetBench)"
echo "============================================================"
echo "  Teacher:  ${TEACHER_DEPLOYMENT}"
echo "  Student:  ${STUDENT_DEPLOYMENT}"
echo "============================================================"

cd "${PROJECT_ROOT}"

python scripts/train.py \
    --config configs/spreadsheetbench_default.yaml \
    --out_root "${DEFAULT_OUT_ROOT}" \
    "$@"

echo ""
echo "Done! Results saved to: ${DEFAULT_OUT_ROOT}"

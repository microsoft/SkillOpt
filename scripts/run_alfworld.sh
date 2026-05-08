#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ReflACT — ALFWorld training launch script
#
# Usage:
#   bash scripts/run_alfworld.sh
#   bash scripts/run_alfworld.sh --num_epochs 2 --edit_budget 6
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE="/home/azureuser/workspace-gzy"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Activate conda environment
export PATH="${WORKSPACE}/miniconda3/envs/skillopt/bin:${WORKSPACE}/miniconda3/bin:${PATH}"

# ALFWorld data — uses ~/.cache/alfworld by default (standard alfworld location)
export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"

# Ensure ReflACT is importable
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Verify ALFWorld data exists ──────────────────────────────────────────────
if [ ! -d "${ALFWORLD_DATA}/json_2.1.1" ]; then
    echo "ERROR: ALFWorld data not found at ${ALFWORLD_DATA}/json_2.1.1"
    echo ""
    echo "To download ALFWorld data, run:"
    echo "  pip install alfworld[full]"
    echo "  alfworld-download"
    echo ""
    echo "Or set ALFWORLD_DATA to the directory containing json_2.1.1/"
    exit 1
fi

# ── Azure OpenAI credentials ────────────────────────────────────────────────
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://agl-dev.cognitiveservices.azure.com/}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-<your-azure-openai-api-key>}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2025-04-01-preview}"

# ── Model configuration ─────────────────────────────────────────────────────
export TEACHER_DEPLOYMENT="${TEACHER_DEPLOYMENT:-gpt-5.5}"
export STUDENT_DEPLOYMENT="${STUDENT_DEPLOYMENT:-gpt-5.5}"

# ── Output directory ─────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_OUT_ROOT="${PROJECT_ROOT}/outputs/skillopt_alfworld_${STUDENT_DEPLOYMENT}_${TIMESTAMP}"

# ── Run ──────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  ReflACT — Reflective Agent Tuning (ALFWorld)"
echo "============================================================"
echo "  Teacher:       ${TEACHER_DEPLOYMENT}"
echo "  Student:       ${STUDENT_DEPLOYMENT}"
echo "  ALFWORLD_DATA: ${ALFWORLD_DATA}"
echo "  Output:        ${DEFAULT_OUT_ROOT}"
echo "============================================================"

cd "${PROJECT_ROOT}"

python scripts/train.py \
    --config configs/alfworld_default.yaml \
    --out_root "${DEFAULT_OUT_ROOT}" \
    "$@"

echo ""
echo "Done! Results saved to: ${DEFAULT_OUT_ROOT}"

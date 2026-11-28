#!/bin/bash
# run_sleep_cron.sh — wrapper for cron-driven nightly sleep cycle
#
# Usage: bash run_sleep_cron.sh [category1 category2 ...]
#   No args: run on all categories in tests/
#   With args: run only on listed categories (research-cron, devops, wiki)
#
# Cron (3am MYT daily):
#   0 3 * * * cd /path/to/SkillOpt/plugins/openclaw && bash run_sleep_cron.sh >> ~/.skillopt-sleep/nightly.log 2>&1

set -euo pipefail

SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$SKILL_DIR/tests"
OPENCLAW_HOME="${OPENCLAW_HOME:-$HOME}"
case "${OPENCLAW_HOME,,}" in
    ""|undefined|null) OPENCLAW_HOME="$HOME" ;;
esac
case "$OPENCLAW_HOME" in
    "~") OPENCLAW_HOME="$HOME" ;;
    "~/"*) OPENCLAW_HOME="$HOME/${OPENCLAW_HOME#\~/}" ;;
    "~\\"*) OPENCLAW_HOME="$HOME/${OPENCLAW_HOME#\~\\}" ;;
esac
OPENCLAW_PROFILE="${OPENCLAW_PROFILE:-}"
if [[ "$OPENCLAW_PROFILE" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]]; then
    if [ "$(printf '%s' "$OPENCLAW_PROFILE" | tr '[:upper:]' '[:lower:]')" = "default" ]; then
        OPENCLAW_PROFILE=""
    fi
else
    OPENCLAW_PROFILE=""
fi
OPENCLAW_STATE_DIR="${OPENCLAW_STATE_DIR:-}"
case "${OPENCLAW_STATE_DIR,,}" in
    ""|undefined|null)
        if [ -n "$OPENCLAW_PROFILE" ]; then
            OPENCLAW_STATE_DIR="$OPENCLAW_HOME/.openclaw-$OPENCLAW_PROFILE"
        else
            OPENCLAW_STATE_DIR="$OPENCLAW_HOME/.openclaw"
        fi
        ;;
esac
case "$OPENCLAW_STATE_DIR" in
    "~") OPENCLAW_STATE_DIR="$HOME" ;;
    "~/"*) OPENCLAW_STATE_DIR="$HOME/${OPENCLAW_STATE_DIR#\~/}" ;;
    "~\\"*) OPENCLAW_STATE_DIR="$HOME/${OPENCLAW_STATE_DIR#\~\\}" ;;
esac
SKILLOPT_STATE_DIR="${SKILLOPT_SLEEP_STATE_DIR:-$HOME/.skillopt-sleep}"
LOG_DIR="$SKILLOPT_STATE_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/night-$TIMESTAMP.log"

# category → test file map
declare -A CATEGORIES=(
    ["research-cron"]="research-cron-tasks.json"
    ["devops"]="devops-tasks.json"
    ["wiki"]="wiki-tasks.json"
)

# Determine which categories to run
if [ $# -eq 0 ]; then
    CATS=("research-cron" "devops" "wiki")
else
    CATS=("$@")
fi

{
    echo "=========================================="
    echo "SkillOpt-Sleep nightly — $TIMESTAMP"
    echo "Categories: ${CATS[*]}"
    echo "=========================================="
} | tee -a "$LOG_FILE"

# Pre-flight: check DeepSeek API key
if ! grep -q "DEEPSEEK_API_KEY=" "$OPENCLAW_STATE_DIR/.env" 2>/dev/null; then
    echo "ERROR: DEEPSEEK_API_KEY not found in $OPENCLAW_STATE_DIR/.env" | tee -a "$LOG_FILE"
    exit 1
fi

EXIT_CODE=0
for cat in "${CATS[@]}"; do
    tasks_file="$TESTS_DIR/${CATEGORIES[$cat]:-}"
    if [ ! -f "$tasks_file" ]; then
        echo "SKIP: $cat (no tasks file: $tasks_file)" | tee -a "$LOG_FILE"
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "--- [$cat] starting cycle ---" | tee -a "$LOG_FILE"

    cd "$SKILL_DIR"
    if python3 "$SKILL_DIR/run_sleep.py" --tasks "$tasks_file" 2>&1 | tee -a "$LOG_FILE"; then
        echo "--- [$cat] OK ---" | tee -a "$LOG_FILE"
    else
        EC=$?
        echo "--- [$cat] FAILED (exit $EC) ---" | tee -a "$LOG_FILE"
        EXIT_CODE=$EC
    fi
done

{
    echo ""
    echo "=========================================="
    echo "Done. Exit: $EXIT_CODE"
    echo "=========================================="
} | tee -a "$LOG_FILE"

exit $EXIT_CODE

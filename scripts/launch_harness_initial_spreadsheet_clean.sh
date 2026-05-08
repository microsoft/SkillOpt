#!/usr/bin/env bash
set -euo pipefail

REPO="/home/azureuser/workspace-gzy/SkillReflection"
PYTHON="/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python"

cd "$REPO"

export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:4343}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-dummy}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-sonnet-4-6}"
export ANTHROPIC_SMALL_FAST_MODEL="${ANTHROPIC_SMALL_FAST_MODEL:-claude-sonnet-4-6}"
export DISABLE_NON_ESSENTIAL_MODEL_CALLS="${DISABLE_NON_ESSENTIAL_MODEL_CALLS:-1}"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}"

if [[ -f ".secrets/teacher_oaidr9.env" ]]; then
  # shellcheck disable=SC1091
  source ".secrets/teacher_oaidr9.env"
else
  echo "missing .secrets/teacher_oaidr9.env" >&2
  exit 1
fi

stamp="$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${1:-outputs/harness_initial_spreadsheet_clean_workers2_timeout1020_${stamp}_run}"
SESSION="${2:-harness_initial_spreadsheet_clean_${stamp}}"
RUN_ID="HARNESS-ClaudeInit-Spreadsheet-lr4-multi-clean"
SPLIT_DIR="${SPREADSHEET_SPLIT_DIR:-data/harness_splits/spreadsheetbench_full}"
DATA_ROOT="${SPREADSHEET_DATA_ROOT:-data/spreadsheetbench/files}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/commands"

cmd_file="$RUN_ROOT/commands/${RUN_ID}.sh"
log_file="$RUN_ROOT/logs/${RUN_ID}.log"
out_root="$RUN_ROOT/$RUN_ID"

cmd=(
  "$PYTHON" -u scripts/train.py
  --config configs/spreadsheetbench/default.yaml
  --cfg-options
  model.teacher_backend=openai_chat
  model.teacher=gpt-5.5
  model.teacher_azure_openai_endpoint="${TEACHER_AZURE_OPENAI_ENDPOINT}"
  model.teacher_azure_openai_api_version="${TEACHER_AZURE_OPENAI_API_VERSION}"
  model.teacher_azure_openai_auth_mode="${TEACHER_AZURE_OPENAI_AUTH_MODE}"
  model.teacher_azure_openai_managed_identity_client_id="${TEACHER_AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID}"
  model.teacher_azure_openai_ad_scope="${TEACHER_AZURE_OPENAI_AD_SCOPE}"
  model.reasoning_effort=medium
  train.num_epochs=4
  train.train_size=0
  train.batch_size=40
  train.accumulation=1
  train.seed=42
  gradient.minibatch_size=8
  gradient.merge_batch_size=8
  gradient.analyst_workers=16
  gradient.use_deep_reflect=false
  optimizer.lr_control_mode=fixed
  optimizer.skill_update_mode=patch
  optimizer.use_slow_update=true
  optimizer.slow_update_samples=20
  optimizer.use_meta_skill=true
  optimizer.use_meta_reflect=false
  evaluation.use_gate=true
  evaluation.eval_test=true
  env.split_mode=split_dir
  env.workers=2
  env.exec_timeout=1020
  model.student_backend=claude_code_exec
  model.student=claude-sonnet-4-6
  model.claude_code_exec_use_sdk=sdk
  model.claude_code_exec_effort=medium
  model.claude_code_exec_max_thinking_tokens=16384
  model.codex_trace_to_teacher=false
  env.out_root="$out_root"
  env.split_dir="$SPLIT_DIR"
  env.data_root="$DATA_ROOT"
  env.mode=multi
  optimizer.learning_rate=4
  optimizer.min_learning_rate=1
  optimizer.lr_scheduler=constant
)

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '$REPO'"
  printf 'export ANTHROPIC_BASE_URL=%q\n' "$ANTHROPIC_BASE_URL"
  printf 'export ANTHROPIC_AUTH_TOKEN=%q\n' "$ANTHROPIC_AUTH_TOKEN"
  printf 'export ANTHROPIC_MODEL=%q\n' "$ANTHROPIC_MODEL"
  printf 'export ANTHROPIC_SMALL_FAST_MODEL=%q\n' "$ANTHROPIC_SMALL_FAST_MODEL"
  printf 'export DISABLE_NON_ESSENTIAL_MODEL_CALLS=%q\n' "$DISABLE_NON_ESSENTIAL_MODEL_CALLS"
  printf 'export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=%q\n' "$CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"
  printf '%q ' "${cmd[@]}"
  printf ' >%q 2>&1 < /dev/null\n' "$log_file"
} > "$cmd_file"
chmod +x "$cmd_file"

tmux new-session -d -s "$SESSION" -n "$RUN_ID" "bash '$cmd_file'; code=\$?; echo EXIT:\$code; sleep 3600"

echo "RUN_ROOT=$RUN_ROOT"
echo "SESSION=$SESSION"
echo "RUN_ID=$RUN_ID"

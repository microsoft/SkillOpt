#!/usr/bin/env bash
set -euo pipefail

REPO="/home/azureuser/workspace-gzy/SkillReflection"
PYTHON="/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python"
CODEX_WRAPPER="$REPO/scripts/codex_azure_mi.sh"

cd "$REPO"

# Claude Code is routed through the local copilot-api proxy on this machine.
# Do not rely on interactive Claude login state inside tmux/train workers.
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
RUN_ROOT="${1:-outputs/harness_canonical_step12_wave1_workers2_timeout1020_${stamp}_run}"
SESSION="${2:-harness_canon_wave1_${stamp}}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/commands"

COMMON_CFG=(
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
)

tmux_started=0

launch_run() {
  local run_id="$1"
  local backend="$2"
  local config="$3"
  local skill="$4"
  local split_dir="$5"
  shift 5

  local -a backend_cfg=()
  if [[ "$backend" == "codex" ]]; then
    backend_cfg=(
      model.student_backend=codex_exec
      model.student=gpt-5.5
      model.codex_exec_path="$CODEX_WRAPPER"
      model.codex_exec_use_sdk=auto
      model.codex_exec_sandbox=workspace-write
      model.codex_exec_approval_policy=never
      model.codex_exec_reasoning_effort=medium
      model.codex_trace_to_teacher=true
    )
  elif [[ "$backend" == "claude" ]]; then
    backend_cfg=(
      model.student_backend=claude_code_exec
      model.student=claude-sonnet-4-6
      model.claude_code_exec_use_sdk=sdk
      model.claude_code_exec_effort=medium
      model.claude_code_exec_max_thinking_tokens=16384
      model.codex_trace_to_teacher=false
    )
  else
    echo "unknown backend: $backend" >&2
    exit 1
  fi

  local cmd_file="$RUN_ROOT/commands/${run_id}.sh"
  local log_file="$RUN_ROOT/logs/${run_id}.log"
  local out_root="$RUN_ROOT/$run_id"

  local -a cmd=(
    "$PYTHON" -u scripts/train.py
    --config "$config"
    --cfg-options
    "${COMMON_CFG[@]}"
    "${backend_cfg[@]}"
    env.split_dir="$split_dir"
    env.skill_init="$skill"
    env.out_root="$out_root"
    "$@"
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

  if [[ "$tmux_started" -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "$run_id" "bash '$cmd_file'; code=\$?; echo EXIT:\$code; sleep 3600"
    tmux_started=1
  else
    tmux new-window -t "$SESSION" -n "$run_id" "bash '$cmd_file'; code=\$?; echo EXIT:\$code; sleep 3600"
  fi
  echo "$run_id"
}

SEARCHQA_SKILL="docs/harness_source_skills/searchqa_best_skill.md"
LIVEMATH_SKILL="docs/harness_source_skills/livemathematicianbench_best_skill.md"

SEARCHQA_CFG="configs/searchqa/default.yaml"
LIVEMATH_CFG="configs/livemathematicianbench/default.yaml"

SEARCHQA_SPLIT="data/searchqa/splits"
LIVEMATH_SPLIT="data/livemathbench/splits"

for backend in codex claude; do
  prefix="HARNESS-Codex"
  [[ "$backend" == "claude" ]] && prefix="HARNESS-Claude"

  launch_run "${prefix}-SearchQA-sched-constant" "$backend" "$SEARCHQA_CFG" "$SEARCHQA_SKILL" "$SEARCHQA_SPLIT" \
    optimizer.learning_rate=4 optimizer.min_learning_rate=2 optimizer.lr_scheduler=constant
  launch_run "${prefix}-SearchQA-sched-linear" "$backend" "$SEARCHQA_CFG" "$SEARCHQA_SKILL" "$SEARCHQA_SPLIT" \
    optimizer.learning_rate=4 optimizer.min_learning_rate=2 optimizer.lr_scheduler=linear
  launch_run "${prefix}-SearchQA-batch-full" "$backend" "$SEARCHQA_CFG" "$SEARCHQA_SKILL" "$SEARCHQA_SPLIT" \
    train.batch_size=400 optimizer.learning_rate=4 optimizer.min_learning_rate=2 optimizer.lr_scheduler=cosine
  launch_run "${prefix}-SearchQA-lr8" "$backend" "$SEARCHQA_CFG" "$SEARCHQA_SKILL" "$SEARCHQA_SPLIT" \
    optimizer.learning_rate=8 optimizer.min_learning_rate=1 optimizer.lr_scheduler=constant
  launch_run "${prefix}-LiveMath-lr8" "$backend" "$LIVEMATH_CFG" "$LIVEMATH_SKILL" "$LIVEMATH_SPLIT" \
    optimizer.learning_rate=8 optimizer.min_learning_rate=1 optimizer.lr_scheduler=constant
  launch_run "${prefix}-LiveMath-lr16" "$backend" "$LIVEMATH_CFG" "$LIVEMATH_SKILL" "$LIVEMATH_SPLIT" \
    optimizer.learning_rate=16 optimizer.min_learning_rate=1 optimizer.lr_scheduler=constant
done

echo "RUN_ROOT=$RUN_ROOT"
echo "SESSION=$SESSION"

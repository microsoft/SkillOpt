#!/usr/bin/env bash
set -euo pipefail

REPO="/home/azureuser/workspace-gzy/SkillReflection"
PYTHON="/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python"

cd "$REPO"

if [[ -f ".secrets/teacher_oaidr9.env" ]]; then
  # shellcheck disable=SC1091
  source ".secrets/teacher_oaidr9.env"
else
  echo "missing .secrets/teacher_oaidr9.env" >&2
  exit 1
fi

stamp="$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${1:-outputs/lrctrl_fullrewrite_neutral3_sq_lm_extra_workers2_timeout1020_${stamp}_run}"
SESSION="${2:-lrctrl_fr_sq_lm_extra_${stamp}}"
START_INDEX="${3:-4}"
N_REPEATS="${4:-3}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/commands"

COMMON_CFG=(
  model.teacher_backend=openai_chat
  model.student_backend=openai_chat
  model.teacher=gpt-5.5
  model.student=gpt-5.5
  model.teacher_azure_openai_endpoint="${TEACHER_AZURE_OPENAI_ENDPOINT}"
  model.teacher_azure_openai_api_version="${TEACHER_AZURE_OPENAI_API_VERSION}"
  model.teacher_azure_openai_auth_mode="${TEACHER_AZURE_OPENAI_AUTH_MODE}"
  model.teacher_azure_openai_managed_identity_client_id="${TEACHER_AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID}"
  model.teacher_azure_openai_ad_scope="${TEACHER_AZURE_OPENAI_AD_SCOPE}"
  model.student_azure_openai_endpoint="${STUDENT_AZURE_OPENAI_ENDPOINT:-$TEACHER_AZURE_OPENAI_ENDPOINT}"
  model.student_azure_openai_api_version="${STUDENT_AZURE_OPENAI_API_VERSION:-$TEACHER_AZURE_OPENAI_API_VERSION}"
  model.student_azure_openai_auth_mode="${STUDENT_AZURE_OPENAI_AUTH_MODE:-$TEACHER_AZURE_OPENAI_AUTH_MODE}"
  model.student_azure_openai_managed_identity_client_id="${STUDENT_AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID:-$TEACHER_AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID}"
  model.student_azure_openai_ad_scope="${STUDENT_AZURE_OPENAI_AD_SCOPE:-$TEACHER_AZURE_OPENAI_AD_SCOPE}"
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
  optimizer.learning_rate=4
  optimizer.min_learning_rate=2
  optimizer.lr_scheduler=cosine
  optimizer.lr_control_mode=none
  optimizer.skill_update_mode=full_rewrite_minibatch
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
  local config="$2"
  shift 2

  local cmd_file="$RUN_ROOT/commands/${run_id}.sh"
  local log_file="$RUN_ROOT/logs/${run_id}.log"
  local out_root="$RUN_ROOT/$run_id"

  local -a cmd=(
    "$PYTHON" -u scripts/train.py
    --config "$config"
    --cfg-options
    "${COMMON_CFG[@]}"
    env.out_root="$out_root"
    "$@"
  )

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd '$REPO'"
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

for ((i=START_INDEX; i<START_INDEX+N_REPEATS; i++)); do
  launch_run "LRCTRL-searchqa-full-rewrite-neutral3-extra-r${i}" "configs/searchqa/default.yaml" \
    env.split_dir=data/ablation_splits/searchqa/2-1-7_seed42

  launch_run "LRCTRL-livemathematicianbench-full-rewrite-neutral3-extra-r${i}" "configs/livemathematicianbench/default.yaml" \
    env.split_dir=data/ablation_splits/livemathematicianbench/2-1-7_seed42
done

echo "RUN_ROOT=$RUN_ROOT"
echo "SESSION=$SESSION"
echo "START_INDEX=$START_INDEX"
echo "N_REPEATS=$N_REPEATS"

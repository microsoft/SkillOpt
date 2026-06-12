#!/usr/bin/env bash
# Install the SkillOpt-Sleep Codex integration as a user-level Codex skill.
# Idempotent; prints what it does.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
AGENTS_SKILLS="${HOME}/.agents/skills"
LEGACY_PROMPT="$CODEX_HOME/prompts/sleep.md"

echo "[install] repo: $REPO_ROOT"

# 1) user-level skill
mkdir -p "$AGENTS_SKILLS/skillopt-sleep"
cp "$REPO_ROOT/plugins/codex/skills/skillopt-sleep/SKILL.md" "$AGENTS_SKILLS/skillopt-sleep/SKILL.md"
echo "[install] skill           -> $AGENTS_SKILLS/skillopt-sleep/SKILL.md"

# 2) retire the old custom prompt entrypoint from previous installs
if [ -f "$LEGACY_PROMPT" ]; then
  backup="${LEGACY_PROMPT}.skillopt-legacy.bak"
  if [ -e "$backup" ]; then
    backup="${LEGACY_PROMPT}.skillopt-legacy.$(date +%Y%m%d%H%M%S).bak"
  fi
  mv "$LEGACY_PROMPT" "$backup"
  echo "[install] legacy prompt  -> $backup"
fi

# 3) record the repo location so the runner is found from anywhere
echo "[install] add to your shell profile:"
echo "    export SKILLOPT_SLEEP_REPO=\"$REPO_ROOT\""

# 4) optional: append an AGENTS.md hint (only if the user opts in)
cat <<EOF

[install] Optional — add this to ~/.codex/AGENTS.md so Codex always knows the tool:

  ## SkillOpt-Sleep
  Use the skillopt-sleep skill when I ask to run a sleep/dream/offline
  self-improvement cycle. The runner is:
  \`bash "$REPO_ROOT/plugins/run-sleep.sh" status --project "\$(pwd)"\`.

Done. Try asking Codex:
  Use the skillopt-sleep skill to run status for this project.
EOF

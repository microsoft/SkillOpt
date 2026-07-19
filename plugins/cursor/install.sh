#!/usr/bin/env bash
# Install the SkillOpt-Sleep Cursor integration as a local Cursor plugin.
# Idempotent; prints what it does.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CURSOR_HOME="${CURSOR_HOME:-$HOME/.cursor}"
PLUGIN_DIR="$CURSOR_HOME/plugins/local/skillopt-sleep"
SOURCE_DIR="$REPO_ROOT/plugins/cursor"

echo "[install] repo: $REPO_ROOT"

mkdir -p "$PLUGIN_DIR/.cursor-plugin" "$PLUGIN_DIR/commands" "$PLUGIN_DIR/skills/skillopt-sleep"
cp "$SOURCE_DIR/.cursor-plugin/plugin.json" "$PLUGIN_DIR/.cursor-plugin/plugin.json"
cp "$SOURCE_DIR/commands/skillopt-sleep.md" "$PLUGIN_DIR/commands/skillopt-sleep.md"
cp "$SOURCE_DIR/skills/skillopt-sleep/SKILL.md" "$PLUGIN_DIR/skills/skillopt-sleep/SKILL.md"
cp "$SOURCE_DIR/README.md" "$PLUGIN_DIR/README.md"
cp "$SOURCE_DIR/LICENSE" "$PLUGIN_DIR/LICENSE"

echo "[install] plugin manifest -> $PLUGIN_DIR/.cursor-plugin/plugin.json"
echo "[install] command         -> $PLUGIN_DIR/commands/skillopt-sleep.md"
echo "[install] skill           -> $PLUGIN_DIR/skills/skillopt-sleep/SKILL.md"

cat <<EOF

[install] Quit and reopen Cursor. The plugin should appear in Settings >
Plugins under Installed.

For source-checkout runs, add this to your shell profile:
    export SKILLOPT_SLEEP_REPO="$REPO_ROOT"

Alternatively, install a SkillOpt release that includes Cursor support so the
\`skillopt-sleep\` command is on PATH.

Done. Try in Cursor:
  /skillopt-sleep status
EOF

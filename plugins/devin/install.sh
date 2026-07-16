#!/usr/bin/env bash
# Install the SkillOpt-Sleep Devin integration into a project.
# Copies the SessionEnd hook and rules snippet into .devin/, and prints
# the MCP server registration command. Idempotent.
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PLUGIN_DIR/../.." && pwd)"
PROJECT="${1:-$(pwd)}"

echo "[install] repo: $REPO_ROOT"
echo "[install] project: $PROJECT"

DEVIN_DIR="$PROJECT/.devin"
mkdir -p "$DEVIN_DIR/hooks" "$DEVIN_DIR/rules"

# 1) SessionEnd hook (on by default — provides activity signal for nightly harvest)
#    Merge into existing hooks.v1.json instead of overwriting, so we don't
#    destroy other project hooks.
HOOK_SCRIPT_SRC="$PLUGIN_DIR/hooks/on-session-end.sh"
HOOK_SCRIPT_DST="$DEVIN_DIR/hooks/on-session-end.sh"
cp "$HOOK_SCRIPT_SRC" "$HOOK_SCRIPT_DST"
chmod +x "$HOOK_SCRIPT_DST"
echo "[install] hook script       -> $HOOK_SCRIPT_DST"

HOOK_CONFIG="$DEVIN_DIR/hooks.v1.json"
if [ -f "$HOOK_CONFIG" ]; then
  # Merge our SessionEnd hook into the existing config (jq deep-merge)
  if command -v jq >/dev/null 2>&1; then
    jq -s '.[0] * .[1]' "$HOOK_CONFIG" "$PLUGIN_DIR/hooks/hooks.v1.json" > "$HOOK_CONFIG.tmp"
    mv "$HOOK_CONFIG.tmp" "$HOOK_CONFIG"
    echo "[install] session-end hook  -> $HOOK_CONFIG (merged)"
  else
    echo "[install] WARNING: jq not found; cannot merge into existing $HOOK_CONFIG"
    echo "[install] Merge this SessionEnd hook manually or install jq:"
    echo "[install]   cat $PLUGIN_DIR/hooks/hooks.v1.json"
    echo "[install] Skipping hook config to avoid overwriting existing hooks."
  fi
else
  cp "$PLUGIN_DIR/hooks/hooks.v1.json" "$HOOK_CONFIG"
  echo "[install] session-end hook  -> $HOOK_CONFIG"
fi

# 2) Rules snippet so Devin proactively offers the tools
cp "$PLUGIN_DIR/devin-rules.snippet.md" "$DEVIN_DIR/rules/skillopt-sleep.md"
echo "[install] rules snippet     -> $DEVIN_DIR/rules/skillopt-sleep.md"

# 3) Print the MCP server registration command
cat <<EOF

[install] Register the MCP server (run once per machine):

  devin mcp add skillopt-sleep \\
    --env "SKILLOPT_DEVIN_CLAUDE_HOME=\$HOME/.skillopt-sleep-devin" \\
    -- python3 $PLUGIN_DIR/mcp_server.py

Done. Try asking Devin:
  Run the sleep cycle for this project.
EOF

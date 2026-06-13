#!/usr/bin/env bash
# Install the SkillOpt-Sleep OpenCode integration into ~/.config/opencode.
# Idempotent; preserves existing config when it cannot safely merge JSON.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENCODE_HOME="${OPENCODE_HOME:-$HOME/.config/opencode}"
COMMANDS_DIR="$OPENCODE_HOME/commands"
SKILLS_DIR="$OPENCODE_HOME/skills/skillopt-sleep"
CONFIG_PATH="$OPENCODE_HOME/opencode.json"

echo "[install] repo: $REPO_ROOT"
echo "[install] opencode home: $OPENCODE_HOME"

mkdir -p "$COMMANDS_DIR" "$SKILLS_DIR"

python3 - "$REPO_ROOT" "$COMMANDS_DIR/sleep.md" <<'PY'
from __future__ import annotations

import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
template = (repo / "plugins" / "opencode" / "commands" / "sleep.md.in").read_text()
out.write_text(template.replace("@REPO_ROOT@", str(repo)), encoding="utf-8")
PY
echo "[install] /sleep command -> $COMMANDS_DIR/sleep.md"

cp "$REPO_ROOT/plugins/opencode/skills/skillopt-sleep/SKILL.md" "$SKILLS_DIR/SKILL.md"
echo "[install] skill          -> $SKILLS_DIR/SKILL.md"

python3 - "$CONFIG_PATH" "$REPO_ROOT" <<'PY'
from __future__ import annotations

import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
repo = pathlib.Path(sys.argv[2])
server = {
    "type": "local",
    "command": ["python3", str(repo / "plugins" / "copilot" / "mcp_server.py")],
    "environment": {"SKILLOPT_SLEEP_REPO": str(repo)},
    "enabled": True,
    "timeout": 3600000,
}

created = False
try:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"$schema": "https://opencode.ai/config.json"}
        created = True
except Exception as exc:
    print(f"[install] skipped MCP config merge: {path} is not plain JSON ({exc})")
    print("[install] add this snippet manually:")
    print(json.dumps({"mcp": {"skillopt-sleep": server}}, indent=2))
    raise SystemExit(0)

if not isinstance(data, dict):
    print(f"[install] skipped MCP config merge: {path} does not contain a JSON object")
    raise SystemExit(0)

data.setdefault("$schema", "https://opencode.ai/config.json")
mcp = data.setdefault("mcp", {})
if not isinstance(mcp, dict):
    print(f"[install] skipped MCP config merge: existing 'mcp' is not an object in {path}")
    raise SystemExit(0)
mcp["skillopt-sleep"] = server

path.parent.mkdir(parents=True, exist_ok=True)
if path.exists() and not created:
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[install] backup config -> {backup}")
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"[install] MCP server    -> {path}")
PY

cat <<EOF

[install] Optional shell profile line:
    export SKILLOPT_SLEEP_REPO="$REPO_ROOT"

Done. Restart OpenCode, then try:
    /sleep status
EOF

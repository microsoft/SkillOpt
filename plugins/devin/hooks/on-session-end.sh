#!/usr/bin/env bash
# SkillOpt-Sleep SessionEnd hook for Devin (best-effort, NON-BLOCKING).
#
# This does NOT run the optimizer. It only appends a tiny marker so the next
# nightly cycle knows there is fresh activity to harvest, and (optionally)
# nudges the user once that a sleep cycle is available. It must never fail the
# session or spend API budget.
#
# Install: copy this file and hooks.v1.json into your project's .devin/hooks/
# directory. Devin CLI reads .devin/hooks.v1.json automatically.
set -uo pipefail

STATE_DIR="${HOME}/.skillopt-sleep"
mkdir -p "$STATE_DIR" 2>/dev/null || exit 0

# Record that a session just ended (cheap; used for "is there new data?").
printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DEVIN_PROJECT_DIR:-${PWD}}" \
  >> "$STATE_DIR/session-end.log" 2>/dev/null || true

exit 0

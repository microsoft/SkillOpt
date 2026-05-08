#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: monitor_harness_claude18.sh RUN_ROOT}"

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "===== $ts CLAUDE18 ====="
  uptime | sed 's/^/uptime=/'
  active="$(pgrep -af "scripts/train.py.*${ROOT}" | grep -v pgrep | wc -l || true)"
  claude_child="$(pgrep -af 'claude.*--output-format stream-json' | grep -v pgrep | wc -l || true)"
  echo "active_train_total=$active"
  echo "active_codex_train=0"
  echo "active_claude_train=$active"
  echo "claude_child=$claude_child"

  for d in "$ROOT"/HARNESS-Claude-*; do
    [[ -d "$d" ]] || continue
    rid="$(basename "$d")"
    read -r base best < <(python3 - "$d" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
s = d / "summary.json"
if not s.exists():
    print("pending pending")
    raise SystemExit
try:
    obj = json.loads(s.read_text())
except Exception:
    print("pending pending")
    raise SystemExit
base = obj.get("baseline_test_hard", obj.get("base_test", "pending"))
best = obj.get("test_hard", obj.get("best_test", "pending"))
def fmt(x):
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)
print(fmt(base), fmt(best))
PY
)
    scan_files="$(mktemp)"
    find "$d" \
      \( -path '*/codex_exec' -o -path '*/codex_multi' \) -prune -o \
      -maxdepth 6 -type f \
      \( -name 'claude_trace_summary.txt' -o -name 'codex_trace_summary.txt' -o -name '*.log' -o -name 'summary.json' \) \
      -print > "$scan_files" 2>/dev/null || true
    auth="$({ xargs -r rg -l 'Not logged in|authentication_failed' < "$scan_files" 2>/dev/null || true; } | wc -l | tr -d ' ')"
    e429="$({ xargs -r rg -l 'Too Many Requests|RateLimitError|Error code: 429|api_error_status.: 429|rate_limit|too_many_requests' < "$scan_files" 2>/dev/null || true; } | wc -l | tr -d ' ')"
    e401="$({ xargs -r rg -l '401 Unauthorized|Error code: 401|HTTP 401|AuthenticationTypeDisabled|PermissionDeniedError' < "$scan_files" 2>/dev/null || true; } | wc -l | tr -d ' ')"
    timeout="$({ xargs -r rg -l 'TimeoutError|Task timed out|timed out after|subprocess.TimeoutExpired|timeout_exceeded' < "$scan_files" 2>/dev/null || true; } | wc -l | tr -d ' ')"
    teacher="$({ xargs -r rg -l 'APITimeoutError|APIConnectionError|AuthenticationError|Azure OpenAI Responses API is enabled only|teacher.*error' < "$scan_files" 2>/dev/null || true; } | wc -l | tr -d ' ')"
    results="$(find "$d" -maxdepth 5 -path '*/results.jsonl' -type f -print0 2>/dev/null | xargs -0 -r wc -l | awk 'END{print $1+0}')"
    empty="$({ xargs -r rg -l 'final response chars: 0|\"final_response\"\\s*:\\s*\"\"|\"result\"\\s*:\\s*\"\"' < "$scan_files" 2>/dev/null || true; } | wc -l | tr -d ' ')"
    rm -f "$scan_files"
    errors=$((auth + e429 + e401 + timeout + teacher))
    echo "$rid Base=$base Best=$best Errors=$errors auth=$auth 429=$e429 401=$e401 timeout=$timeout teacher=$teacher Results=$results Empty=$empty"
  done | sort
  echo
  sleep 60
done

#!/bin/bash
# Smoke test for Superpowers adapter integration.
# Run this manually (not in CI) to verify the adapter works with the real harness.
#
# Prerequisites:
# - Claude Code installed, ANTHROPIC_API_KEY set (or SKILLOPT_HOST_AUTH=1 for a
#   trusted candidate on your own machine)
# - Same model/settings/pinned SHA for baseline and candidate runs
#
# Usage:
#   ./scripts/smoke_superpowers.sh [candidate_skill_path]
#
# Output goes to smoke_results/ (gitignored). Results embed raw agent output and
# local paths: do NOT commit them. Sanitized excerpts are written alongside each
# run for pasting into a PR description or attaching as an artifact.

set -euo pipefail

SKILL="${1:-}"
SCENARIO="${SKILLOPT_SCENARIO:-test-passes-verify}"
SHA="${SKILLOPT_SHA:-d884ae04edebef577e82ff7c4e143debd0bbec99}"
OUTDIR="smoke_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "Smoke test: Superpowers adapter"
echo "Output:   $OUTDIR (gitignored - do not commit)"
echo "Scenario: $SCENARIO"
echo "SHA:      $SHA"
echo ""

run_scenario() {
    local name="$1"
    local candidate="${2:-}"
    local outfile="$OUTDIR/${name}.json"

    echo "=== $name ==="

    local args=(
        --skill verification-before-completion
        --scenario "$SCENARIO"
        --sha "$SHA"
        --json
    )
    if [[ -n "$candidate" ]]; then
        args+=(--candidate "$candidate")
    fi

    # No || true - fail if runner errors
    python -m skillopt_sleep.adapters.superpowers "${args[@]}" > "$outfile"

    # Sanitized summary for sharing: no raw output, no host paths, no secrets
    python - "$outfile" "$OUTDIR/${name}.summary.txt" <<'PY'
import json, os, re, sys
data = json.load(open(sys.argv[1]))
home = os.path.expanduser("~")
def clean(t):
    t = t.replace(home, "~")
    return re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-REDACTED", t)
lines = [
    f"skill={data['skill']} version={data['version']} pinned_sha={data['pinned_sha']}",
    f"candidate_hash={data['candidate_hash'] or '(baseline)'}",
    f"score={data['score']:.2f} passed={data['passed']} failed={data['failed']}",
]
for s in data["scenarios"]:
    lines.append(f"\n[{s['id']}] passed={s['passed']} error={s.get('error') or 'none'}")
    lines.append(f"  evidence: {json.dumps(s.get('evidence', {}))}")
    for c in s["checks"]:
        lines.append(f"  {'PASS' if c['passed'] else 'FAIL'} {c['description']}")
    lines.append("  output excerpt:")
    for ln in clean(s.get("output", ""))[:800].splitlines():
        lines.append(f"    {ln}")
text = "\n".join(lines)
open(sys.argv[2], "w").write(text + "\n")
print(text)
PY
}

run_scenario "baseline"

if [[ -n "$SKILL" ]]; then
    run_scenario "candidate" "$SKILL"
fi

echo ""
echo "Results in $OUTDIR (gitignored)."
echo "Share the *.summary.txt excerpts in the PR; do not commit the raw JSON."

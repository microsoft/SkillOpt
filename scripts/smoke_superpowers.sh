#!/bin/bash
# Smoke test for Superpowers adapter integration.
# Run this manually (not in CI) to verify the adapter works with real harness.
#
# Prerequisites:
# - Harness installed and authenticated
# - Same model/settings for baseline and candidate runs
#
# Usage:
#   SKILLOPT_UNSAFE=1 ./scripts/smoke_superpowers.sh [candidate_skill_path]
#
# Output: writes results + raw output to smoke_results/ for PR evidence.
# Fails on any runner error (no silent swallowing).

set -euo pipefail

SKILL="${1:-}"
OUTDIR="smoke_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "Smoke test: Superpowers adapter"
echo "Output: $OUTDIR"
echo "SKILLOPT_UNSAFE=${SKILLOPT_UNSAFE:-0}"
echo ""

run_scenario() {
    local name="$1"
    local candidate="${2:-}"
    local outfile="$OUTDIR/${name}.json"

    echo "=== $name ==="

    local args=(
        --skill verification-before-completion
        --scenario test-passes-verify
        --json
    )
    if [[ -n "$candidate" ]]; then
        args+=(--candidate "$candidate")
    fi

    # No || true - fail if runner errors
    python -m skillopt_sleep.adapters.superpowers "${args[@]}" > "$outfile"

    # Extract and preserve raw output
    python -c "
import json, sys
data = json.load(open('$outfile'))
for s in data.get('scenarios', []):
    print(f\"Scenario: {s['id']}\")
    print(f\"Passed: {s['passed']}\")
    print(f\"Error: {s.get('error', 'none')}\")
    # Raw output preserved in JSON, print preview
    out = s.get('output', '')
    if out:
        print(f\"Output preview ({len(out)} chars):\")
        print(out[:500])
    print()
"
}

# Baseline run (stock skill)
run_scenario "baseline"

# Candidate run if provided
if [[ -n "$SKILL" ]]; then
    run_scenario "candidate" "$SKILL"
fi

echo "Results saved to $OUTDIR"
echo "Include these files in your PR as evidence of smoke test."

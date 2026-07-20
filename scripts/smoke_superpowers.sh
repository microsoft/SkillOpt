#!/bin/bash
# Smoke test for Superpowers adapter integration.
# Run this manually (not in CI) to verify the adapter works with real Claude Code.
#
# Prerequisites:
# - Claude Code installed and authenticated
# - Same model/settings for baseline and candidate runs
#
# Usage:
#   ./scripts/smoke_superpowers.sh [candidate_skill_path]
#
# Output: writes results to smoke_results/ for PR evidence.

set -euo pipefail

SKILL="${1:-}"
OUTDIR="smoke_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "Smoke test: Superpowers adapter"
echo "Output: $OUTDIR"
echo ""

# Run baseline (no candidate overlay)
echo "=== Baseline run (stock skill) ==="
python -m skillopt_sleep.adapters.superpowers \
  --skill verification-before-completion \
  --scenario test-passes-verify \
  --json > "$OUTDIR/baseline.json" 2>&1 || true

# Run with candidate if provided
if [[ -n "$SKILL" ]]; then
  echo "=== Candidate run ($SKILL) ==="
  python -m skillopt_sleep.adapters.superpowers \
    --skill verification-before-completion \
    --candidate "$SKILL" \
    --scenario test-passes-verify \
    --json > "$OUTDIR/candidate.json" 2>&1 || true
fi

echo ""
echo "Results saved to $OUTDIR"
echo "Include these in your PR as evidence of smoke test."

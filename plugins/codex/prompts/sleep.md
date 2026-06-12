# /sleep — SkillOpt-Sleep for Codex
#
# Custom prompt: copy this file to ~/.codex/prompts/sleep.md and invoke with
# `/sleep` in the Codex CLI. ($ARGUMENTS is the text after /sleep.)

Run the SkillOpt-Sleep offline self-evolution cycle. Action: $ARGUMENTS
(empty → "status").

Use the bundled runner via shell. If $ARGUMENTS is empty, use `status`. If the
arguments do not already include `--source`, add `--source codex` so Codex
Desktop sessions are harvested from `~/.codex/archived_sessions`.

    bash "${SKILLOPT_SLEEP_REPO:?set SKILLOPT_SLEEP_REPO to the repo root}/plugins/run-sleep.sh" <action-and-args> --project "$(pwd)" --source codex

Then:
- For `run`/`dry-run`: read the staged `report.md` and show the held-out
  baseline → candidate score and the proposed edits. `run` only stages a
  proposal; nothing live changes until `adopt`.
- For `adopt`: confirm which files were updated and that a backup was written.
- Never edit the user's AGENTS.md / skills yourself; only `adopt` does that.

Default backend is `mock` (no API spend). Add `--backend codex` for real
improvement on the user's Codex budget.

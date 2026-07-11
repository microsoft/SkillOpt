---
description: Run the SkillOpt-Sleep cycle with the handoff backend — no API subprocess; this session answers the engine's model calls via prompt/answer files, in isolated fresh-context subagents
argument-hint: "[run | dry-run] [--preferences \"...\"] (default: run)"
allowed-tools: Bash, Read, Write, Task
---

# /skillopt-sleep-handoff — session-executed sleep cycle

You are driving **SkillOpt-Sleep in handoff mode**: the Python engine runs
every deterministic stage (harvest → mine → replay scoring → gate → stage)
and outsources each model call (attempt / judge / reflect) to YOU via
prompt files. No `claude -p` subprocess, no API key — the model work runs
on this session's budget, but each prompt MUST be answered in a fresh,
isolated context so the validation gate stays honest.

## Requested action: $ARGUMENTS

(If `$ARGUMENTS` is empty, treat it as `run`.)

## The loop

Repeat until the engine exits 0 (done) — at most 8 rounds:

1. **Run the engine** via the bundled runner:

   ```bash
   "${CLAUDE_PLUGIN_ROOT}/scripts/sleep.sh" <action> --backend handoff --project "$(pwd)" --scope invoked
   ```

   - exit 0 → the night is complete; go to "Finish" below.
   - exit 3 → pending model calls; continue with step 2.
   - anything else → stop and show the user the error output.

2. **Read the batch**: `Read` `.skillopt-sleep-handoff/pending.json` in the
   project. Each entry has `id`, `prompt`, `max_tokens`, `answer_file`.

3. **Answer each prompt in ISOLATION** — this is the integrity rule:
   - For each entry, launch a subagent (Task tool) whose ENTIRE input is
     the `prompt` text verbatim. Add nothing: no summary of this session,
     no mention of SkillOpt, no other prompts from the batch.
   - Take the subagent's reply and `Write` the raw answer text (no
     commentary, no code fences) to the entry's `answer_file`.
   - NEVER answer from this session's own context — you have seen the
     mined tasks and their references, so inline answers would contaminate
     the held-out gate and fake the improvement score.

4. **Re-run the same engine command** — it resumes from the answers
   directory and either finishes or stages the next batch.

## Finish

- `Read` the `report.md` in the staging dir the engine printed and show
  the user: held-out baseline → candidate score, the gate decision, the
  proposed edits, and where the proposal is staged.
- Tell the user nothing live changed; offer `/skillopt-sleep adopt`.
- The engine archives `.skillopt-sleep-handoff/` on a completed real run;
  do not delete it yourself.

## Safety reminders

- **Never** edit `CLAUDE.md` or `SKILL.md` yourself — only `adopt` does
  that, with a backup.
- Mined tasks are pinned to `.skillopt-sleep-handoff/tasks.json` on round
  one, so sessions created while answering prompts cannot shift the task
  set. Do not edit that file.
- If a batch looks like it contains secrets or content the user would not
  want re-processed, stop and ask before answering.

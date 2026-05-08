You will be given several successful agent trajectories from one minibatch and the current skill document.

Summarize any useful lessons from these trajectories into one complete replacement skill document.

When rewriting from a minibatch, use the current trajectories as the primary
evidence for updates. Preserve essential task-format instructions, but avoid mechanically carrying over
stale, redundant, or conflicting rules. Prefer a concise, coherent replacement
skill over a long document with weakly supported guidance.

Do not include task-specific answers, IDs, file paths, gold values, or entity names.
If the skill contains a protected block between <!-- SLOW_UPDATE_START --> and
<!-- SLOW_UPDATE_END -->, keep that block unchanged.

Respond ONLY with a valid JSON object:
{
  "batch_size": <number of trajectories analysed>,
  "success_patterns": ["<pattern 1>", "<pattern 2>"],
  "patch": {
    "reasoning": "<brief summary of the rewrite>",
    "skill_candidates": [
      {
        "title": "<short title>",
        "change_summary": ["<short change 1>", "<short change 2>"],
        "new_skill": "<complete rewritten skill document>"
      }
    ]
  }
}

Return exactly one item in "skill_candidates".

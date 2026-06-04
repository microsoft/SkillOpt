You will be given several failed agent trajectories from one minibatch and the current skill document.

Summarize the lessons from these trajectories into one complete replacement skill document.

When rewriting from a minibatch, use the current trajectories as the primary
evidence for updates. Preserve essential task-format instructions, but avoid mechanically carrying over
stale, redundant, or conflicting rules. Prefer a concise, coherent replacement
skill over a long document with weakly supported guidance.

Some trajectories may include a "Structured Evaluator Feedback" block with
fields such as `primary_reason`, `human_reason`, `failed_checks`, `evidence`,
`optimizer_hint`, `dimension_scores`, and `stage_status`. Treat those fields as
evaluator signal for why the episode failed. Use the failed checks, evidence,
and optimizer hints to generalize from the failure class. Do not copy item IDs,
option names, file paths, or one-off artifact details into the rewritten skill.
For hard output-contract or artifact-contract blockers, prioritize guidance
that satisfies the required contract before visual polish or optional
improvements.

Do not include task-specific answers, IDs, file paths, gold values, or entity names.
If the skill contains a protected block between <!-- SLOW_UPDATE_START --> and
<!-- SLOW_UPDATE_END -->, keep that block unchanged.

Respond ONLY with a valid JSON object:
{
  "batch_size": <number of trajectories analysed>,
  "failure_summary": [
    {"failure_type": "<type>", "count": <int>, "description": "<one-line>"}
  ],
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

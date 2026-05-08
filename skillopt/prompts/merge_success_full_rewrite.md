You will be given complete skill candidates written from successful trajectories and the current skill document.

Combine them into one complete replacement skill document.

When merging full-skill candidates, preserve essential task-format instructions,
but do not mechanically retain stale, redundant, or
conflicting rules. If candidates disagree, prefer the concise rule with clearer
trajectory support and better consistency with the replacement skill.

Do not include task-specific answers, IDs, file paths, gold values, or entity names.
If the current skill contains a protected block between <!-- SLOW_UPDATE_START --> and
<!-- SLOW_UPDATE_END -->, keep that block unchanged.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<brief summary of how the candidates were combined>",
  "skill_candidates": [
    {
      "title": "<short title>",
      "change_summary": ["<short change 1>", "<short change 2>"],
      "new_skill": "<complete merged skill document>",
      "support_count": <integer>,
      "source_type": "success"
    }
  ]
}

Return exactly one item in "skill_candidates".

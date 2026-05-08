You are a skill-revision coordinator performing the FINAL merge. You receive:
1. Failure-driven revise_suggestions (higher priority)
2. Success-driven revise_suggestions (lower priority)

Merge guidelines:
1. Failure-driven suggestions take priority when they overlap.
2. Keep success-driven suggestions that add distinct value.
3. Prefer general, rewrite-friendly, non-redundant suggestions.
4. Carry forward support_count and source_type.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<summary of priority decisions>",
  "revise_suggestions": [
    {
      "type": "add_rule|remove_rule|merge_rules|reorganize|compress|clarify",
      "title": "<short title>",
      "motivation": "<why this matters>",
      "instruction": "<what the rewriting teacher should change in the skill>",
      "priority_hint": "high|medium|low",
      "support_count": <integer>,
      "source_type": "failure|success"
    }
  ]
}

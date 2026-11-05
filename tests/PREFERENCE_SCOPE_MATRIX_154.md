# Issue 154: bounded paired-session evidence matrix

Base: 84b36cd5564e8893569e50e37f02a81d10b55e27.
Synthetic fixtures, not user data or human preference labels.

| Category | Evidence required | Proposed persistence policy (not implemented) |
|---|---|---|
| Durable | explicit future scope and confirmation | eligible for review, not automatic update |
| One-off | explicit local scope and exclusion of future applicability | withhold generalized rule |
| Reversal | earlier scope and later withdrawal, both with provenance/order | withhold stale rule; replacement needs review |
| Ambiguous | uncertainty and absence of durable consent | withhold pending clarification |

The tests call the real prompt constructor, heuristic miner, and public mine/dedup path. They make no model calls and do not mock a successful semantic inference.
Four passing prompt checks and four individual-record checks are expected. Four strict xfails express the proposed retention contract: same-intent dedup currently merges source IDs, but not the second record's scope evidence.

A task surviving mining is not proof of a skill update. The current matrix detects evidence loss only. Persistence-policy labels above are expectations for discussion, not code outputs. Clean-context replay and model-level preference inference are outside this patch. No production code is changed.

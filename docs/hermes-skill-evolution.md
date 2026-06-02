# Hermes Skill Evolution Engine Prototype

## Status

Prototype. The current implementation is intentionally conservative: it runs offline static checks, preserves rejected candidates, requires human review, and never writes to a live skill unless promotion is explicitly approved.

## Goal

Turn SkillOpt output into a governed Hermes skill evolution loop:

```text
SkillOpt candidate → staging → eval → gate → human review → promote
```

This prevents optimizer output from directly mutating `~/.hermes/skills` while still making skill improvement repeatable, auditable, and testable.

## Components

- `skillopt/evolution/pipeline.py`
  - Local governance implementation.
  - Does not call external models.
  - Writes staged artifacts, eval reports, gate decisions, review requests, and promotion records.
- `scripts/hermes_skill_evolve.py`
  - CLI wrapper for the governance flow.
- `tests/test_evolution_pipeline.py`
  - Deterministic tests for staging, eval, gate, review, promotion, qmd grounding, regression fixtures, and dry-run promotion.

## Flow

### 1. Stage

```bash
python scripts/hermes_skill_evolve.py stage \
  --registry .hermes-skill-evolution \
  --skill-name qmd \
  --candidate outputs/best_skill.md \
  --base ~/.hermes/skills/research/qmd/SKILL.md
```

Purpose: copy a generated candidate into a staging registry and record manifest metadata, source path, candidate hash, and optional base skill hash.

### 2. Eval

```bash
python scripts/hermes_skill_evolve.py eval \
  --registry .hermes-skill-evolution \
  --skill-name qmd \
  --candidate-id <candidate-id> \
  --regression-fixture tests/fixtures/qmd_skill_regression.json
```

Purpose: run deterministic local evaluation. Current score dimensions:

- `correctness`: frontmatter, description, and minimum body length.
- `tool_grounding`: command/code blocks must explain purpose near the snippet.
- `qmd_grounding`: candidate cites `qmd://...` sources and/or tells the agent to use qmd lookup commands.
- `regression`: fixture-defined required terms are present in candidate text.
- `safety`: rejects secret-like strings and dangerous command patterns.
- `user_style`: rewards Traditional Chinese / local purpose explanation style.
- `operational_quality`: candidate includes verification guidance.
- `cost_latency`: candidate acknowledges cost or latency.

### 3. Gate

```bash
python scripts/hermes_skill_evolve.py gate \
  --registry .hermes-skill-evolution \
  --skill-name qmd \
  --candidate-id <candidate-id> \
  --min-score 0.8
```

Purpose: reject candidates with any score below the threshold. Rejected candidates are copied into `registry/rejected/<skill>/<candidate-id>/` for audit instead of being deleted.

### 4. Human review

```bash
python scripts/hermes_skill_evolve.py review \
  --registry .hermes-skill-evolution \
  --skill-name qmd \
  --candidate-id <candidate-id>
```

Purpose: create `human_review.md` containing candidate metadata, scores, diff path, and a review checklist.

Review checklist requires the human to:

- read `candidate.md`
- read `candidate.diff` when a base skill exists
- confirm qmd grounding references are relevant and non-sensitive
- confirm regression fixture failures are absent or intentionally accepted
- confirm no secrets or dangerous commands are present
- confirm promotion dry-run output before writing live skill

### 5. Promotion dry-run

```bash
python scripts/hermes_skill_evolve.py promote \
  --registry .hermes-skill-evolution \
  --skill-name qmd \
  --candidate-id <candidate-id> \
  --live-skill-path ~/.hermes/skills/research/qmd/SKILL.md \
  --approved-by chiahsin \
  --dry-run
```

Purpose: show what would be written without modifying the live skill. This should be run before real promotion.

### 6. Promote

```bash
python scripts/hermes_skill_evolve.py promote \
  --registry .hermes-skill-evolution \
  --skill-name qmd \
  --candidate-id <candidate-id> \
  --live-skill-path ~/.hermes/skills/research/qmd/SKILL.md \
  --approved-by chiahsin
```

Purpose: after explicit approval, archive the previous live skill and copy the candidate to the live path.

## qmd / MCP integration pattern

qmd is used as the local retrieval layer for skill evolution:

- Use BM25 (`qmd search`) for fast smoke tests and exact lookup.
- Use `qmd://...` references in candidate skills when they rely on local wiki or vault knowledge.
- Use qmd MCP tools from Hermes after configuring:

```bash
hermes mcp add qmd --command qmd --args mcp
```

Purpose: register qmd's MCP server so Hermes can query local collections through native MCP tools.

Important: vector/hybrid search requires embeddings and may download large local models. Do not run `qmd embed`, `qmd vsearch`, or `qmd query` in unattended optimization jobs unless model downloads and compute cost are explicitly accepted.

## Safety rules

- Never optimize directly against the live `~/.hermes/skills` directory.
- Use staging registry or temporary Hermes profiles for candidates.
- Treat optimizer output as untrusted until it passes eval, gate, and review.
- Preserve rejected candidates for audit.
- Promotion requires an explicit approver name.
- Prefer promotion dry-run before every real write.
- Do not include API keys, tokens, internal customer data, or banking records in fixtures or staged artifacts.

## Next extensions

- Replace the simple required-term regression fixture with task/rubric execution against realistic Hermes workflows.
- Add qmd MCP live checks that verify every `qmd://...` reference resolves.
- Add held-out evaluation comparing candidate and baseline responses.
- Replace file-copy promotion with Hermes `skill_manage` integration for live profile edits.
- Add a cron-friendly report mode that proposes candidates but never promotes them automatically.

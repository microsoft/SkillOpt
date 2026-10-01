# Community rule exchange

SkillOpt-Sleep can exchange distilled skill rules without exchanging session
transcripts. Imported rules are untrusted candidates: each rule must be reviewed
and must strictly improve the importer's own task set without regressing any task
before it is staged for adoption.

This is a local workflow. It does not upload manifests, operate a registry, or
automatically adopt imported rules.

An empirical gate is not a sandbox or a security review. Imported text changes
agent behavior during validation, so inspect the rule and license before passing
`--reviewed`, and use the same provider and execution-boundary precautions as an
ordinary Sleep replay.

## Export accepted rules

Export reads an accepted staging report and includes only accepted `skill/add`
edits. Memory edits, tasks, responses, session identifiers, and evidence logs are
not copied.

```bash
skillopt-sleep export-rules \
  --staging .skillopt-sleep/staging/20260722-031700 \
  --output community-rules.json \
  --category coding \
  --license MIT
```

The observed effect belongs to the complete candidate set evaluated by that
staging run, not to an individual rule in isolation. The exporter labels this as
`"scope": "candidate_set"`. It also refuses secret-shaped rule or rationale text,
but that is not an anonymization guarantee. Inspect the output before publishing
it to a public Git repository.

## Review and import

The first invocation prints the complete manifest and exits without running it:

```bash
skillopt-sleep import-rules --manifest community-rules.json
```

After reviewing every rule and the manifest license, run the local gate with a
reviewed task file and an explicit target skill:

```bash
skillopt-sleep import-rules \
  --manifest community-rules.json \
  --reviewed \
  --project /path/to/project \
  --target-skill-path .agents/skills/my-skill/SKILL.md \
  --tasks-file reviewed-tasks.json \
  --backend codex
```

The importer evaluates rules sequentially using only the task file's `val`
split; `train` and `test` remain outside the import decision. For each rule it
replays the same validation tasks against the current skill and the candidate
skill. A rule enters the staged proposal only when its configured gate score
strictly increases and no task score decreases. Publisher-reported effects are
informational and never participate in the local decision.

Review an accepted `proposed_SKILL.md`, then use the existing explicit adoption
step:

```bash
skillopt-sleep adopt --legacy
```

## Manifest v1

```json
{
  "schema": "skillopt.community-rules",
  "schema_version": 1,
  "license": "MIT",
  "rules": [
    {
      "id": "rule-63e143cbd8ab167d",
      "category": "coding",
      "rule": "Run focused tests before reporting a change as complete.",
      "rationale": "Prevents false completion reports.",
      "observed_effect": {
        "metric": "local_gate_score",
        "baseline": 0.5,
        "candidate": 0.75,
        "delta": 0.25,
        "sample_size": 20,
        "scope": "candidate_set"
      }
    }
  ]
}
```

The v1 parser rejects unknown fields. This keeps the public artifact bounded to
the rule, a provenance-free rationale, aggregate effect metadata, category, and
license; raw trajectories have no field in the format.

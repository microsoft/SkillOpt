# Gitmoot MVP Workflow

This guide shows the local boundary between Gitmoot and Gitmoot-SkillOpt.
Gitmoot owns agent-template state, eval artifacts, feedback collection, and the
human promote/reject decision. Gitmoot-SkillOpt consumes an exported training
package, proposes a candidate package, and writes optimizer artifacts for
Gitmoot to verify on import.

The MVP uses saved baseline and candidate artifacts from a completed Gitmoot
eval run. Live pairwise evaluation is intentionally out of scope for this repo
flow and is tracked separately in Gitmoot issue #77.

## 1. Export Training Data From Gitmoot

Run this from the Gitmoot repo or any project where `gitmoot` is configured:

```bash
gitmoot skillopt export --run <run-id> --output training.json
```

The exported package includes:

- the current agent-template snapshot
- eval run metadata
- eval items split by metadata when available
- artifact refs for saved baseline, candidate, preview, or diff outputs
- feedback events gathered from Markdown or GitHub collectors
- evaluator config for deterministic fixture or judge-backed scoring

Gitmoot artifact blobs stay in Gitmoot's content-addressed store. Pass that blob
root to Gitmoot-SkillOpt as `--artifact-root`.

## 2. Optimize With Gitmoot-SkillOpt

Install this repo in editable mode, then run the optimizer:

```bash
pip install -e .

gitmoot-skillopt optimize \
  --training-package training.json \
  --artifact-root ~/.gitmoot/evals/blobs \
  --out-root outputs/run-1 \
  --candidate-output outputs/run-1/candidate.json
```

For a no-network contract smoke, use the tracked fixture:

```bash
gitmoot-skillopt optimize \
  --training-package examples/gitmoot/mvp-fixture/training.json \
  --artifact-root examples/gitmoot/mvp-fixture/blobs \
  --out-root /tmp/gitmoot-skillopt-smoke \
  --candidate-output /tmp/gitmoot-skillopt-smoke/candidate.json \
  --dry-run
```

The dry run validates package loading, Gitmoot artifact resolution, candidate
package emission, and generated artifact manifests without starting the trainer
or calling external model APIs.

## 3. Import The Candidate Into Gitmoot

Import the optimizer output back into Gitmoot:

```bash
gitmoot skillopt import \
  --file outputs/run-1/candidate.json \
  --artifact-dir outputs/run-1/artifacts
```

Gitmoot verifies every manifest entry before mutating candidate state:

- artifact paths must stay inside `--artifact-dir`
- absolute paths and path traversal are rejected
- SHA256 hashes and optional sizes must match
- blobs are stored in Gitmoot content-addressed storage
- candidate metadata is registered as a pending template version

Candidate packages without artifact manifest entries still import through the
legacy path.

## 4. Inspect The Pending Candidate

List and inspect pending versions:

```bash
gitmoot skillopt candidate list --template <template-id>
gitmoot skillopt candidate show <version-id>
```

`candidate show` displays the candidate state, eval report, preference summary,
content diff, and imported `diff_artifact_id` when present.

## 5. Collect Human Feedback

Use Markdown packets for local review:

```bash
gitmoot skillopt feedback markdown export \
  --run <run-id> \
  --output .gitmoot/evals/<run-id>

gitmoot skillopt feedback markdown import \
  --packet .gitmoot/evals/<run-id> \
  --reviewer <name>
```

Use GitHub when collaborators should review in issues or PR comments:

```bash
gitmoot skillopt feedback github publish \
  --run <run-id> \
  --repo owner/reviews

gitmoot skillopt feedback github sync \
  --run <run-id> \
  --repo owner/reviews \
  --issue <number>
```

Humans choose baseline or candidate per review item and may include reasoning.
Feedback becomes optimizer context for later runs; it does not promote a
candidate automatically.

## 6. Promote Or Reject

After inspection and feedback, explicitly decide:

```bash
gitmoot skillopt candidate promote <version-id>
gitmoot skillopt candidate reject <version-id> --reason "Needs narrower scope"
```

Promotion makes the candidate the current Gitmoot agent-template version.
Rejection records an auditable reason and leaves the current template unchanged.

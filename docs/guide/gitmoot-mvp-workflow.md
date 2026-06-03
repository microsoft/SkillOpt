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

The direct dry run validates package loading, Gitmoot artifact resolution,
candidate package emission, and generated artifact manifests without starting
the trainer or calling external model APIs. In Gitmoot train mode, run dry-runs
only on a disposable or reset train session because the CLI imports the dry-run
candidate and advances the iteration to candidate review.

Real optimizer runs are different from dry runs. Before training starts,
Gitmoot-SkillOpt runs preflight canaries for the optimizer, target, and
evaluator paths. The optimizer and target must return exact canary text. The
evaluator must return structured JSON with hard and soft scores; specialized
evaluators can require extra fields. A failed canary blocks the run before any
candidate is trained.

For landing-page work, use the Vue preview flow in Gitmoot and request the
landing-page evaluator:

```bash
gitmoot-skillopt optimize \
  --training-package training.json \
  --artifact-root ~/.gitmoot/evals/blobs \
  --out-root outputs/landing-page-train \
  --candidate-output outputs/landing-page-train/candidate.json \
  --optimizer-backend openai_chat \
  --target-backend codex_exec \
  --evaluator-id landing_page_v1 \
  --evaluator-backend openai_chat \
  --optimizer-model gpt-5.5 \
  --target-model gpt-5.5 \
  --evaluator-model gpt-5.5 \
  --skill-update-mode full_rewrite_minibatch \
  --num-epochs 1 \
  --batch-size 2 \
  --gate-metric mixed
```

When Gitmoot exports a manual-review package whose items identify a Vue preview
or landing page, Gitmoot-SkillOpt can infer `landing_page_v1`. Passing
`--evaluator-id landing_page_v1` is still recommended for operational retry
commands because it makes the intended scoring contract explicit.

To test the Gitmoot import boundary too, build or point to a current `gitmoot`
binary and run:

```bash
.venv/bin/python scripts/gitmoot_contract_smoke.py --gitmoot-bin /path/to/gitmoot
```

The script uses a temporary Gitmoot home, installs
`examples/gitmoot/mvp-fixture/template.md` as a local template, imports the
generated candidate with `--artifact-dir`, verifies the pending candidate shows
`candidate-diff`, and rejects it with a test reason.

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

In train-mode review packets, ranked feedback may include `quality`,
`continue_mode`, and `promote`:

- `quality` describes the reviewer confidence in the option set. `poor` means
  the optimizer should treat the set as weak, `acceptable` means there is a
  usable direction, and `strong` means the winner is strong enough to refine.
- `continue_mode` is a phase hint such as `explore`, `refine`, `distill`, or
  `validate`.
- `promote` is a human decision hint. It does not bypass Gitmoot's explicit
  promote/reject command.

These feedback fields are optimizer context, not evaluator scores. Candidate
acceptance uses evaluator result artifacts produced during SkillOpt rollout.
If target execution fails, evaluator execution fails, or hard/soft scores are
missing, the gate records an unscored blocker instead of treating the result as
a numeric zero.

## 6. Promote Or Reject

After inspection and feedback, explicitly decide:

```bash
gitmoot skillopt candidate promote <version-id>
gitmoot skillopt candidate reject <version-id> --reason "Needs narrower scope"
```

Promotion makes the candidate the current Gitmoot agent-template version.
Rejection records an auditable reason and leaves the current template unchanged.

## Preview And Evaluator Selection

Use rendered previews when reviewers or evaluators must inspect a built
artifact:

| Output type | Preview expectation | Evaluator |
| --- | --- | --- |
| Vue landing page or UI | Gitmoot `vue-vite` preview, usually required | `landing_page_v1` |
| Markdown/text answer or X/social post copy | no preview required unless the review surface needs one | package evaluator or default LLM judge |
| LaTeX/PDF, image, notebook, Storybook | future preview adapter | future evaluator-specific contract |

The current implemented specialized preview is Vue/Vite. LaTeX, PDF, social
post, image, and notebook previews should be added later as separate adapters
instead of changing the optimizer loop.

## Troubleshooting

- Wrong review repo: Gitmoot owns review publication. Use explicit
  `gh`/`gitmoot` repo flags and check `review.expected_repo` in the train status
  before publishing or syncing feedback.
- Missing preview links: create the train session with `--preview-repo`, register
  that checkout in Gitmoot, and use Vue-compatible generated outputs.
- Missing evaluator: pass `--evaluator-id landing_page_v1` for landing-page
  retries or configure a supported package evaluator for text/fixture flows.
- Invalid evaluator JSON: fix the evaluator backend/model or evaluator prompt
  and rerun. Invalid JSON blocks the gate.
- Dry-run versus real optimizer: `--dry-run` is a no-network package smoke. In
  Gitmoot train mode it should use a disposable or reset session; the real
  optimizer pass omits `--dry-run` and requires working optimizer, target, and
  evaluator credentials.

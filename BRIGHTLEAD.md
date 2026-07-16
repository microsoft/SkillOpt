# BrightLead SkillOpt Pilot

Installed locally for internal research/ops use at `tools/skillopt/`.

Source:

- Repo: `https://github.com/microsoft/SkillOpt`
- Local commit at install: `50fed29`
- Package version: `0.2.0`
- License: MIT
- Install style: editable install inside `tools/skillopt/.venv`

Use the BrightLead wrapper by default:

```sh
tools/skillopt/bin/brightlead-skillopt-sleep --help
```

Run the local pilot smoke check before any manual SkillOpt trial:

```sh
tools/skillopt/bin/brightlead-skillopt-smoke
```

The smoke check verifies the wrapper help path and the sanitized LOL-010 regression fixture. It does not schedule, adopt, push, contact GitHub, or touch WordPress.

Record a local preflight report before a manual pilot dry run:

```sh
tools/skillopt/bin/brightlead-skillopt-preflight
```

The preflight writes branch, commit, Git status, smoke output, and GitHub Actions guardrail evidence under `runtime/skillopt-preflight-*`. It is local-only and does not dispatch automation, push branches, adopt skills, contact external services, or touch WordPress.

Run a reviewed local mock pilot dry run after preflight passes:

```sh
tools/skillopt/bin/brightlead-skillopt-pilot-dry-run
```

The pilot dry run creates a sanitized temporary project, uses a reviewed task file with `--backend mock`, and writes report-only evidence under `runtime/skillopt-pilot-dry-run-*`. It confirms SkillOpt can propose a missing rule without adopting edits, staging live changes, harvesting transcripts, pushing branches, dispatching automation, contacting external services, or touching WordPress.

Run the stronger repeatable BrightLead QA fixture for next-version QA:

```sh
tools/skillopt/bin/brightlead-skillopt-pilot-dry-run --fixture brightlead-qa
```

This uses three sanitized BrightLead-style QA tasks and a local mock backend. It should improve score from `0.4` to `1.0`, keep `adopted: False`, keep `staging_dir` empty, and write `reviewed-brightlead-qa-tasks.json`, `dry-run.json`, and `brightlead-skillopt-pilot-dry-run.md`. Use this as the repeatable source-side check before building a human QA packet.

Run the known-gap fixture when QA needs proof that SkillOpt can stage a useful new rule from a reviewed local task file:

```sh
tools/skillopt/bin/brightlead-skillopt-pilot-dry-run --fixture brightlead-known-gap
```

This uses three sanitized BrightLead-style QA tasks that require SI units. It should improve score from `0.5` to `1.0`, propose `Always include SI units in numeric answers.`, keep `adopted: False`, keep `staging_dir` empty, and write `reviewed-brightlead-known-gap-tasks.json`, `dry-run.json`, and `brightlead-skillopt-pilot-dry-run.md`.

Run the broader BrightLead regression suite when checking guardrail coverage before another SkillOpt batch:

```sh
tools/skillopt/bin/brightlead-skillopt-regression-suite
```

The suite runs reviewed mock fixtures for answer formatting, SI units, no-live-write closeouts, source-citation hygiene, and draft-first publication recovery. It writes `manifest.json` and `brightlead-skillopt-regression-suite.md`, and every fixture must stay report-only with `adopted: False`, an empty `staging_dir`, reviewed task files, and `n_sessions: 0`.

Create a redacted task-file draft from reviewed local snippets before any real-backend replay:

```sh
tools/skillopt/bin/brightlead-skillopt-sanitize-tasks snippets.jsonl runtime/skillopt-reviewed-tasks.json --target-skill-path skills/qa-output/SKILL.md
```

The sanitizer accepts JSON or JSONL snippets, redacts obvious secrets, emails, URLs, home paths, long IDs, Discord-like IDs, and WordPress-style item IDs, then writes `skillopt_sleep.tasks.v1`. It defaults to `reviewed: false`; pass `--mark-reviewed` only after a human has inspected the output and filled any missing expected outcome/rubric.


Run the disposable staged-adoption test only after a separate adoption-path approval:

```sh
tools/skillopt/bin/brightlead-skillopt-disposable-adoption-test
```

This creates a generated disposable project, stages a mock SkillOpt proposal, auto-adopts it into that disposable `SKILL.md`, verifies a backup was made, and confirms the staging path stayed inside the disposable project. It does not target live BrightLead skills, harvest transcripts, push branches, contact external services, or touch WordPress.

Create a single human-review packet after the dry-run command is stable:

```sh
tools/skillopt/bin/brightlead-skillopt-review-bundle
```

The review bundle reruns the local preflight and reviewed mock dry run, verifies both passed, writes a manifest with evidence paths, and produces a checklist for reviewing the proposed rule before any future adoption batch. It is local-only and does not push, open PRs, dispatch automation, contact external services, harvest private transcripts, adopt edits, stage live changes, or touch WordPress.

Pilot guardrails:

- Internal use only until BrightLead has validated its output quality.
- Use `--backend mock` for first dry runs and workflow checks.
- Do not use `--auto-adopt` on BrightLead skills unless David has approved that exact adoption batch.
- Review generated/staged skill changes as normal code/content changes before adopting.
- Do not feed client-private content, credentials, lead data, or production incident logs into non-local model backends without explicit approval.
- Keep staged outputs and pilot evidence under `runtime/skillopt-*` where practical.

Suggested first BrightLead pilot:

```sh
tools/skillopt/bin/brightlead-skillopt-sleep dry-run \
  --project /home/hugh-brightlead/.openclaw/sandboxes/agent-brightlead-os-coder-6467ab13 \
  --backend mock \
  --max-sessions 1 \
  --max-tasks 1
```

Operational recommendation:

- Treat SkillOpt as a skill-improvement assistant, not an automatic skill maintainer.
- Prefer reviewed task files and explicit target skill paths for early pilots.
- Keep all generated edits behind held-out validation plus human review.

LOL-010 regression fixture:

```sh
cd tools/skillopt
PYTHONNOUSERSITE=1 python3 -m unittest tests.test_brightlead_lol010_regression
```

This fixture is BrightLead-local but sanitized enough to serve as an upstream issue/PR reference. It verifies two behaviors: SkillOpt can learn a missing draft-then-publish recovery rule from a pre-rule skill, and it no-ops once that recovery rule already exists. Keep it reviewed/manual only; it does not adopt skills or touch WordPress.


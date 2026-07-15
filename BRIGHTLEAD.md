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


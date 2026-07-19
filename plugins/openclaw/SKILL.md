---
name: skillopt-sleep
description: Reference-only OpenClaw adaptation of SkillOpt-Sleep. Use it to study or port the contributed DeepSeek wrapper, not as a ready-to-run installation.
---

# SkillOpt-Sleep OpenClaw reference adaptation

This directory is a contributed **reference**, not a supported, plug-and-play
OpenClaw integration. It illustrates one way to connect the shared
`skillopt_sleep` cycle to a custom DeepSeek Chat Completions backend and a set of
environment-specific task fixtures.

Do not run or schedule the files unchanged. Several scripts and the sample
configuration preserve assumptions from the contributor's original machine,
and parts of the wrapper have not yet been ported to the current shared-engine
interfaces. Start with the directory's [README.md](README.md), which is the
authoritative status and adaptation guide.

## What is included

- `skillopt_sleep_openclaw.py` — a contributed DeepSeek backend prototype. It
  also contains an Ollama embedding helper, but that helper is not wired into
  the current shared sleep cycle.
- `run_sleep.py` — a custom cycle wrapper with environment-specific paths and a
  backend-registration shim.
- `slash_sleep.py` — an experimental command helper written for an older
  staging-manifest shape.
- `run_sleep_cron.sh` — a machine-specific category runner, not a portable cron
  installer.
- `config.json` — a sample configuration, not a set of guaranteed or enforced
  runtime limits.
- `tests/*.json` — example task fixtures from one environment, not a universal
  OpenClaw benchmark.

## Known porting gaps

Before treating this as an integration, a maintainer must at least:

1. Replace every absolute workspace, repository, state, skill, log, and task
   path with explicit user configuration.
2. Update the custom backend factory to the current `get_backend` call contract,
   including the project directory, and update its backend methods and edit
   records to the current protocol.
3. Replace the experimental adoption logic with the current staging manifest
   and `skillopt_sleep.staging.adopt` behavior. Current staging artifacts use
   `proposed_SKILL.md` / `proposed_CLAUDE.md`, `manifest.json`, and report files;
   they do not expose the old `manifest.proposed_skill` field.
4. Decide how real OpenClaw transcripts are converted into a supported session
   format. Pointing `claude_home` at an arbitrary agent directory does not by
   itself make its files Claude Code-compatible JSONL.
5. Build scheduling around the adapted wrapper. The shared scheduler launches
   the shared CLI; it does not automatically preserve this custom backend or
   its category task-file flow.
6. Add isolated end-to-end tests for dry-run, accepted/rejected gates, staging,
   adoption and backup, credential failure, and scheduled execution.

Until those gaps are resolved, use the supported shared
`python -m skillopt_sleep` CLI with `--backend mock` to test SkillOpt-Sleep itself,
and treat this directory only as source material for a future OpenClaw port.

## Shared-engine features are not wrapper features

At this revision the supported shared CLI backends are `mock`, `claude`,
`codex`, `copilot`, `cursor`, `handoff`, and `azure_openai`; the
[plugin integration reference](../README.md#supported-cli-surface) is the
authoritative list. The shared engine can consolidate a selected skill and
project `CLAUDE.md` memory (controlled by `evolve_skill` and `evolve_memory`),
and its `schedule` / `unschedule` actions manage shared-engine cron entries.
Those capabilities do **not** make the custom OpenClaw wrapper portable: the
shared scheduler will not invoke the prototype backend or its category
fixtures. Use the shared documentation for those features, not this reference
SKILL.

## Data and credential boundary

The prototype DeepSeek backend sends task, skill, memory, response, rubric, and
reflection content to its configured Chat Completions endpoint. Its source also
contains a helper that can send text to an Ollama service if a future port wires
that helper into the cycle. Neither path should be assumed to remove every
secret or private detail.

Before any port is tested with real data:

- use isolated, synthetic or explicitly reviewed task files;
- replace sample business names, personal references, URLs, and machine paths;
- load credentials through the operator's secret-management mechanism;
- verify TLS and retention policy for every remote endpoint; and
- inspect all staged artifacts before adoption.

The bundled fixtures are examples only. Their scores and any old cost estimates
do not establish effectiveness, safety, or a stable nightly price for another
OpenClaw deployment.

## Further information

- [OpenClaw README](README.md) — current reference status and adaptation checklist
- [plugin integration reference](../README.md) — supported shared-engine CLI
  surface and data boundary
- [SkillOpt-Sleep documentation](../../docs/sleep/README.md) — concepts,
  results, and limitations

Contributions that turn this reference into a portable integration should add
tests and update all three documents together.

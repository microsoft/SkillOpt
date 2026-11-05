# OpenClaw reference adaptation for SkillOpt-Sleep

This directory is a contributed reference for connecting
[SkillOpt-Sleep](https://github.com/microsoft/SkillOpt) to
[OpenClaw](https://github.com/openclaw/openclaw) with a custom DeepSeek/Ollama
backend.

> **Reference status.** This is not one of the shared, plug-and-play
> `skillopt_sleep` wrappers. Several scripts and the sample config contain
> environment-specific absolute paths and assumptions from the original setup,
> and the contributed wrapper has unresolved Python 3.10 syntax and backend
> factory-signature gaps. The current checkout is not directly runnable; treat
> it as porting source material, not an installation.

## Included components

| File | Purpose |
|---|---|
| `run_sleep.py` | custom cycle entry point |
| `skillopt_sleep_openclaw.py` | DeepSeek Chat Completions backend plus local Ollama embeddings |
| `run_sleep_cron.sh` | category-oriented cron wrapper |
| `slash_sleep.py` | experimental `/sleep` command helper |
| `config.json` | example engine configuration |
| `SKILL.md` | OpenClaw skill manifest |
| `tests/*.json` | example task sets for research, DevOps, and wiki workflows |

The adaptation imports the shared engine but registers its own backend and
maintains its own wrapper behavior. Changes to the shared CLI documentation do
not automatically make every option available through these custom scripts.

## Intended cycle

```text
harvest supported session data or load a task file
  → replay with the current skill
  → propose bounded edits
  → validate the candidate on held-out tasks
  → stage a proposal for operator review
```

The intended safety boundary is manual adoption: review the generated report and
staged files before changing a live skill.

## Adapt before use

1. Clone SkillOpt into a location you control:

   ```bash
   git clone https://github.com/microsoft/SkillOpt.git
   cd SkillOpt/plugins/openclaw
   ```

2. Inspect and replace the sample absolute paths in `run_sleep.py`,
   `slash_sleep.py`, `run_sleep_cron.sh`, and `config.json`. Confirm the engine
   checkout, OpenClaw workspace, state directory, skill directory, and task-file
   paths all point to isolated test locations.

3. Review `config.json`. In particular, do not assume that values such as
   `replay_mode` are enforced by this custom wrapper merely because they appear
   in the example config. `max_tokens_per_night` is applied by the shared cycle,
   but it is measured with the wrapper backend's rough `tokens_used()` estimate
   (a quarter of the attempt prompt length plus 200 per attempt, 600 per rubric
   judge call, and a third of the reflect prompt plus 1500 per reflect call), so
   it is not a billing limit.

4. Supply credentials through your normal secret-management mechanism. Do not
   commit a DeepSeek key or place it in a world-readable file.

5. Resolve every known porting gap listed in [`SKILL.md`](SKILL.md), add
   isolated tests for your adapted backend, and verify that `--help` imports
   cleanly on Python 3.10+. Only then start with a dry run and one reviewed
   task file. The target command should be shaped like:

   ```bash
   cd /path/to/SkillOpt/plugins/openclaw
   python3 run_sleep.py --config /path/to/reviewed-config.json \
     --tasks tests/research-cron-tasks.json --dry-run
   ```

6. Inspect the report, paths, network destinations, and proposed edits before
   considering a non-dry run or scheduling.

## Data boundary

The custom `openclaw-deepseek` backend sends task, skill, response, rubric, and
reflection content to the configured DeepSeek endpoint. Its embedding helper can
send truncated text to the configured local Ollama service. Do not assume these
outbound prompts have been fully redacted; inspect transcript/task inputs and the
provider's retention policy before using real data.

Use HTTPS for a remote DeepSeek-compatible endpoint. Keep any plaintext Ollama
endpoint on a trusted loopback interface. For a network-free engine smoke test,
use the shared SkillOpt-Sleep CLI with `--backend mock` rather than assuming this
custom wrapper is isolated.

## Scheduling

`run_sleep_cron.sh` and the scheduling helpers are examples, not portable
installers. Adapt their paths, create log directories, verify their environment,
and run the exact command manually before adding a cron entry. Scheduled runs
must preserve the same manual-adoption and credential boundaries as interactive
runs.

## Validation scope

The bundled JSON files are example held-out task sets, not a universal OpenClaw
benchmark. Provider cost and quality depend on the selected model, task content,
number of calls, and pricing at run time; this reference does not promise a fixed
nightly cost. Validate the adapted workflow in an isolated workspace before
using it on live skills.

For the supported shared-engine CLI and its current flags, see the
[integration reference](../README.md#supported-cli-surface). For measured
SkillOpt-Sleep results and limitations, see
[`docs/sleep/RESULTS.md`](../../docs/sleep/RESULTS.md).

## License

MIT, consistent with SkillOpt core.

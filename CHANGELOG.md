# Changelog

All notable changes to SkillOpt are documented here. This project adheres to
[Semantic Versioning](https://semver.org/) and the format is based on
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Handoff backend** (`--backend handoff`) for SkillOpt-Sleep — runs the
  sleep cycle with no model subprocess or API key: the engine writes each
  pending model call to `PROMPTS.md`/`pending.json` (exit code 3) and the
  user's own agent session answers into `answers/<id>.md`; re-running the
  same command resumes statelessly from the answers (typically 3–6 rounds
  per night). Mined tasks are pinned per night so answering sessions cannot
  shift the task set. Ships a `/skillopt-sleep-handoff` Claude Code command
  that automates the loop with fresh-context subagents to protect the
  held-out gate.

## [0.2.0] — 2026-07-02

The headline of this release is **SkillOpt-Sleep**: a nightly offline
self-evolution engine that harvests a coding agent's real session
transcripts, mines recurring tasks, replays them offline, and consolidates
short-term experience into long-term memory and skills — all behind the same
held-out validation gate that keeps SkillOpt training honest. It ships as a
decoupled top-level package (`skillopt_sleep/`, zero dependency on the
research code) and as the new `skillopt-sleep` CLI.

### Added
- **SkillOpt-Sleep engine** — nightly offline self-evolution cycle
  (harvest → mine → replay → consolidate) behind a validation gate, exposed
  as the `skillopt-sleep` console script and `python -m skillopt_sleep`.
  - Multi-objective reward (accuracy / tokens / latency) with user preferences.
  - Multi-rollout contrastive reflection under a token/time budget.
  - Experience replay + controllable dream rollouts (opt-in).
  - Slow-update long-term memory field (runs even with the gate off).
  - 3-way train/val/test split with `gate_mode on|off`.
  - Verifier-discipline validation gate, with a stress-test suite
    (thanks @Tanmay9223, #87).
- **Cross-tool backends & plugin shells** for Claude Code, Codex, Copilot,
  Devin, and OpenClaw:
  - Codex Desktop transcript harvesting, skill-first Codex integration, and a
    reviewed task-file flow (thanks @Kirchberg, #48, #49, #60).
  - GitHub Copilot backend (`CopilotCliBackend`) + research-engine MCP plugin
    (thanks @Dongbumlee, #50).
  - Devin plugin: MCP server + ATIF-v1.7 harvest (thanks @xerxes-y, #88).
  - OpenClaw shell for SkillOpt-Sleep (thanks @Elzlxx, #59).
- **SearchQA** split materialization helper and fail-fast on systemic rollout
  failures, with a `searchqa` install extra (thanks @summerview1997,
  #63, #64, #65).
- WebUI environment loading and backend preflight (thanks @summerview1997, #63).

### Changed
- Decoupled the Sleep engine into a standalone top-level `skillopt_sleep/`
  package with zero dependency on the research code.
- Made `EnvAdapter.reflect` a shared default so reflect kwargs are no longer
  dropped (thanks @imshunsuke, #44).
- English-only pass across the engine, plugins, and docs.

### Fixed
- Windows robustness for the Claude/Codex backends, plus a hardened JSON
  fallback path (thanks @Yif-Yang, #79).
- Reject prose pseudo-JSON wrapped in single quotes/backticks (#82).
- Surface Codex auth/model/version failures instead of silently scoring 0
  (thanks @dmmdea, #92).
- Redact secrets before persisting cycle diagnostics.
- Configure the `qwen_chat`/`minimax` backends so local LLM endpoints work
  (thanks @imrehg, #85).
- Forward the Qwen target timeout and gate `enable_thinking` for vLLM targets
  (thanks @mvanhorn, #40).
- Make `--bare` conditional on `ANTHROPIC_API_KEY` (#68), add a
  `SKILLOPT_SLEEP_PYTHON` override with a lookback-hours first-run fallback
  (#74), and fix ALFWorld gamefile paths relative to `ALFWORLD_DATA`.

### Packaging
- Bump `skillopt`, `skillopt.__version__`, and `skillopt_sleep.__version__`
  to `0.2.0`.
- Restore `skillopt_webui` to the built wheel (it was dropped when the
  `packages.find` include list was made explicit).
- Add the `searchqa` extra and include `json_repair` in the `claude`, `qwen`,
  and `all` extras.

### Acknowledgements 🙏
v0.2.0 landed thanks to our community contributors — thank you!

- @Kirchberg — Codex Desktop harvesting, skill-first Codex integration,
  reviewed task-file flow (#48, #49, #60)
- @Dongbumlee — GitHub Copilot backend + research-engine MCP plugin (#50)
- @summerview1997 — SearchQA materialization, rollout fail-fast, WebUI
  preflight (#63, #64, #65)
- @xerxes-y — Devin plugin: MCP server + ATIF-v1.7 harvest (#88)
- @Elzlxx — OpenClaw shell for SkillOpt-Sleep (#59)
- @imshunsuke — shared `EnvAdapter.reflect` default + docs fixes (#43, #44)
- @mvanhorn — Qwen timeout forwarding + `enable_thinking` gating (#40)
- @dmmdea — surface Codex auth/model/version failures (#92)
- @Tanmay9223 — verifier-discipline stress test (#87)
- @imrehg — `configure_qwen_chat` for local LLM endpoints (#85)
- @samuelgoofus-boop — community contributions

Special thanks to @Yif-Yang for driving the SkillOpt-Sleep engine.

**Full changelog:** https://github.com/microsoft/SkillOpt/compare/v0.1.0...v0.2.0

## [0.1.0] — 2026-06-02

Initial public release: the full training loop (rollout → reflect →
aggregate → select → update → evaluate), multi-backend support
(OpenAI / Azure / Claude / Qwen / MiniMax), six built-in benchmarks, and the
WebUI dashboard.

[0.2.0]: https://github.com/microsoft/SkillOpt/releases/tag/v0.2.0
[0.1.0]: https://github.com/microsoft/SkillOpt/releases/tag/v0.1.0

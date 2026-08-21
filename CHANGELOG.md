# Changelog

All notable changes to SkillOpt are documented here. This project adheres to
[Semantic Versioning](https://semver.org/) and the format is based on
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **SkillOpt-Sleep opt-in `llm_dream`**: paraphrase-only dream variants from
  the optimizer model, with deterministic template fallback on parse or
  fidelity failure. Default template dreams stay byte-identical; generated
  variants are train-only (thanks @bogdanbaciu21).
- **SkillOpt-Sleep multi-skill fan-out and reviewed subset adoption**: each
  hinted skill is consolidated from its own pinned live baseline, staged as an
  independent proposal with per-skill gate evidence, and promoted only through
  an explicit `--skill`, `--all-skills`, or managed `--legacy` choice. Adoption
  uses a versioned fail-closed manifest, provenance hashes, canonical target
  pins, immutable backups/receipts, cross-night locking, durable publication,
  and restart-recoverable transactions. Fan-out discovers native project skill
  roots, supports repeatable `--skill-root` overrides, and is enabled by
  `multi_skill_fanout` (`multi_skill_report` remains an alias). MCP adapters
  enforce typed arguments and preserve engine failures without copying outside
  the transaction (thanks @bogdanbaciu21, #212).
- Additive `section_contains` rule-judge operator for literal,
  case-insensitive matching within numbered, bilingual, or annotated ATX
  Markdown headings. The legacy `section_present` behavior is unchanged.
- **OpenCode transcript source** (`--source opencode`) for SkillOpt-Sleep. It
  reads visible user/assistant text and tool names from OpenCode's local SQLite
  history without requiring its CLI, login, or a provider connection.
- **OpenCode CLI backend** (`--backend opencode`) for SkillOpt-Sleep model calls.
  It uses an installed OpenCode CLI with the user's existing login and
  file-based global configuration, and supports plain task replay plus opt-in
  tool-aware replay. Plain calls disable project configuration,
  model-initiated tool invocation, external plugins, and configured MCP servers.
  Tool-aware replay exposes only temporary synthetic tools with randomized names
  and fixed results, verifies which tools OpenCode actually invokes, and never
  retains or replays historical tool arguments or results.
- **GitHub Copilot CLI backend**, in two forms: `copilot_chat` (usable as both
  optimizer and target) and `copilot_exec` (target-only execution harness).
  Because the Copilot CLI carries its own sign-in, `--backend copilot` selects
  `copilot_chat` for both roles and runs a complete train/eval loop with no
  separate provider API key; inference still uses the GitHub Copilot cloud
  service. Chat calls disable all built-in tools, built-in MCP servers, and
  custom instructions, strip inherited `COPILOT_ALLOW_ALL`, and never pass
  `--allow-all-tools`; `copilot_exec` requires an explicit
  `copilot_exec_allow_all_tools` opt-in before granting unattended tool use.
  The CLI reports no token counts, so usage totals are zero for these backends.
- A non-destructive Devin installer and SessionEnd activity marker, preserving
  existing project hooks across repeated installation.
- Per-night SkillOpt-Sleep `evidence.jsonl` chains for reconstructing harvest,
  mining, replay, reflection, and gate decisions, plus a live prompt-template
  registry with user overrides.
- Native SkillOpt-Sleep support for Cursor, including a local plugin command
  and skill, Cursor transcript harvesting, and an optional Cursor Agent CLI
  backend. Cursor tool-aware replay remains disabled pending live permission-
  boundary validation.
- **Cursor Agent research target harness** (`cursor_exec`) for running
  supported benchmark rollouts through an installed, authenticated
  `cursor-agent`, with sandboxed workspaces, structured trace capture, and
  target-only optimizer separation.
- **Handoff backend** (`--backend handoff`) for SkillOpt-Sleep — runs the
  sleep cycle with no model subprocess or API key: the engine writes each
  pending model call to `PROMPTS.md`/`pending.json` (exit code 3) and the
  user's own agent session answers into `answers/<id>.md`; re-running the
  same command resumes statelessly from the answers (typically 3–6 rounds
  per night). Mined tasks are pinned per night so answering sessions cannot
  shift the task set. Ships a `/skillopt-sleep-handoff` Claude Code command
  that automates the loop with fresh-context subagents to protect the
  held-out gate (thanks @dimitarvdenev, #125).
- **Generic OpenAI-compatible research backend** for optimizer and target
  calls, with configurable base URL, API key, model, and timeout (thanks
  @nankingjing, #115).
- **OpenAI-compatible SkillOpt-Sleep endpoint support** for providers such as
  DeepSeek and self-hosted vLLM servers (thanks @Alphaxalchemy, #129; hardened
  in #138).
- End-to-end wiring for the documented reflection `--preferences` option
  (thanks @AKhozya, #131).

### Changed
- Claude Code's Sleep plugin can now use a `pip`/`uv`-installed
  `skillopt-sleep` when no repository checkout is present (thanks
  @ichoosetoaccept, #107).
- Qwen reasoning-model requests now use `max_completion_tokens` and omit
  unsupported temperature parameters (thanks @chirag127, #128).
- Configuration files are read explicitly as UTF-8 (thanks @nankingjing,
  #124).
- `gradient.max_analyst_rounds` is retired: it was parsed and logged but never
  reached reflection. Every way of still supplying it (`--max_analyst_rounds`,
  `--cfg-options`, or a structured or legacy flat config file) is accepted and
  warns rather than being dropped in silence (reported by @xs229 in #213;
  implemented by @wilyan09007 in #219).

### Fixed
- Apply Codex executable and sandbox aliases consistently across inherited
  YAML, CLI overrides, training, and eval-only entry points. Codex CLI calls
  now use explicit sandbox and approval settings instead of the retired
  `--full-auto` flag (thanks @RohithPariki, #220).
- Preserve fractional rollout hard scores instead of coercing them to binary
  values (thanks @zixuanguo786-ctrl, #104).
- Reject duplicate and overlapping IDs while materializing SearchQA manifests
  (thanks @zixuanguo786-ctrl, #105).
- Make JSON-array extraction robust to unmatched braces and keep malformed
  scans linear-time (thanks @zixuanguo786-ctrl, #103; follow-up #136).
- Package Markdown prompt assets in wheels and tolerate Windows temporary-file
  cleanup failures (thanks @nankingjing, #135; follow-up #137).
- Exclude sub-agent transcripts and plugin-generated sessions from Sleep task
  mining (thanks @codeL1985, #99).
- Normalize validation-gate density against the proposed edits and handle
  zero-edit candidates safely (thanks @SparshGarg999, #102).
- Route optimizer-role MiniMax calls through the MiniMax backend (thanks
  @jcforever1, #116).
- Surface Claude CLI spawn failures instead of silently turning them into zero
  scores (thanks @Phoenix0531-sudo, #126).
- Improve Claude CLI behavior on Windows, including `.cmd` resolution and
  long-prompt handling (thanks @codeL1985, #98).
- Preserve the scheduler's established annealing contract while expanding its
  endpoint and sequence coverage (thanks @nankingjing, #123; follow-up #133).

### Security
- Prevent managed-identity credentials from being sent to non-Azure or
  non-HTTPS endpoints, and isolate compatible-provider request extensions
  from native Azure mode in SkillOpt-Sleep (#138, following
  @Alphaxalchemy's #129).

### Tests
- Add focused OpenCode backend coverage and opt-in real-CLI smoke tests for
  tool-aware replay and a seeded plain cycle.
- Strengthen SkillOpt-Sleep verifier-discipline assertions, including recorded
  scores and gate actions (thanks @Tanmay9223, #96).
- Add focused coverage for the validation-gate decision core and edit-budget
  schedulers (thanks @nankingjing, #122, #123).

### Acknowledgements 🙏
Thank you to the contributors behind this unreleased work:
@AKhozya, @Alphaxalchemy, @Phoenix0531-sudo, @SparshGarg999,
@Tanmay9223, @chirag127, @codeL1985, @dimitarvdenev,
@bogdanbaciu21, @ichoosetoaccept, @jcforever1, @nankingjing, @wilyan09007,
@xs229, and @zixuanguo786-ctrl.

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

# Security Considerations for Superpowers Adapter

## Execution Model

The Superpowers adapter runs Claude Code with candidate skills that control agent behavior. A candidate skill is **untrusted input**: it can

- Execute arbitrary shell commands
- Read/write files reachable by the process
- Read environment variables passed to the process
- Make network requests

`--allowedTools` scopes which tools the agent may call. It is **not** an
isolation boundary — `Bash` and `Read` are granted, so anything the process can
reach, the candidate can reach.

## Current Mitigations

1. **No host credential reuse by default.** The scenario `HOME` is empty; host
   `~/.claude/credentials.json` and `settings.json` are never copied or
   symlinked. Reuse is opt-in via `SKILLOPT_HOST_AUTH=1`, which warns.
2. **Fail closed.** With neither `ANTHROPIC_API_KEY` nor `SKILLOPT_HOST_AUTH=1`,
   the scenario errors (`NO_AUTH`) instead of running unauthenticated.
3. **Scrubbed environment.** Only `HOME`, `PATH`, `TERM`, `LANG` and (if set)
   `ANTHROPIC_API_KEY` are passed; the host environment is not inherited.
4. **Isolated project and HOME** per scenario, inside a temp workspace.
5. **OS-level sandbox**, opt-in via `SKILLOPT_SANDBOX=bwrap|docker`.
6. **Execution evidence.** `harness_test_passes` re-runs the tests in the parent
   process after the agent exits — this is unforgeable and is the authoritative
   gate. `pytest_runs` (nonce-tagged invocation log) is tamper-**evident**, not
   tamper-proof: an unsandboxed agent runs as the same OS user with `Bash` and
   can reach the log, so treat the count as corroborating unless running under
   `SKILLOPT_SANDBOX`.

## Known Limitations

- **API key exposure**: `ANTHROPIC_API_KEY`, if set, is visible to the agent
  process. Use a scoped/disposable key, or run under `SKILLOPT_SANDBOX=docker`
  with a key injected per run.
- **`SKILLOPT_HOST_AUTH=1` exposes host credentials** to the candidate. Trusted
  candidates only. Combining it with `SKILLOPT_SANDBOX` is refused (the host
  `~/.claude` is not mounted, so the symlinks would dangle) — use
  `ANTHROPIC_API_KEY` inside the sandbox instead.
- **`SKILLOPT_UNSAFE=1`** disables permission checks entirely. Trusted
  candidates only.
- **No network isolation** in the default (unsandboxed) path.

## Recommendations

### Trusted candidates (your own skill, local machine)
```bash
ANTHROPIC_API_KEY=... python -m skillopt_sleep.adapters.superpowers --candidate my_skill.md
```

### Untrusted or model-generated candidates
```bash
# Linux
SKILLOPT_SANDBOX=bwrap ANTHROPIC_API_KEY=... \
  python -m skillopt_sleep.adapters.superpowers --candidate untrusted.md

# Container
SKILLOPT_SANDBOX=docker SKILLOPT_SANDBOX_IMAGE=skillopt-sandbox ANTHROPIC_API_KEY=... \
  python -m skillopt_sleep.adapters.superpowers --candidate untrusted.md
```

## Follow-up Work

- [ ] Published sandbox image with Claude Code + pytest preinstalled
- [ ] Network egress allowlist (api.anthropic.com only)
- [ ] Per-run scoped API keys

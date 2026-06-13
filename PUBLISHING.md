# Publishing SkillOpt-Sleep — how people install and use it

This is the open-source SkillOpt-Sleep tool: a nightly offline "sleep cycle" for
local coding agents, shipped as plugins for **Claude Code**, **Codex**,
**Copilot**, and **OpenCode**. One engine
([`skillopt_sleep/`](skillopt_sleep)), four thin shells ([`plugins/`](plugins)),
decoupled from the research code.

## How end users install it

### Claude Code

The Claude Code plugin ships a marketplace manifest at
`plugins/claude-code/.claude-plugin/marketplace.json`.

```text
# inside Claude Code:
/plugin marketplace add microsoft/SkillOpt
/plugin install skillopt-sleep
/sleep status
```

(`/plugin marketplace add <owner>/<repo>` reads the marketplace manifest from the
repo; the entry points at `plugins/claude-code`.)

### Codex

```bash
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt
bash plugins/codex/install.sh           # installs /sleep prompt + skill
export SKILLOPT_SLEEP_REPO="$(pwd)"      # so the runner is found anywhere
# then, in Codex:  /sleep status
```

### Copilot

```bash
git clone https://github.com/microsoft/SkillOpt.git
# register the MCP server with your Copilot config (see plugins/copilot/README.md
# and plugins/copilot/mcp-config.example.json), pointing SKILLOPT_SLEEP_REPO at
# the clone. Then ask Copilot to "run the sleep cycle".
```

### OpenCode

```bash
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt
bash plugins/opencode/install.sh          # installs /sleep, skill, and MCP server
# then restart OpenCode and run:  /sleep status
```

Requirements for all four: Python >= 3.10, and the corresponding agent CLI on
PATH. The default backend is `mock` (no API spend); `--backend claude|codex|opencode`
uses the user's own budget.

## Wider distribution (optional, maintainer steps)

1. **GitHub Release.** Tag the milestone so users can pin a version:
   ```bash
   gh release create sleep-v0.1.0 --title "SkillOpt-Sleep v0.1.0" \
      --notes "Nightly offline self-evolution plugins for Claude Code, Codex, Copilot, OpenCode."
   ```

2. **Official Claude Code plugin marketplace.** To appear in the public
   directory, open a PR adding a `marketplace.json` entry to
   [`anthropics/claude-code` / the official marketplace repo], pointing at
   `microsoft/SkillOpt` subdir `plugins/claude-code`. Users could then
   `/plugin install skillopt-sleep@<official-marketplace>`.

3. **PyPI (optional).** `skillopt_sleep` is a standalone package
   (`pyproject.toml` lists it). A `pip install skillopt-sleep` distribution would
   let users run `python -m skillopt_sleep ...` without cloning. Build with
   `python -m build` and publish with `twine`.

4. **README News.** The main [`README.md`](README.md) already announces the
   release and links to [`plugins/`](plugins) and
   [`docs/sleep/FINAL_REPORT.md`](docs/sleep/FINAL_REPORT.md).

## Verifying a release works

```bash
# deterministic, no API key:
python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves
# the unit suite:
python -m unittest tests.test_sleep_engine
# the MCP server (Copilot):
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | SKILLOPT_SLEEP_REPO="$(pwd)" python3 plugins/copilot/mcp_server.py
```

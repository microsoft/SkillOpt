# Installation

## Requirements

- Python ≥ 3.10
- For research training/evaluation, access to at least one configured model
  backend (hosted API, local server, or an installed execution CLI)
- The SkillOpt-Sleep `mock` backend needs no credentials

## Choose an Install

### PyPI

Use PyPI for the Python packages and installed commands:

```bash
python -m pip install skillopt
skillopt-sleep --help
```

This installs `skillopt-train`, `skillopt-eval`, and `skillopt-sleep`. The wheel
does not include the repository's benchmark configs, data materializers,
agent-integration shells/MCP servers, or development tests; use a source
checkout for those files.

!!! important "PyPI versus `main`"
    These docs track the latest `main`. The current PyPI release is `0.2.0`.
    The generic research `openai_compatible` backend, SkillOpt-Sleep handoff,
    Sleep support for non-Azure OpenAI-compatible endpoints, the Sleep
    `--preferences` flag, and Cursor source/backend/plugin support landed after
    that release and require a source install from `main` until the next
    release.

### Source checkout

```bash
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt
python -m pip install -e .
```

Use the source checkout for paper reproduction, built-in benchmark configs,
and contributions.

## Optional Dependencies

Install extras for specific benchmarks or backends:

=== "ALFWorld"

    ```bash
    python -m pip install -e ".[alfworld]"
    ```

=== "Claude agent SDK (optional)"

    ```bash
    python -m pip install -e ".[claude]"
    ```

    This extra does not install the `claude` executable. The research
    `claude_chat` backend launches `claude -p`, so install and authenticate the
    Claude Code CLI separately. The SDK extra is only needed when selecting an
    SDK-backed Claude Code exec path.

=== "Qwen (Local)"

    ```bash
    python -m pip install -e ".[qwen]"
    ```

=== "SearchQA data"

    ```bash
    python -m pip install -e ".[searchqa]"
    ```

=== "WebUI"

    ```bash
    python -m pip install -e ".[webui]"
    ```

=== "Development"

    ```bash
    python -m pip install -e ".[dev]"
    ```

=== "All"

    ```bash
    python -m pip install -e ".[alfworld,claude,qwen,searchqa,webui,docs,dev]"
    ```

## Environment Variables

From a source checkout, copy the template and fill in only the backend you
will use:

```bash
cp .env.example .env
```

SkillOpt does not automatically load `.env`; export it into the current shell
before running commands:

```bash
set -a
source .env
set +a
```

For Azure OpenAI with API-key authentication, the minimum settings are:

```ini
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_AUTH_MODE=api_key
```

Use `AZURE_OPENAI_AUTH_MODE=azure_cli` for Azure CLI credentials, or
`managed_identity` with an optional
`AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID`.

The research `claude_chat` backend is a Claude Code CLI adapter, not a direct
Anthropic API client. Install and authenticate `claude`, and set
`CLAUDE_CLI_BIN` only if the executable is not available as `claude` on
`PATH`. `ANTHROPIC_API_KEY` is one authentication option the CLI may consume.

The SkillOpt-Sleep `cursor` backend similarly requires a separately installed
and authenticated `cursor-agent`; harvesting with `--source cursor` alone does
not. Set `SKILLOPT_SLEEP_CURSOR_PATH` when the executable is not on `PATH`, and
`SKILLOPT_SLEEP_CURSOR_MODEL` to override its model. Cursor plugin installation
and the explicit project skill target are documented in the
[Cursor integration guide](https://github.com/microsoft/SkillOpt/blob/main/plugins/cursor/README.md).

OpenAI-compatible servers have three distinct entry points:

1. The research engine's generic `openai_compatible` backend uses
   `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_API_KEY`, and
   `OPENAI_COMPATIBLE_MODEL`.
2. The research `openai_chat` backend can use
   `AZURE_OPENAI_AUTH_MODE=openai_compatible` with
   `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`.
3. SkillOpt-Sleep uses the same Azure-family variables as item 2 with
   `skillopt-sleep run --backend azure_openai`.

For research train/eval commands, `model.optimizer` and `model.target` in the
YAML config are applied after backend initialization. They override model-name
environment variables such as `OPENAI_COMPATIBLE_MODEL` and
`QWEN_CHAT_MODEL`; set both role models explicitly when selecting those
backends.

!!! tip
    You only need to configure the backend you plan to use. See
    [Configuration](configuration.md#model-backends) for exact backend names
    and role-specific overrides.

## Verify Installation

```bash
python -c "import skillopt; print('SkillOpt ready!')"
skillopt-train --help
skillopt-eval --help
skillopt-sleep --help
```

## Next Steps

→ [Run your first experiment](first-experiment.md)

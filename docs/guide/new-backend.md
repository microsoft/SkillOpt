# Add a New Model Backend

SkillOpt's model layer is function-based: each chat backend is a Python module
that exposes the call, token-tracking, and deployment-setting functions used by
`skillopt.model`. There is no backend base class or registry object to subclass.

## Built-in: the generic OpenAI-compatible backend

!!! note "Version requirement"
    This backend landed after v0.2.0. Install from the latest `main` until it is
    included in the next release.

Before writing a new backend, check whether your provider already speaks the
OpenAI Chat Completions protocol. Most do, in which case you can use the
built-in **`openai_compatible`** backend
(`skillopt/model/openai_compatible_backend.py`) with no code changes.

A single `base_url` + `api_key` pair lets you point SkillOpt at, for example:

| Provider | `base_url` | Example model |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Together AI | `https://api.together.xyz/v1` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Novita AI | `https://api.novita.ai/openai/v1` | `zai-org/glm-5.2` |
| Ollama (local) | `http://localhost:11434/v1` | `qwen2.5:7b` |
| vLLM / SGLang / TGI | `http://localhost:8000/v1` | your served model |
| LiteLLM proxy | `http://localhost:4000` | any proxied model |
| OpenRouter / Fireworks / xAI / … | provider base URL | provider model id |
| OrcaRouter | `https://api.orcarouter.ai/v1` | `openai/gpt-5.5` |
| Cheaper Inference | `https://api.cheaperinference.com/v1` | `gpt-5.4-mini` |

[OrcaRouter](https://www.orcarouter.ai) exposes OpenAI, Anthropic, Google,
DeepSeek and other models behind one OpenAI-compatible endpoint. Its model IDs
are namespaced by upstream provider (e.g. `openai/gpt-5.5`,
`anthropic/claude-sonnet-4.6`), so pass a namespaced ID as the model.

Cheaper Inference model IDs are not namespaced (e.g. `gpt-5.4-mini`), so pass
the bare ID as the model. See its [API docs](https://www.cheaperinference.com/docs)
for the base URL and its [model list](https://www.cheaperinference.com/markets)
for the model IDs.

### Python API

Select and configure the backend directly when embedding SkillOpt as a Python
library:

```python
import skillopt.model as model

# Use the generic backend for both optimizer and target calls.
model.set_backend("openai_compatible")
model.configure_openai_compatible(
    base_url="https://api.deepseek.com/v1",
    api_key="sk-...",
    model="deepseek-chat",
)
```

`configure_openai_compatible()` also accepts `optimizer_*` and `target_*`
arguments when the two roles use different endpoints or models.

### Environment variables

The shared variables below configure both roles. Role-specific
`OPTIMIZER_OPENAI_COMPATIBLE_*` and `TARGET_OPENAI_COMPATIBLE_*` variables take
precedence:

```bash
export OPENAI_COMPATIBLE_BASE_URL="https://api.groq.com/openai/v1"
export OPENAI_COMPATIBLE_API_KEY="gsk_..."
export OPENAI_COMPATIBLE_MODEL="llama-3.3-70b-versatile"
# Optional: OPENAI_COMPATIBLE_TEMPERATURE, _MAX_TOKENS, _TIMEOUT_SECONDS
```

For direct library use, `OPTIMIZER_BACKEND=openai_compatible` and/or
`TARGET_BACKEND=openai_compatible` select the role. The training and evaluation
scripts resolve backend selection from their config, so set the split fields
explicitly there:

```yaml
model:
  optimizer_backend: openai_compatible
  target_backend: openai_compatible
  optimizer: llama-3.3-70b-versatile
  target: llama-3.3-70b-versatile
```

Equivalently, override those fields on the command line:

```bash
python scripts/train.py --config configs/searchqa/default.yaml \
  --cfg-options \
  model.optimizer_backend=openai_compatible \
  model.target_backend=openai_compatible \
  model.optimizer=llama-3.3-70b-versatile \
  model.target=llama-3.3-70b-versatile
```

`model.backend` is a high-level shorthand for SkillOpt's built-in role
pairings. It replaces role fields that are still at the shipped
`openai_chat` default, while non-default role fields are preserved. For a
custom optimizer/target split, set both role-specific fields and do not combine
them with a conflicting high-level label.

The generic backend uses the official `openai` SDK and the Chat Completions
API. It records token usage through the shared tracker, supports provider tool
calling through `chat_*_messages(..., tools=...)`, and exposes `count_tokens()`
(tiktoken when available, with a character-based fallback). Provider-specific
Responses API features are outside this backend's contract.

Only write a new backend when the provider is not compatible with this surface
or requires behavior that cannot be expressed by its configuration.

## Backend architecture

The active split optimizer/target dispatcher is the public
`skillopt/model/__init__.py` module:

```text
skillopt/model/
├── common.py                       # aliases, default models, token/response helpers
├── backend_config.py               # optimizer/target whitelists and runtime selection
├── __init__.py                     # public API and split-role dispatch
├── openai_compatible_backend.py    # generic Chat Completions example
├── qwen_backend.py                 # raw-HTTP chat example with per-role config
├── minimax_backend.py              # compact raw-HTTP chat example
├── codex_harness.py                # target-only exec harnesses
└── router.py                       # legacy single-backend compatibility surface
```

`router.py` is not the dispatcher used by the current training loop. Update it
only if the new backend must also be exposed through that legacy single-backend
API.

## Step 1: implement the module contract

Create a module such as `skillopt/model/your_backend.py`. Copy the signatures
from `openai_compatible_backend.py` or `qwen_backend.py`; model calls in the
current framework are synchronous.

For a chat backend that supports both roles, the public module surface is:

| Function | Purpose |
|---|---|
| `chat_optimizer(...)` | Optimizer system/user call; returns `(text, usage)` |
| `chat_target(...)` | Target system/user call; returns `(text, usage)` |
| `chat_optimizer_messages(...)` | Optimizer message-list call, including optional tools |
| `chat_target_messages(...)` | Target message-list call, including optional tools |
| `get_token_summary()` | Return per-stage counters plus `_total` |
| `reset_token_tracker()` | Clear this backend's counters |
| `set_optimizer_deployment(name)` | Change the optimizer model at runtime |
| `set_target_deployment(name)` | Change the target model at runtime |
| `set_reasoning_effort(effort)` | Apply or safely ignore the shared reasoning setting |

Every call returns a usage dict with `prompt_tokens`, `completion_tokens`, and
`total_tokens`. Use `TokenTracker` from `skillopt.model.common` and record each
call exactly once. Message-list calls that accept tools should return the
compatibility message objects from `common.py` when `return_message=True`.

Provider-specific configuration helpers and `count_tokens()` are optional, but
their state must be safe to update while calls may run concurrently. Keep
credentials out of logs and persisted artifacts.

Exec-style targets such as `claude_code_exec` and `cursor_exec` do not
implement this chat contract. They are target-only and are integrated through
`codex_harness.py` plus environment-specific rollout code.

## Step 2: register and route the backend

A new backend normally requires all of the following:

1. Add its canonical name, aliases, and default model to
   `skillopt/model/common.py`.
2. Add the canonical name to the appropriate optimizer and/or target whitelist
   in `skillopt/model/backend_config.py`. Do not advertise a role the module
   cannot execute.
3. Import the module in `skillopt/model/__init__.py` and add dispatch branches
   for every supported call surface.
4. Include its counters in `get_token_summary()` / `reset_token_tracker()` and
   forward the shared deployment/reasoning setters where applicable.
5. If it has YAML settings, add structured-to-flat mappings in
   `skillopt/config.py`, wire them through `scripts/train.py` and
   `scripts/eval_only.py`, and document their precedence over environment
   variables.
6. Update `router.py` only when legacy single-backend compatibility is part of
   the intended feature.

Backend selection in `scripts/train.py` must support
`model.optimizer_backend` and `model.target_backend`. If the backend also has a
high-level `model.backend` alias, resolve its built-in role pairing before
applying backend-specific model defaults, without replacing non-default role
fields.

## Step 3: test the integration

Add focused tests under `tests/` that do not call a live provider. At minimum,
cover:

- optimizer and target whitelist validation;
- routing for text and message-list calls;
- role-specific configuration precedence;
- tool-call compatibility, if supported;
- deployment/reasoning setters;
- token accounting, including a single correct `_total`;
- actionable errors for missing credentials or invalid responses.

Then run the focused test, the full suite, and the documentation build:

```bash
python -m pytest tests/test_your_backend.py -q
python -m pytest tests/ -q
mkdocs build --strict
```

Also update `.env.example`, the configuration reference, and the backend table
in the API reference. Add an optional dependency extra only when the backend
requires a package that is not already a core dependency.

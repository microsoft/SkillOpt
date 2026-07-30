# Configuration Reference

SkillOpt loads structured YAML, resolves `_base_` inheritance, and flattens
the result for the trainer. Shipped defaults live in
`configs/_base_/default.yaml`; benchmark configs override them.

## Model and Backend Selection

Use explicit optimizer and target backends when the two roles differ or when
selecting the generic OpenAI-compatible backend.

| Backend | Optimizer | Target |
|---|:---:|:---:|
| `openai_chat` | ✓ | ✓ |
| `openai_compatible` | ✓ | ✓ |
| `claude_chat` | ✓ | ✓ |
| `qwen_chat` | ✓ | ✓ |
| `minimax_chat` | ✓ | ✓ |
| `codex_exec` | ✓ | ✓ |
| `claude_code_exec` | — | ✓ |
| `cursor_exec` | — | ✓ |

MiniMax currently has one shared deployment. `model.minimax_model` is applied
when MiniMax is the target; mixed-backend runs cannot independently choose a
MiniMax optimizer model and a different target model.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model.backend` | str | `azure_openai` | Backward-compatible high-level run label |
| `model.optimizer` | str | `gpt-5.5` | Optimizer deployment/model |
| `model.target` | str | `gpt-5.5` | Target deployment/model |
| `model.optimizer_backend` | str | `openai_chat` | Optimizer client path; chat backends plus `codex_exec` |
| `model.target_backend` | str | `openai_chat` | Target client path; chat or exec backend |
| `model.reasoning_effort` | str | `medium` | Shared reasoning effort |
| `model.rewrite_reasoning_effort` | str | empty | Optional full-rewrite effort override |
| `model.rewrite_max_completion_tokens` | int | `64000` | Full-rewrite output cap |

### Azure/OpenAI `openai_chat`

| Parameter | Default | Description |
|---|---|---|
| `model.azure_openai_endpoint` | empty | Shared Azure resource URL or compatibility-mode base URL |
| `model.azure_openai_api_version` | `2024-12-01-preview` | Azure API version |
| `model.azure_openai_api_key` | empty | Key for `api_key` or compatibility auth |
| `model.azure_openai_auth_mode` | empty | Config value; empty falls back to env, whose default is `azure_cli` |
| `model.azure_openai_ad_scope` | Azure Cognitive Services scope | AAD token scope |
| `model.azure_openai_managed_identity_client_id` | empty | Optional user-assigned identity client ID |

Every shared key also has an `optimizer_azure_openai_*` and
`target_azure_openai_*` form.

### Claude `claude_chat`

`claude_chat` launches an installed, authenticated Claude Code CLI with
`claude -p`; it does not instantiate an Anthropic API client. The executable
defaults to `claude` and can be overridden with `CLAUDE_CLI_BIN`.
`ANTHROPIC_API_KEY` is one authentication option understood by the CLI.

### Qwen, MiniMax, and Exec Backends

| Parameter family | Description |
|---|---|
| `model.qwen_chat_*` | Shared `base_url`, `api_key`, `temperature`, `timeout_seconds`, `max_tokens`, and `enable_thinking` |
| `model.optimizer_qwen_chat_*` / `model.target_qwen_chat_*` | Per-role Qwen overrides |
| `model.minimax_*` | MiniMax `base_url`, `api_key`, shared `minimax_model`, `temperature`, `max_tokens`, and `enable_thinking`; `minimax_model` applies when MiniMax is the target |
| `model.codex_exec_*` | Codex path, sandbox, profile, SDK mode, reasoning, network/search, and approval policy |
| `model.claude_code_exec_*` | Claude path, profile, SDK mode, effort, and thinking-token cap |
| `model.cursor_exec_path` | Cursor Agent executable path; default `cursor-agent` |
| `model.cursor_exec_sandbox` | Cursor sandbox mode: `enabled` (default) or `disabled`; file-edit rollouts require `enabled` |

## Training (`train`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train.num_epochs` | int | `4` | Training epochs |
| `train.train_size` | int | `0` | `0` derives the size from the dataset split |
| `train.steps_per_epoch` | int | derived | Runtime field recomputed from train size, batch size, and accumulation; configured values are overwritten |
| `train.batch_size` | int | `40` | Tasks sampled per step |
| `train.accumulation` | int | `1` | Accumulation rounds per step |
| `train.seed` | int | `42` | Random seed |

## Gradient / Reflection (`gradient`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gradient.minibatch_size` | int | `8` | Reflect minibatch size |
| `gradient.merge_batch_size` | int | `8` | Patch merge batch size |
| `gradient.analyst_workers` | int | `16` | Parallel reflection workers |
| `gradient.max_analyst_rounds` | int | `3` | Maximum analyst rounds |
| `gradient.failure_only` | bool | `false` | Reflect only on failures |

## Optimizer (`optimizer`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer.learning_rate` | int | `4` | Maximum edit patches per step |
| `optimizer.min_learning_rate` | int | `2` | Floor for decaying schedules |
| `optimizer.lr_scheduler` | str | `cosine` | `constant`, `linear`, `cosine`, or `autonomous` |
| `optimizer.lr_control_mode` | str | `fixed` | `fixed`, `autonomous`, or `none` |
| `optimizer.skill_update_mode` | str | `patch` | `patch`, `rewrite_from_suggestions`, or `full_rewrite_minibatch` |
| `optimizer.use_slow_update` | bool | `true` | Epoch-boundary longitudinal update |
| `optimizer.slow_update_samples` | int | `20` | Longitudinal evaluation samples |
| `optimizer.slow_update_gate_with_selection` | bool | `false` | Gate slow-update guidance on the selection split |
| `optimizer.longitudinal_pair_policy` | str | `mixed` | `mixed`, `changed`, or `unchanged` |
| `optimizer.use_meta_skill` | bool | `true` | Cross-epoch optimizer memory |
| `optimizer.use_skill_aware_reflection` | bool | `false` | Enable skill-defect vs execution-lapse routing |
| `optimizer.skill_aware_appendix_source` | str | `both` | `both` or `failure_only` |
| `optimizer.skill_aware_consolidate_threshold` | int | `0` | Appendix compaction threshold; `0` disables it |

## Evaluation (`evaluation`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `evaluation.use_gate` | bool | `true` | Accept only improvements when enabled; `false` records validation but force-accepts each candidate |
| `evaluation.gate_metric` | str | `hard` | `hard`, `soft`, or `mixed` |
| `evaluation.gate_mixed_weight` | float | `0.5` | Soft-score weight for `mixed` |
| `evaluation.use_semantic_density` | bool | `false` | Add the optional instruction-density bonus |
| `evaluation.semantic_density_weight` | float | `0.05` | Density bonus weight |
| `evaluation.leading_words` | list/str | built in | Optional custom high-influence words |
| `evaluation.sel_env_num` | int | `0` | Selection size; `0` uses the full split |
| `evaluation.test_env_num` | int | `0` | Test size; `0` uses the full split |
| `evaluation.eval_test` | bool | `true` | Run final test evaluation |

## Environment (`env`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `env.name` | str | empty | Benchmark name |
| `env.skill_init` | str | empty | Initial skill document |
| `env.split_mode` | str | `ratio` | `ratio` or `split_dir` |
| `env.split_ratio` | str | benchmark/default | Train:validation:test ratio |
| `env.split_seed` | int | `42` | Deterministic split seed |
| `env.split_dir` | str | empty | Materialized train/val/test directory |
| `env.data_path` | str | empty | Raw data path for ratio mode |
| `env.split_output_dir` | str | empty | Optional materialized split output |
| `env.exec_timeout` | int | `120` | Per-task timeout in seconds |
| `env.out_root` | str | generated by the train/eval CLIs | Output directory |

Benchmark-specific `env` keys are passed through to the adapter.

## Persistent Memory (mem0) — optional, off by default

Stores skill iterations and reflection summaries in [mem0](https://mem0.ai) and
reads a small amount of relevant history back into the Reflect stage.

**Disabled unless `mem0_enabled` is explicitly true.** A `MEM0_API_KEY` present
in the environment for another application does *not* enable it. Install with
`pip install 'skillopt[mem0]'`.

Set these under `train:` in a structured config (every shipped config is
structured), at the top level of a flat config, or via
`--cfg-options mem0_enabled=true`. A ready-made example ships at
`configs/features/mem0_memory.yaml`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train.mem0_enabled` | bool | `false` | Master switch. Nothing is sent unless this is true |
| `train.mem0_api_key` | str | empty | Falls back to `MEM0_API_KEY`, read only when enabled. Redacted from `config.json` and the run summary; prefer the env var |
| `train.mem0_namespace` | str | derived | Override the namespace; default is derived per project |
| `train.mem0_retrieval_enabled` | bool | `true` | Read memory back into reflection; writes continue if false |
| `train.mem0_retrieval_limit` | int | `5` | Max records fetched per retrieval |
| `train.mem0_timeout_seconds` | float | `5.0` | Hard per-call bound. On timeout the executor is retired and training continues; after 3 consecutive failures memory disables itself for the run |
| `train.mem0_max_chars` | int | `4000` | Cap on any single stored payload, applied after redaction |

### What leaves the machine

When enabled, exactly two record types are sent, both redacted first:

1. **Skill iteration** — epoch, step, score, skill hash and length, the `env`
   and model names, and the skill text (capped at `mem0_max_chars`).
2. **Reflection summary** — epoch, step, patch count, rollout scores, and a
   JSON summary of up to 10 patches with any `skill_text` field removed.

Retrieval sends the first 600 characters of the current skill as a similarity
query.

Before transmission every payload passes through
`skillopt.memory.redaction.redact_for_upload`, which strips vendor API keys,
bearer/basic tokens, JWTs, private keys, `key = value` secret assignments, the
project root, and `/home/<user>`-style prefixes. Relative paths and filenames
are deliberately preserved so stored memories stay useful.

### Failure behaviour

A call that exceeds `mem0_timeout_seconds` releases the training step on time and
retires the worker, so the next step does not queue behind a wedged request. After
three consecutive failures the backend stops calling out for the remainder of the run
and logs once. A persistently unreachable service therefore costs a bounded total
rather than a bounded amount on every step.

Text retrieved from mem0 is redacted again before it is inserted into the reflection
prompt. Outbound redaction alone is not sufficient: the store is external and may hold
records written by an older version, another tool, or a shared namespace, and anything
it returns is forwarded to the optimizer's model provider.

### Namespacing

Memories are scoped to `skillopt:<env>:<digest>`, where the digest is a SHA-256
prefix of a **stable project identity** — the enclosing git repository root, or the
current working directory when there is no repository. Stable across runs of one
project, distinct across projects, and the raw path is never transmitted.

Identity is deliberately *not* derived from `out_root`. The train/eval CLIs default that
to `outputs/skillopt_<env>_<model>_<timestamp>`, so deriving from it would mint a new
namespace on every run and cross-run retrieval would never return anything. `out_root`
is still used to anchor path redaction, which is what it is right for.

## Credential Environment Variables

### Azure-family backend

| Variable | Description |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Shared Azure endpoint or compatibility base URL |
| `AZURE_OPENAI_API_VERSION` | Azure API version |
| `AZURE_OPENAI_AUTH_MODE` | `api_key`, `azure_cli`, `managed_identity`, or `openai_compatible` |
| `AZURE_OPENAI_API_KEY` | Key for `api_key` or `openai_compatible` mode |
| `AZURE_OPENAI_AD_SCOPE` | Optional AAD scope |
| `AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID` | Optional managed-identity client ID |

Use `OPTIMIZER_AZURE_OPENAI_*` and `TARGET_AZURE_OPENAI_*` for role-specific
overrides.

### Generic OpenAI-compatible backend

| Variable suffix | Shared / per-role forms |
|---|---|
| `BASE_URL` | `OPENAI_COMPATIBLE_BASE_URL`, `OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL`, `TARGET_OPENAI_COMPATIBLE_BASE_URL` |
| `API_KEY` | Corresponding shared/optimizer/target `*_API_KEY` names |
| `MODEL` | Corresponding shared/optimizer/target `*_MODEL` names |
| `TEMPERATURE` | Corresponding shared/optimizer/target `*_TEMPERATURE` names |
| `MAX_TOKENS` | Corresponding shared/optimizer/target `*_MAX_TOKENS` names |
| `TIMEOUT_SECONDS` | Corresponding shared/optimizer/target `*_TIMEOUT_SECONDS` names |

The train/eval entry points set deployments from YAML `model.optimizer` and
`model.target` after backend initialization. For selected OpenAI-compatible or
Qwen roles, those values override the corresponding `*_MODEL` environment
variables; the environment model names mainly seed direct library use.

Other backend families use the authenticated Claude CLI (`CLAUDE_CLI_BIN`;
optionally `ANTHROPIC_API_KEY`), `QWEN_CHAT_*`, and `MINIMAX_*`.
SkillOpt-Sleep's compatible endpoint uses `AZURE_OPENAI_*`, not the research
backend's `OPENAI_COMPATIBLE_*`; see
[the Sleep endpoint guide](../sleep/openai-compatible-endpoints.md).

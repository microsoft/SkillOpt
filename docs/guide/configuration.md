# Configuration Guide

SkillOpt uses YAML configuration files with a hierarchical override system.

## Config Structure

```
configs/
├── _base_/
│   └── default.yaml          # Global defaults
├── searchqa/
│   └── default.yaml          # SearchQA overrides
├── docvqa/
│   └── default.yaml          # DocVQA overrides
└── alfworld/
    └── default.yaml          # ALFWorld overrides
```

Benchmark configs inherit from `_base_/default.yaml` and override specific values.

## Key Parameters

### Model

```yaml
model:
  backend: azure_openai          # azure_openai | openai_chat | claude_code_exec | qwen
  optimizer: gpt-5.5               # Optimizer model (for reflection)
  target: gpt-5.5               # Target model (for rollout)
```

### Training

```yaml
train:
  num_epochs: 4                  # Number of training epochs
  batch_size: 40                 # Tasks per step (batch size)
  accumulation: 1                # Gradient accumulation
  seed: 42
```

### Gradient (Reflection)

```yaml
gradient:
  minibatch_size: 8              # Reflect minibatch size
  analyst_workers: 16            # Parallel reflection workers
  max_analyst_rounds: 3          # Max rounds of analyst reflection
  failure_only: false            # Only reflect on failures
```

### Optimizer

```yaml
optimizer:
  learning_rate: 4               # Max edits per step (edit budget)
  min_learning_rate: 2           # Min edits for decay schedulers
  lr_scheduler: cosine           # constant | linear | cosine | autonomous
  use_slow_update: true          # Momentum-like blending at epoch boundary
  slow_update_samples: 20        # Samples for slow update evaluation
  use_meta_skill: true           # Cross-epoch strategy memory
```

### Skill-Aware Reflection (optional, off by default)

EmbodiSkill-style failure routing: the failure analyst classifies each
failure pattern as **SKILL_DEFECT** (the rule is wrong or missing → normal
gated body edit) or **EXECUTION_LAPSE** (a valid rule exists but was not
followed → a short reminder appended to a protected appendix region inside
the skill that step-level edits can never modify).

```yaml
optimizer:
  use_skill_aware_reflection: false    # Master switch (default off = baseline-identical)
  skill_aware_appendix_source: both    # both | failure_only (paper-faithful S_app)
  skill_aware_consolidate_threshold: 0 # >0: LLM-compact the appendix past N notes (experimental)
```

Notes:

- The switch is resolved process-wide from the config
  (`configure_skill_aware_reflection`), so it applies to every benchmark
  with no per-adapter wiring.
- `failure_only` restricts appendix notes to the failure analyst, matching
  the original S_app formulation; `both` additionally lets the success
  analyst re-emphasize existing rules.
- Appendix notes bypass the validation gate by design and accumulate with
  order-preserving dedup; lapse-only steps (no body edits) still flush
  their notes.
- Not supported together with `skill_update_mode=rewrite_from_suggestions`
  or the full-rewrite modes: whole-document rewrites can drop the appendix
  region.

### Evaluation

```yaml
evaluation:
  use_gate: true                 # Validation gating (accept/reject updates)
  eval_test: true                # Run test evaluation after training
```

### Environment (Data)

```yaml
env:
  name: searchqa                 # Benchmark name
  split_mode: ratio              # ratio | split_dir
  split_ratio: "2:1:7"           # train:val:test ratio
  data_path: ""                  # Path to dataset
  exec_timeout: 120              # Per-task timeout (seconds)
```

## CLI Overrides

Override any config value from the command line:

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  optimizer.learning_rate=16 \
  optimizer.lr_scheduler=linear \
  gradient.analyst_workers=8
```

## Environment Variables

Model credentials are loaded from environment variables:

| Variable | Backend | Description |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | azure_openai | Azure resource endpoint |
| `AZURE_OPENAI_API_KEY` | azure_openai | Azure API key |
| `OPENAI_API_KEY` | openai | OpenAI API key |
| `ANTHROPIC_API_KEY` | claude | Anthropic API key |
| `QWEN_API_BASE` | qwen | Local Qwen vLLM endpoint |

## Full Reference

See [Configuration Reference](../reference/config.md) for the complete parameter list.

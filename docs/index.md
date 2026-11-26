---
hide:
  - navigation
---

<div class="hero" markdown>

# SkillOpt

### Train Agent Skills Like Neural Networks

*Optimize natural-language skill documents through iterative rollout, reflection, and gated validation — with epochs, learning rates, and validation gates — without touching model weights.*

[Get Started :material-rocket-launch:](guide/installation.md){ .md-button .md-button--primary }
[View on GitHub :material-github:](https://github.com/microsoft/SkillOpt){ .md-button }

</div>

---

## Two Complementary Workflows

| Workflow | Package / command | Use it for |
|---|---|---|
| **Research engine** | `skillopt`, `skillopt-train`, `skillopt-eval` | Train and evaluate skill documents on explicit benchmark splits. |
| **SkillOpt-Sleep (preview)** | `skillopt_sleep`, `skillopt-sleep` | Review supported coding-agent sessions and stage proposed memory/skill updates for human adoption. |

They share the idea of bounded text updates and validation, but they are
separate entry points with different configs and safety boundaries. Start with
the [SkillOpt-Sleep overview](sleep/README.md) before using real session data.

---

## How It Works

<div class="pipeline-container" markdown>
<div class="pipeline-wrapper">

<div class="pipeline-stage" id="stage-rollout">
<div class="stage-icon">🎯</div>
<div class="stage-label">Rollout</div>
<div class="stage-desc">Target executes tasks</div>
</div>

<div class="pipeline-arrow"><div class="flow-line"></div></div>

<div class="pipeline-stage" id="stage-reflect">
<div class="stage-icon">🔍</div>
<div class="stage-label">Reflect</div>
<div class="stage-desc">Optimizer analyzes trajectories</div>
</div>

<div class="pipeline-arrow"><div class="flow-line"></div></div>

<div class="pipeline-stage" id="stage-aggregate">
<div class="stage-icon">🔗</div>
<div class="stage-label">Aggregate</div>
<div class="stage-desc">Merge edit patches</div>
</div>

<div class="pipeline-arrow"><div class="flow-line"></div></div>

<div class="pipeline-stage" id="stage-select">
<div class="stage-icon">✂️</div>
<div class="stage-label">Select</div>
<div class="stage-desc">Rank & clip edits</div>
</div>

<div class="pipeline-arrow"><div class="flow-line"></div></div>

<div class="pipeline-stage" id="stage-update">
<div class="stage-icon">📝</div>
<div class="stage-label">Update</div>
<div class="stage-desc">Apply to skill doc</div>
</div>

<div class="pipeline-arrow"><div class="flow-line"></div></div>

<div class="pipeline-stage" id="stage-gate">
<div class="stage-icon">🚦</div>
<div class="stage-label">Gate</div>
<div class="stage-desc">Validate & accept</div>
</div>

</div>

<div class="pipeline-epoch-bar">
<div class="epoch-mechanism">🔄 Slow Update</div>
<div class="epoch-mechanism">🧠 Meta Skill</div>
<div class="epoch-label">Epoch Boundary</div>
</div>

</div>

---

## Deep Learning Analogy

SkillOpt brings the familiar deep-learning training paradigm to agentic prompt optimization:

| Deep Learning | SkillOpt | 
|---|---|
| Model weights | Skill document (Markdown) |
| Forward pass | Rollout (target executes tasks) |
| Loss / gradient | Reflect (optimizer produces edit patches) |
| Gradient clipping | Edit selection (`learning_rate` = max edits) |
| SGD step | Patch application to skill |
| Validation set | Gated evaluation on selection split |
| LR schedule | `lr_scheduler`: cosine, linear, constant |
| Epochs | Multi-epoch with slow update & meta skill memory |

---

## Supported Benchmarks

| Benchmark | Type | Config |
|---|---|---|
| **DocVQA** | Document QA | `configs/docvqa/` |
| **ALFWorld** | Embodied AI | `configs/alfworld/` |
| **OfficeQA** | Enterprise QA | `configs/officeqa/` |
| **SearchQA** | Open-domain QA | `configs/searchqa/` |
| **LiveMathematicianBench** | Math reasoning | `configs/livemathematicianbench/` |
| **SpreadsheetBench** | Spreadsheet editing | `configs/spreadsheetbench/` |
| **AppWorld** | Interactive coding agents over app APIs | `configs/appworld/` ([guide](guide/appworld.md)) |

---

## Model Backends

Optimizer and target roles are configured separately. Chat backends include
Azure OpenAI (`openai_chat`), the provider-neutral
`openai_compatible` backend, the Claude Code CLI (`claude_chat`), Qwen, and
MiniMax. Codex and Claude Code exec harnesses are target-only and require
adapter support. Despite its name, `claude_chat` launches `claude -p`; it is
not a direct Anthropic API client.

If a provider implements OpenAI Chat Completions, begin with the
[built-in compatible backend](guide/new-backend.md#built-in-the-generic-openai-compatible-backend)
instead of adding a new integration. See [Configuration](guide/configuration.md)
for authentication and per-role overrides.

---

## Quick Example

```bash
# Clone and install the research checkout plus the SearchQA data extra
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt
python -m pip install -e ".[searchqa]"

# Configure credentials (choose one auth mode in .env)
cp .env.example .env
set -a; source .env; set +a

# Materialize the runnable split from the checked-in ID manifest
python scripts/materialize_searchqa.py

# Train on SearchQA into a predictable output directory
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --out_root outputs/searchqa_quickstart

# Evaluate best skill
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa_quickstart/best_skill.md \
  --split valid_unseen
```

---

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **Getting Started**

    ---

    Install SkillOpt, configure your API keys, and run your first experiment in 5 minutes.

    [:octicons-arrow-right-24: Installation](guide/installation.md)

-   :material-puzzle:{ .lg .middle } **Add a Benchmark**

    ---

    Extend SkillOpt with your own benchmark in ~100 lines of code.

    [:octicons-arrow-right-24: Extension Guide](guide/new-benchmark.md)

-   :material-cog:{ .lg .middle } **Configuration**

    ---

    Full reference for all hyperparameters with deep learning analogies.

    [:octicons-arrow-right-24: Config Reference](reference/config.md)

-   :material-monitor-dashboard:{ .lg .middle } **WebUI**

    ---

    Configure, launch, and monitor training from your browser.

    [:octicons-arrow-right-24: WebUI Guide](guide/first-experiment.md#webui)

-   :material-weather-night:{ .lg .middle } **SkillOpt-Sleep**

    ---

    Test the deployment companion with the no-provider mock path, then review
    its data boundary before selecting a real backend.

    [:octicons-arrow-right-24: Sleep Overview](sleep/README.md)

</div>

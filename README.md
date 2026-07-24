# SkillOpt: Executive Strategy for Self-Evolving Agent Skills

*Train agent skills like you train neural networks — with epochs, (mini-)batchsize, learning rates, and validation gates — but without touching model weights.*

[![Project Page](https://img.shields.io/badge/Project%20Page-SkillOpt-8dbb3c)](https://microsoft.github.io/SkillOpt/) [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2605.23904) [![Project Video](https://img.shields.io/badge/Project%20Video-Watch%20Demo-ff0000)](https://youtu.be/JUBMDTCiM0M) [![PyPI](https://img.shields.io/badge/PyPI-skillopt-green.svg)](https://pypi.org/project/skillopt/) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <a href="https://trendshift.io/repositories/38498?utm_source=trendshift-badge&utm_medium=badge&utm_campaign=badge-trendshift-38498" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/38498/daily?language=Python" alt="microsoft%2FSkillOpt | Trendshift" width="250" height="55"/></a>
  <a href="https://trendshift.io/repositories/38498?utm_source=trendshift-badge&utm_medium=badge&utm_campaign=badge-trendshift-38498" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/38498/weekly?language=Python" alt="microsoft%2FSkillOpt | Trendshift" width="250" height="55"/></a>
</p>

> 📖 **For installation, data preparation, training/eval commands, configuration, and framework internals, start with the versioned [SkillOpt documentation](https://github.com/microsoft/SkillOpt/blob/main/docs/index.md). A concise rendered overview is available in the [Documentation & Reproduction Guide](https://microsoft.github.io/SkillOpt/docs/guideline.html), and longer-form engineering analysis appears on the [Technical Blog](https://microsoft.github.io/SkillOpt/blog/). We also maintain a [Changelog](CHANGELOG.md) for released and unreleased changes.**

---

## News 🔥🔥🔥
- **[2026-07-24]** 📰 **SkillOpt in the news.** Read the official [Microsoft Research feature](https://www.microsoft.com/en-us/research/blog/skillopt-agent-skills-as-trainable-parameters/), along with recent coverage from [VentureBeat](https://venturebeat.com/orchestration/microsofts-open-source-skillopt-automatically-upgrades-ai-agent-skills-without-touching-model-weights), [Synced (机器之心)](https://mp.weixin.qq.com/s/pMlyj3a3KOh8L7cIHClRXA), [Flowtivity](https://flowtivity.ai/blog/microsoft-skillopt-train-ai-agent-skills/), and [The Decoder](https://the-decoder.com/microsofts-skillopt-boosts-gpt-5-5-by-using-nothing-but-a-trained-markdown-file/).
- **[2026-07-02]** 🚀 **SkillOpt [v0.2.0](https://github.com/microsoft/SkillOpt/releases/tag/v0.2.0) is out on [PyPI](https://pypi.org/project/skillopt/)!** Headline feature: **SkillOpt-Sleep**, a nightly offline self-evolution engine (harvest → mine → replay → consolidate behind a held-out validation gate), now shipped as the `skillopt-sleep` CLI. It also includes experimental multi-objective, replay, and dream-rollout controls; the main CLI keeps conservative defaults and does not expose every experiment-harness control as a flag. The release source adds integration shells for **Claude Code, Codex, Copilot, and Devin**, plus an **OpenClaw reference adaptation**; these plugin/MCP files live in the repository rather than the PyPI wheel. It also adds SearchQA split materialization, Windows robustness, and hardened JSON parsing. See the [release notes](https://github.com/microsoft/SkillOpt/releases/tag/v0.2.0) for full release details and contributor acknowledgements.
- **[2026-06-15]** 😴 **SkillOpt-Sleep (preview)** — a nightly offline self-evolution companion for local coding agents (Claude Code / Codex / Copilot): review past sessions, replay recurring tasks, and consolidate validated skills behind a held-out gate. See **[`docs/sleep/README.md`](docs/sleep/README.md)** for what it is, how to use it, and results.
- **[2026-06-03]** 🎉 **[gbrain](https://github.com/garrytan/gbrain), [gbrain-evals](https://github.com/garrytan/gbrain-evals/blob/main/docs/benchmarks/2026-06-03-skillopt.md), and [darwin-skill](https://github.com/alchaincyf/darwin-skill) have all integrated SkillOpt.**
- **[2026-06-02]** 🎉 **SkillOpt [v0.1.0](https://github.com/microsoft/SkillOpt/releases/tag/v0.1.0) is now available on [PyPI](https://pypi.org/project/skillopt/)!** Install with `pip install skillopt`. This initial release includes the full training loop (rollout → reflect → aggregate → select → update → evaluate), multi-backend support (OpenAI / Azure / Claude / Qwen / MiniMax), six built-in benchmarks, and WebUI dashboard.

---

## Overview

Modern agent skills are usually hand-crafted, generated one-shot by a strong
LLM, or evolved through loosely controlled self-revision — none of which
behaves like a deep-learning optimizer for the skill itself, and none of
which reliably improves over its starting point under feedback.

**SkillOpt treats the skill document as the trainable state of a frozen
agent**, and trains it with the discipline that makes weight-space
optimization reproducible. A separate optimizer model turns scored rollouts
into bounded add / delete / replace edits on a single skill document; in the
default paper-style path, a candidate edit is accepted only when it strictly
improves a held-out validation score. A textual learning-rate budget, a rejected-edit buffer,
and an epoch-wise slow / meta update make skill training stable while
adding **zero inference-time model calls** at deployment.

The deployed artifact is a compact `best_skill.md` (typically 300–2,000
tokens) that runs against the unchanged target model. Across **six
benchmarks, seven target models, and three execution harnesses** (direct
chat, Codex CLI, Claude Code CLI), SkillOpt is best or tied-best on **all
52 evaluated (model, benchmark, harness) cells** and on GPT-5.5 lifts the
average no-skill accuracy by **+23.5 points in direct chat, +24.8 inside
the Codex agentic loop, and +19.1 inside Claude Code**. Optimized skill
artifacts transfer across model scales, between Codex and Claude Code
harnesses, and to nearby benchmarks without further optimization.

For the full method, ablations, and per-cell results see the [paper](https://arxiv.org/abs/2605.23904); for a visual walkthrough of the loop see the [project page](https://microsoft.github.io/SkillOpt/); for deeper API / backend / benchmark docs see [`docs/`](docs/).

## 🎬 Demo Video

https://github.com/user-attachments/assets/eb12d3bc-371c-467f-904d-91b61f339ed7

<p align="center">
  <a href="https://youtu.be/JUBMDTCiM0M"><b>▶ Watch the full demo on YouTube</b></a>
</p>

---

## Extensibility & WebUI

### Adding a new backend

A backend = a chat / exec target (e.g. `openai_chat`, `claude_chat`,
`qwen_chat`, `minimax_chat`, `openai_compatible`, `codex_exec`,
`claude_code_exec`, `cursor_exec`). If a provider implements the OpenAI Chat Completions
protocol, try the built-in `openai_compatible` backend before adding code. See
[`docs/guide/new-backend.md`](docs/guide/new-backend.md) for the full
contract. Chat backends add a `skillopt/model/<name>_backend.py` module;
target-only exec backends use the shared harness in `codex_harness.py`.
Both register through `common.py`, `backend_config.py`, and
`skillopt/model/__init__.py`.

### Adding a new benchmark

A benchmark = a `skillopt/envs/<name>/` package with an adapter, a data loader,
a scored rollout helper, a YAML config, and optionally an initial seed skill.
See
[`docs/guide/new-benchmark.md`](docs/guide/new-benchmark.md) for the full
contract; the simplest reference is `skillopt/envs/searchqa/`.

### WebUI

Launch the monitoring dashboard (optional):

```bash
pip install -e ".[webui]"
python -m skillopt_webui.app
```

| Flag | Default | Description |
|---|---|---|
| `--port` | 7860 | Server port |
| `--host` | `0.0.0.0` | Bind address |
| `--share` | off | Create a public Gradio share link |

The default host listens on every network interface. Use
`--host 127.0.0.1` for local-only access.

---

## Citation

```bibtex
@article{yang2026skillopt,
  title={Skillopt: Executive strategy for self-evolving agent skills},
  author={Yang, Yifan and Gong, Ziyang and Huang, Weiquan and Yang, Qihao and Zhou, Ziwei and Huang, Zisu and Li, Yan and Gao, Xuemei and Dai, Qi and Liu, Bei and others},
  journal={arXiv preprint arXiv:2605.23904},
  year={2026}
}
```

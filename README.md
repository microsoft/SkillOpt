# SkillOpt: Executive Strategy for Self-Evolving Agent Skills

*Train agent skills like you train neural networks — with epochs, (mini-)batchsize, learning rates, and validation gates — but without touching model weights.*

[![Project Page](https://img.shields.io/badge/Project%20Page-SkillOpt-8dbb3c)](https://microsoft.github.io/SkillOpt/) [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2605.23904) [![Project Video](https://img.shields.io/badge/Project%20Video-Watch%20Demo-ff0000)](https://youtu.be/JUBMDTCiM0M) [![PyPI](https://img.shields.io/badge/PyPI-skillopt-green.svg)](https://pypi.org/project/skillopt/) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📖 **For installation, data preparation, training/eval commands, the full configuration reference, and framework internals, see the [Documentation & Reproduction Guide](docs/guideline.html)** — view it [rendered online](https://htmlpreview.github.io/?https://github.com/microsoft/SkillOpt/blob/main/docs/guideline.html) or via [GitHub Pages](https://microsoft.github.io/SkillOpt/docs/guideline.html).

---

## News 🔥🔥🔥
- **[2026-06-08]** 😴 **SkillOpt-Sleep is here — plugins for Claude Code, Codex, and Copilot.** Give your local coding agent a nightly *sleep cycle*: it reviews your past sessions offline, replays your recurring tasks, and consolidates validated long-term memory + skills behind a held-out gate, so it gets better the more you use it. Validated on the public [gbrain-evals](https://github.com/garrytan/gbrain-evals) `skillopt-v1` benchmark with **real Claude and Codex** (deficient skills 0.00 → 1.00 on held-out, all 4 seeds). It's an **open-source tool decoupled from the paper code**. See [`plugins/`](plugins/) and the [SkillOpt-Sleep section](#-skillopt-sleep--the-deployment-time-companion) below.
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
into bounded add / delete / replace edits on a single skill document; a
candidate edit is accepted only when it strictly improves a held-out
validation score. A textual learning-rate budget, a rejected-edit buffer,
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

## 😴 SkillOpt-Sleep — the deployment-time companion

SkillOpt (above) trains a skill offline on a benchmark. **SkillOpt-Sleep**
applies the same discipline to *your own daily usage*: it gives a local coding
agent a nightly **sleep cycle** that reviews your past sessions, replays your
recurring tasks on your own API budget, and consolidates what it learns into
**validated** long-term memory and skills — behind a held-out gate, staged for
your review. The agent gets better the more you use it, with no weight training.

It synthesizes **SkillOpt** (validation-gated bounded text edits), **Claude
Dreams** (offline consolidation; review-then-adopt), and the **agent sleep**
idea (short-term experience → long-term competence). One "night":

```
harvest session transcripts → mine recurring tasks → replay offline
   → consolidate (reflect → bounded edit → GATE on real held-out tasks)
   → stage proposal → (you) adopt
```

**Plugins for three agents** (one engine, three thin shells — see [`plugins/`](plugins/)):

| Platform | Folder | Install |
|---|---|---|
| **Claude Code** | [`plugins/claude-code`](plugins/claude-code) | `/plugin marketplace add ./plugins/claude-code` → `/sleep` |
| **Codex** | [`plugins/codex`](plugins/codex) | `bash plugins/codex/install.sh` → `skillopt-sleep` skill |
| **Copilot** | [`plugins/copilot`](plugins/copilot) | register `plugins/copilot/mcp_server.py` as an MCP server |

**Validated on real models.** On the public
[gbrain-evals](https://github.com/garrytan/gbrain-evals) `skillopt-v1` benchmark,
deficient skills go **0.00 → 1.00** on held-out sets with **both Claude and
Codex** (all 4 seeds, including a real tool-use loop), cross-model transfer is
positive, and the gate blocks regressions
([full results](docs/sleep/FINAL_REPORT.md)).

> **Open-source tool, decoupled from the research.** The engine lives in the
> top-level [`skillopt_sleep/`](skillopt_sleep) package with **zero dependency**
> on the paper's `skillopt/` experiment code (the validation gate is vendored).
> Controls — optional gate, multi-rollout contrastive reflection, token/time
> budget, multi-objective reward, user preferences, optimizer/target split — are
> documented in [`docs/sleep/CONTROLLABLE_DREAMING.md`](docs/sleep/CONTROLLABLE_DREAMING.md).

Deterministic proof (no API key): `python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves`.

---

## Extensibility & WebUI

### Adding a new backend

A backend = a chat / exec target (e.g. `openai_chat`, `claude_chat`,
`qwen_chat`, `minimax_chat`, `codex_exec`, `claude_code_exec`). See
[`docs/guide/new-backend.md`](docs/guide/new-backend.md) for the full
contract; in short you add a `skillopt/model/<name>_backend.py` module,
register it in `skillopt/model/common.py` + `backend_config.py`, and wire
it through the router in `skillopt/model/__init__.py`. `qwen_backend.py`
and `minimax_backend.py` are good templates.

### Adding a new benchmark

A benchmark = a `skillopt/envs/<name>/` package with a `dataloader.py`, a
`rollout.py`, and an `initial.md` seed skill. See
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

---

## Citation

```bibtex
@misc{yang2026skilloptexecutivestrategyselfevolving,
      title={SkillOpt: Executive Strategy for Self-Evolving Agent Skills}, 
      author={Yifan Yang and Ziyang Gong and Weiquan Huang and Qihao Yang and Ziwei Zhou and Zisu Huang and Yan Li and Xuemei Gao and Qi Dai and Bei Liu and Kai Qiu and Yuqing Yang and Dongdong Chen and Xue Yang and Chong Luo},
      year={2026},
      eprint={2605.23904},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.23904}
}
```

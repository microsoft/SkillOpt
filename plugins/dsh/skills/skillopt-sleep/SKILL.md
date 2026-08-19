---
name: skillopt-sleep
description: "Use when the user wants the dsh agent to self-improve from past usage, asks about a nightly/offline 'sleep' or 'dream' cycle, skill/memory consolidation, or says things like 'make my agent better the more I use it', 'review my past sessions', 'learn my preferences', 'consolidate what you learned', 'run the sleep cycle', or wants to schedule background self-optimization. Drives the skillopt_sleep engine through the skillopt_* tools: harvest past sessions -> mine recurring tasks -> replay via a selected backend -> consolidate validated skills behind a held-out gate."
---

# SkillOpt-Sleep：让 dsh 智能体从日常使用中自我进化

SkillOpt-Sleep 是微软 [SkillOpt](https://github.com/microsoft/SkillOpt) 的部署期伴生引擎：
它回顾你过去的会话（harvest），挖掘重复性任务（mine），用所选后端重放（replay），
并在**留出验证门（held-out gate）**之后把学到的内容沉淀为技能文档（consolidate）。

本技能通过 dsh-skillopt 插件暴露的 7 个 `skillopt_*` 工具驱动该引擎。默认 `mock`
后端不产生任何模型调用，可用于验证链路；真实后端才会消耗你的 API 预算。

## 什么时候用

- "让我的 agent 越用越强 / 从我的用法中学习 / 跨会话记住我的偏好"
- 要求一次**离线自我进化 / 睡眠 / 梦境**运行（即时或定时）
- 回顾过去的会话/轨迹，提炼重复任务
- 把反馈沉淀进 `AGENTS.md` / `SKILL.md` / 受管技能
- 定时（cron）运行该循环，或采纳（adopt）已暂存（staged）的提案

## 一个循环（六阶段）

1. **Harvest** — 只读读取支持的本地会话记录 → 会话摘要
2. **Mine** — 摘要 → 重复性任务记录（意图 + 结果标签 + 可校验引用）
3. **Replay** — 在当前技能+记忆下用所选后端重放任务 → (hard, soft) 分数
4. **Consolidate** — 反思失败 → 提出有界编辑 → 在留出集上**验证门控**，默认只在严格变好时接受
5. **Stage** — 把接受的提案写入 `<project>/.skillopt-sleep/staging/<timestamp>/`。
   **线上文件不变。** 被拒的运行仍有报告但没有提案文件。
6. **Adopt** — 显式（或 `--auto-adopt`）把暂存文件复制到线上文件，先备份。

## 怎么驱动

优先使用工具，而不是手工编辑文件：

| 工具 | 行为 |
|---|---|
| `skillopt_status` | 查看状态、引擎可用性、最新暂存提案与报告 |
| `skillopt_dry_run` | 完整预览循环（harvest+mine+replay），**不暂存任何东西** |
| `skillopt_run` | 跑完整循环并暂存提案（默认不改变线上文件） |
| `skillopt_adopt` | 应用最新暂存提案（先备份）——这是线上变更的边界 |
| `skillopt_harvest` | 只读查看/导出挖掘出的任务 |
| `skillopt_schedule` / `skillopt_unschedule` | 安装/移除本项目的夜间 cron 条目 |

典型用法：

```text
# 1. 先看状态（默认 mock 后端，零花费）
skillopt_status

# 2. 预览循环，确认任务挖掘是否合理
skillopt_dry_run project=<项目目录> source=<claude|codex|…>

# 3. 真实运行（消耗所选后端的 API 预算）
skillopt_run project=<项目目录> backend=<codex|claude|…> preferences="优先 pytest；提交信息用祈使句"

# 4. 用户审阅报告后，采纳提案
skillopt_adopt project=<项目目录>

# 5. 定时：每天凌晨 3:17 自动跑
skillopt_schedule project=<项目目录> hour=3 minute=17 backend=<codex>
```

## 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `project` | 配置或 cwd | 要进化的项目目录 |
| `backend` | `mock` | `mock\|claude\|codex\|copilot\|cursor\|pi\|opencode\|handoff\|azure_openai`（mock=不调用模型） |
| `source` | 配置 | 会话来源：`claude\|codex\|copilot\|cursor\|pi\|opencode\|auto` |
| `model` | 后端默认 | 重放模型覆盖 |
| `max_tasks` | 40 | 挖掘任务上限 |
| `preferences` | 空 | 注入反思先验的"家规"（如"总是用 async/await"） |

## 配置（cordis.yml / bundle patch）

```yaml
- insert:
    - id: skillopt
      name: './src/index.js'
      config:
        backend: codex
        project: /path/to/project
        preferences: 'Always use async/await'
```

高级引擎配置放 `~/.skillopt-sleep/config.json`：
`gate_mode`（on/off）、`gate_metric`（hard/soft/mixed）、`gate_no_regression`、
`dream_rollouts`、`recall_k`、`evolve_memory` / `evolve_skill` 等。

## 硬性规则

- **绝不**绕过 `skillopt_adopt` 手工改 `AGENTS.md` / `SKILL.md`；由引擎的 adopt
  或用户要求的 `--auto-adopt` 应用暂存清单，并先备份线上文件。
- Harvest 只读。`mock` 重放无副作用。
- 真实后端会把截断的会话摘录与派生任务发给所选提供方做挖掘/重放/评判/反思。
  敏感会话请先用 `skillopt_harvest output=任务文件` 导出，人工审查脱敏并把
  顶层 `"reviewed"` 置为 `true` 后，再用 `--tasks-file` 重放；真实后端会拒绝未审阅的任务文件。
- 建议采纳前，把 **留出基线 → 候选** 分数和确切编辑内容展示给用户。先有证据再采纳。

## 验证 / 演示（无 API 花费）

```bash
pip install skillopt
python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves
```

确定性合成演示：分数上升、门控阻止回退。验证的是机制本身，不代表在你任务上的真实效果。

更多信息：[SkillOpt-Sleep 文档](https://github.com/microsoft/SkillOpt/tree/main/docs/sleep)

# dsh-skillopt 文档

## 快速上手

1. 安装引擎：`pip install skillopt`（或克隆 [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt) 并把其根目录加入 `PYTHONPATH`）
2. 在 profile 的 `cordis.patch.yml` 插入插件（见根 README）
3. 启动 dsh 后向 agent 提问："用 skillopt_status 查看睡眠循环状态"

## 工具与引擎命令对照

| dsh 工具 | skillopt_sleep 动作 | 说明 |
|---|---|---|
| `skillopt_status` | `status` | 状态与暂存提案 |
| `skillopt_dry_run` | `dry-run` | 预览，不暂存 |
| `skillopt_run` | `run` | 完整循环并暂存 |
| `skillopt_adopt` | `adopt` | 应用提案（先备份） |
| `skillopt_harvest` | `harvest` | 只读导出任务 |
| `skillopt_schedule` | `schedule` | 安装夜间 cron |
| `skillopt_unschedule` | `unschedule` | 移除 cron |

## 引擎进阶配置（`~/.skillopt-sleep/config.json`）

```json
{
  "gate_mode": "on",
  "gate_metric": "mixed",
  "gate_no_regression": false,
  "dream_rollouts": 1,
  "recall_k": 0,
  "evolve_memory": true,
  "evolve_skill": true,
  "preferences": "Prefer pytest. Keep commits imperative."
}
```

详见上游文档：https://github.com/microsoft/SkillOpt/tree/main/docs/sleep

# AppWorld

[AppWorld](https://github.com/StonyBrookNLP/appworld) is a benchmark of everyday tasks: an agent writes Python against 9 simulated apps (Spotify, Venmo, a file system, a phone, and others) behind 457 APIs. Each task is graded by the benchmark's own state-based unit tests.

SkillOpt's AppWorld environment runs the official `react_code` protocol:

- AppWorld's few-shot ReAct prompt.
- One `execute_code` tool.
- A budget of 50 executed tool calls per task.

The skill is appended to the system prompt under `## Skill Knowledge`. An empty skill sends the official prompt unchanged.

The prompt's placeholders are filled the way AppWorld's own `react_code` agent fills them: the task instruction, the supervisor's name, email and phone, and `{{ app_descriptions }}` (the app list, read once from the AppWorld venv at startup; a venv that cannot produce it stops the run).

## Install

AppWorld's dependencies conflict with SkillOpt's, so install AppWorld into its own virtual environment. SkillOpt starts `appworld serve environment` processes from that venv and talks to them over HTTP.

```bash
# 1. AppWorld in its own venv
python -m venv ~/venvs/appworld
~/venvs/appworld/bin/pip install appworld   # or `pip install -e <clone>` from source
~/venvs/appworld/bin/appworld install
~/venvs/appworld/bin/appworld download data --root ~/.appworld-roots/skillopt-appworld

# 2. Tell SkillOpt where it is
export APPWORLD_VENV=~/venvs/appworld      # or env.appworld_venv in the config
```

`env.appworld_root` (default `~/.appworld-roots/skillopt-appworld`) must contain the downloaded `data/` directory.

## Build the split

Task instructions stay in your local AppWorld data; SkillOpt does not ship them. Build the split once:

```bash
python -m skillopt.envs.appworld.make_split \
    --appworld_root ~/.appworld-roots/skillopt-appworld --out data/appworld_split_full
```

| Split | AppWorld source | Tasks | Use |
|---|---|---|---|
| train | `train` | 90 | rollouts the optimizer reflects on |
| val | `dev` | 57 | selection gate and Best-skill choice only |
| test | `test_normal` + `test_challenge` | 168 + 417 = 585 | final report only |

`--test_set scen1` builds a cheaper 195-task test split instead. It takes one task per scenario, using the lists in `skillopt/envs/appworld/splits/`.

## Run

The model endpoint is any OpenAI-compatible server (vLLM, SGLang, or a hosted gateway) set through `QWEN_CHAT_BASE_URL`, plus `QWEN_CHAT_API_KEY` if the server needs one.

```bash
export QWEN_CHAT_BASE_URL=http://<host>:<port>/v1
python scripts/train.py --config configs/appworld/qwen38_nothink_gated.yaml \
    --out_root outputs/appworld_gated

python scripts/eval_only.py --config configs/appworld/qwen38_nothink_gated.yaml \
    --skill outputs/appworld_gated/best_skill.md --split valid_unseen   # valid_unseen = test split
```

| Config | Slow update | Per-step gate |
|---|---|---|
| `qwen38_nothink_gated.yaml` | candidate must pass the selection gate (paper setting) | on |
| `qwen38_nothink_force.yaml` | written unconditionally | on |
| `qwen38_nothink_noslow.yaml` | off (meta skill stays on) | on |
| `qwen38_nothink_accept_all.yaml` | written unconditionally | off (every candidate accepted; Best still tracked) |

Each episode needs one AppWorld server, and `env.workers` sets how many run at once. Servers bind free ports in `[env.port_base, env.port_base + env.port_span)`. Give every SkillOpt process that runs at the same time its own range.

## How tasks are scored and what the optimizer sees

- **Official interface only.** SkillOpt talks to AppWorld only through the `appworld serve environment` routes `/initialize` (with `load_ground_truth=False`), `/execute`, `/task_completed` and `/evaluate`.
  - `hard` is AppWorld's task goal completion: all of the task's tests pass.
  - `soft` is the fraction of tests that pass.
  - An episode that crashes on infrastructure is kept and counted as a failure.
- **Failure feedback.** For a failed train task, the analysts and the slow update see up to 8 of the failed test descriptions from `/evaluate` (for example "assert answers match"). This feedback comes only from train rollouts. Val is used only for its score, and test is never seen during training. Set `env.fail_reason_style`:
  - `checks` (default): leads with the failed tests. It names the stop reason only when the stop reason itself explains the failure: the call budget ran out, a generation or context limit was hit, or `complete_task` never succeeded.
  - `legacy`: opens every failure with the stop reason. `no_tool_call` is how every AppWorld episode normally ends, so naming it as the outcome of a failure can reinforce a mistaken "emit no text after `complete_task`" rule once such a rule is in the skill. Use `legacy` only to reproduce older runs.

## Results

These are single runs with Qwen3.8-27B (FP8) as both target and optimizer, thinking off, temperature 0.7. Training is 4 epochs × 3 steps (batch 40). Scores are test success rates on all 585 tasks.

| Skill | slow update (Gated) | slow update (Force-accept) | slow update (turned off) | Accept Every Edit |
|---|---|---|---|---|
| No skill | 68.0% (398/585) | 68.0% (398/585) | 68.0% (398/585) | 68.0% (398/585) |
| Best skill | 84.6% (495/585) | 86.7% (507/585) | 75.7% (443/585) | 86.5% (506/585) |
| Last skill | 84.6% (495/585) | 86.3% (505/585) | 75.7% (443/585) | 87.5% (512/585) |

Notes:

- **Prompt and failure text.** These runs were made with an earlier version of this environment, which left `{{ app_descriptions }}` unfilled: the few-shot example showed that literal text where the app list should be. They also used `fail_reason_style: legacy`. This code fills the app list, and its default failure text is `checks`, so expect scores to differ.
- **Code versions.** The Gated, turned-off and Accept-Every-Edit runs predate #408, and the Force-accept run includes #408 and #422.
- **Ties.** Where Last equals Best, it is the same skill and was tested once.
- **Infrastructure failures.** They count as failures. This affected 3 tasks in Accept Every Edit / Best.
- **Noise and dev saturation.**
  - Each cell is one run. Differences of a few points are within run-to-run variation; re-testing an identical skill moved 4B results by about ±2 points.
  - The 57-task val split saturates at this model size: most candidates score 89–98%. The Best choice and gate decisions therefore hinge on one or two val tasks.
  - The turned-off run reached 56/57 at step 3, after which only a perfect 57/57 could replace it under the strict `>` gate.

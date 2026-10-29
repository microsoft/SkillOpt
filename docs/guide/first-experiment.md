# Your First Experiment

This guide walks through running a complete SkillOpt training on SearchQA.

## 1. Choose a Benchmark

SkillOpt includes ready-to-use configs for several benchmarks. End-to-end
runtime depends on the chosen models, provider latency, worker limits, and
dataset size, so the project does not promise fixed wall-clock estimates.

| Benchmark | Modality | Additional setup |
|---|---|---|
| SearchQA | Text QA | Materialize the released ID manifest |
| DocVQA | Document/image QA | Obtain and materialize images and examples |
| ALFWorld | Embodied agent | Install ALFWorld and download its assets |

We'll use **SearchQA** because it is the simplest text-only walkthrough.

## 2. Install and Materialize SearchQA

The repository contains a stable SearchQA ID manifest, not the full runnable
examples. From a source checkout, install the data extra and materialize the
split once:

```bash
python -m pip install -e ".[searchqa]"
python scripts/materialize_searchqa.py
```

By default, the materializer reads `data/searchqa_id_split/` and writes the
train/validation/test payloads expected by the config to
`data/searchqa_split/`; both paths have command-line overrides.

## 3. Configure

Configure and export one model backend as described in
[Installation](installation.md#environment-variables). For example:

```bash
cp .env.example .env
# Edit .env, choose one authentication mode, then export it:
set -a; source .env; set +a
```

Review the config file:

```bash
cat configs/searchqa/default.yaml
```

Key parameters (deep learning analogy in parentheses):

```yaml
train:
  num_epochs: 4           # (epochs)
  batch_size: 40          # (batch size)

optimizer:
  learning_rate: 4        # (max edits per step)
  lr_scheduler: cosine    # (learning rate schedule)
  use_slow_update: true   # (momentum at epoch boundary)
  use_meta_skill: true    # (cross-epoch optimizer memory)

gradient:
  analyst_workers: 16     # (parallel reflection workers)

evaluation:
  use_gate: true          # (validation gating)
```

## 4. Train

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --out_root outputs/searchqa_first_run
```

The command prints the resolved backend/data configuration, per-step rollout
and gate progress, and the generated output directory.

## 5. Monitor

The explicit `--out_root` above creates this run directory:

```
outputs/searchqa_first_run/
├── config.json
├── runtime_state.json
├── history.json
├── best_skill.md
├── skills/
│   └── skill_vXXXX.md
├── steps/
│   └── step_XXXX/
│       ├── candidate_skill.md
│       ├── step_record.json
│       └── trajectory_digest.json
├── slow_update/
│   └── epoch_XX/
└── meta_skill/
    └── epoch_XX/
```

## 6. Evaluate

Evaluate the best skill on the test split:

```bash
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa_first_run/best_skill.md \
  --split valid_unseen
```

The `--skill` path above is the training artifact. Evaluation writes
`eval_summary.json` to its own timestamped `outputs/eval_.../` directory unless
you pass an explicit `--out_root`; it does not overwrite the training run.

## WebUI

Prefer a graphical interface? Launch the WebUI:

```bash
pip install -e ".[webui]"
python -m skillopt_webui.app
```

Then open `http://localhost:7860` in your browser to configure parameters and
launch training. The default host is `127.0.0.1`; exposing it with `--host 0.0.0.0`
requires `--auth-user`/`--auth-pass`.

## Next Steps

- [Understand the training loop](training-loop.md)
- [Configuration reference](../reference/config.md)
- [Add a new benchmark](new-benchmark.md)

# CLI Reference

## Gitmoot Optimizer

```bash
gitmoot-skillopt optimize \
  --training-package training.json \
  --artifact-root ~/.gitmoot/evals/blobs \
  --out-root outputs/run-1 \
  --candidate-output outputs/run-1/candidate.json
```

### Arguments

| Argument | Description |
|---|---|
| `--training-package` | Gitmoot SkillOpt training package from `gitmoot skillopt export` |
| `--artifact-root` | Gitmoot blob root, usually `~/.gitmoot/evals/blobs` |
| `--out-root` | Optimizer output directory |
| `--candidate-output` | Candidate package JSON path to import back into Gitmoot |
| `--dry-run` | Emit deterministic fixture output without trainer/model calls |

### Contract Smoke

```bash
gitmoot-skillopt optimize \
  --training-package examples/gitmoot/mvp-fixture/training.json \
  --artifact-root examples/gitmoot/mvp-fixture/blobs \
  --out-root /tmp/gitmoot-skillopt-smoke \
  --candidate-output /tmp/gitmoot-skillopt-smoke/candidate.json \
  --dry-run
```

Import the generated candidate with:

```bash
gitmoot skillopt import \
  --file /tmp/gitmoot-skillopt-smoke/candidate.json \
  --artifact-dir /tmp/gitmoot-skillopt-smoke/artifacts
```

## Training

```bash
python scripts/train.py --config <config.yaml> [overrides...]
```

### Arguments

| Argument | Description |
|---|---|
| `--config` | Path to YAML config file (required) |
| `key=value` | Override any config parameter |

### Examples

```bash
# Basic training
python scripts/train.py --config configs/searchqa/default.yaml

# With overrides
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options optimizer.learning_rate=16 optimizer.lr_scheduler=linear

# With custom initial skill
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options env.skill_init=skills/my_seed.md
```

## Evaluation

```bash
python scripts/eval_only.py --config <config.yaml> --skill <skill.md>
```

### Arguments

| Argument | Description |
|---|---|
| `--config` | Path to YAML config file (required) |
| `--skill` | Path to skill document to evaluate (required) |
| `--split` | Evaluation split: `test` (default), `valid`, `train` |

### Examples

```bash
# Evaluate best skill on test set
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa/run_001/skills/best_skill.md

# Evaluate on validation set
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa/run_001/skills/best_skill.md \
  --split valid
```

## WebUI

```bash
python -m skillopt_webui.app [--port PORT] [--share]
```

| Argument | Default | Description |
|---|---|---|
| `--port` | 7860 | Port number |
| `--share` | false | Create public Gradio link |

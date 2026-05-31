# Paper-aligned SkillOpt reference skills (GPT-5.5)

This folder provides a subset of the paper's main Table 1 GPT-5.5 optimized
skills as reference artifacts — one `gpt5.5_skill.md` per currently included
benchmark. You can plug them into `scripts/eval_only.py` to evaluate the
provided skills on a given split without re-running the training loop.

> These are checkpoints associated with the paper, not a general-purpose
> tool. They're here so you can verify the reported numbers and use the
> skills as portable artifacts. If you want to *train* your own skill,
> use `scripts/train.py` per the top-level README.
>
> This is the first artifact batch. We plan to continue uploading the
> remaining optimized skills and benchmark split manifests as they are
> cleaned and verified.

## What's here

| Benchmark | Skill artifact | Matching config |
|---|---|---|
| SearchQA | `ckpt/searchqa/gpt5.5_skill.md` | `configs/searchqa/default.yaml` |
| ALFWorld | `ckpt/alfworld/gpt5.5_skill.md` | `configs/alfworld/default.yaml` |
| DocVQA | `ckpt/docvqa/gpt5.5_skill.md` | `configs/docvqa/default.yaml` |
| LiveMathematicianBench | `ckpt/livemath/gpt5.5_skill.md` | `configs/livemathematicianbench/default.yaml` |
| OfficeQA | `ckpt/officeqa/gpt5.5_skill.md` | `configs/officeqa/default.yaml` |
| SpreadsheetBench | `ckpt/spreadsheetbench/gpt5.5_skill.md` | `configs/spreadsheetbench/default.yaml` |

Each file is a plain Markdown skill document (~2k–13k chars). It contains a
protected `SLOW_UPDATE` section at the end that holds epoch-wise
longitudinal guidance — that's expected, not a formatting issue.

## How to evaluate a provided skill

`scripts/eval_only.py` runs a single skill against a data split without
invoking the optimizer. Example for SearchQA against the test split:

```bash
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill ckpt/searchqa/gpt5.5_skill.md \
  --split valid_unseen \
  --split_dir data/searchqa_id_split \
  --azure_openai_endpoint https://your-resource.openai.azure.com/ \
  --target_model gpt-5.5
```

Substitute the benchmark, config, skill path, and `--split_dir` to evaluate
any of the other five. `--split valid_unseen` is the test split, `valid_seen`
is the selection / validation split, `train` is the training split, and
`all` runs all three.

## On comparing to the paper numbers

To compare against the paper-reported cells, use the same dataset split and
scorer. SearchQA's split is checked in at `data/searchqa_id_split/` (400
train / 200 selection / 1400 test). For the other benchmarks, point
`--split_dir` at your own materialized split; the loader is deterministic
from `split_seed` (default `42`) + `split_ratio` (default `2:1:7`) when
`split_mode: ratio` is used, so a given `data_path` + seed reproduces
across machines. Explicit per-benchmark split manifests are being prepared
for upload — see issues #14 and #21.

## Why force-accept vs. gated slow-update matters

These `ckpt/` skills were produced with the gated slow-update semantics
described in paper Section 3.6:

```yaml
optimizer:
  slow_update_gate_with_selection: true
```

Current `main` defaults to `false` (force-accept mode), a newer
post-submission behavior where the slow-update guidance is written into
`current_skill` and `best_skill` unconditionally at the epoch boundary. If
you re-train with the current default, you may produce a *different*
`best_skill.md` than the one checked in here. Both modes are supported;
see the top-level README's "Configuration -> Slow-update acceptance mode"
section.

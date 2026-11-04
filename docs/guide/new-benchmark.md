# Add a New Benchmark

Extend SkillOpt with your own benchmark in ~200 lines of code. We will use
a tiny worked example, `docfaithful`, that scores a target model on
how faithfully it answers questions grounded in a small reference doc.

> **Working reference.** The easiest way to copy-cargo-cult a new env is
> to read [`skillopt/envs/officeqa/`](https://github.com/microsoft/SkillOpt/tree/main/skillopt/envs/officeqa).
> Everything below is the same shape, simplified.

## What you need to build

To add a benchmark you implement four things:

1. **A `SplitDataLoader` subclass** — knows how to load train / val / test
   item dicts from disk.
2. **A rollout helper** — runs the target model on a batch of items, scores
   each prediction, and persists the per-item conversation consumed by the
   shared reflection stage.
3. **An `EnvAdapter` subclass** — wires the loader + rollout helper into
   SkillOpt's lifecycle (`build_*_env`, `rollout`, and `get_task_types`).
   The shared `reflect()` implementation is inherited unless the benchmark
   needs custom reflection logic.
4. **A YAML config** — references your env name plus the standard
   train / optimizer / gradient knobs.

Then lazy registration in the training and evaluation scripts makes it
discoverable without importing optional dependencies at startup.

---

## Step 1 — Create the package

```bash
mkdir -p skillopt/envs/docfaithful
touch skillopt/envs/docfaithful/__init__.py
```

## Step 2 — Implement the data loader

`skillopt/envs/docfaithful/dataloader.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


def _normalize(raw: dict) -> dict:
    """Make sure every item has an ``id``. Other keys are env-specific."""
    return {
        "id": str(raw["uid"]),
        "question": raw["question"],
        "ground_truth": raw["answer"],
        "reference_text": raw.get("reference", ""),
        "task_type": raw.get("category", "docfaithful"),
    }


class DocFaithfulDataLoader(SplitDataLoader):
    """Load DocFaithful items from JSON files inside each split dir."""

    def load_split_items(self, split_path: str) -> list[dict]:
        # split_path is e.g. data/docfaithful_split/train/
        json_files = sorted(Path(split_path).glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json file found in {split_path}")
        with json_files[0].open(encoding="utf-8") as f:
            raw = json.load(f)
        return [_normalize(item) for item in raw]
```

Only `load_split_items()` is mandatory. If you also want to support
`split_mode="ratio"` (auto-split a single raw file into train/val/test),
override `load_raw_items(data_path)` as well — see
`skillopt/datasets/base.py` docstrings.

## Step 3 — Write the rollout helper

`skillopt/envs/docfaithful/rollout.py`:

```python
from __future__ import annotations

import json
import os
from pathlib import Path

from skillopt.model import chat_target


def _score(prediction: str, ground_truth: str) -> tuple[int, float]:
    """Trivial exact-match scorer. Replace with F1 / ROUGE / LLM-judge."""
    p = (prediction or "").strip().lower()
    g = (ground_truth or "").strip().lower()
    hard = int(p == g and bool(g))
    soft = 1.0 if hard else 0.0
    return hard, soft


def _rollout_one(item: dict, skill_content: str, *, prediction_dir: Path,
                 max_completion_tokens: int) -> dict:
    system = skill_content
    user = (
        f"Question: {item['question']}\n\n"
        f"Reference:\n{item.get('reference_text', '')}\n\n"
        "Answer:"
    )
    prediction, _usage = chat_target(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
    )
    hard, soft = _score(prediction, item.get("ground_truth", ""))

    # EnvAdapter.reflect() reads this exact trajectory path. Keep item IDs
    # unique and filesystem-safe.
    task_dir = prediction_dir / str(item["id"])
    task_dir.mkdir(parents=True, exist_ok=True)
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": prediction},
    ]
    (task_dir / "conversation.json").write_text(
        json.dumps(conversation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "id": str(item["id"]),
        "hard": hard,
        "soft": soft,
        "predicted_answer": prediction,
        "task_description": item.get("question", ""),
        "question": item.get("question", ""),
        "reference_text": item.get("reference_text", ""),
        "task_type": item.get("task_type", "docfaithful"),
        "target_system_prompt": system,
        "target_user_prompt": user,
        "n_turns": 1,
    }


def run_batch(*, items: list[dict], skill_content: str, out_root: str,
              workers: int = 4, max_completion_tokens: int = 4096) -> list[dict]:
    """Run a batch of episodes sequentially or with a thread pool."""
    os.makedirs(out_root, exist_ok=True)
    prediction_dir = Path(out_root, "predictions")
    # For brevity we go sequentially — swap in concurrent.futures.ThreadPoolExecutor
    # when network / model latency dominates.
    results = [
        _rollout_one(item, skill_content,
                     prediction_dir=prediction_dir,
                     max_completion_tokens=max_completion_tokens)
        for item in items
    ]
    Path(out_root, "rollouts.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results
```

Two design points worth flagging:

- **Scoring lives here, not in `EnvAdapter`.** There is no `evaluate()`
  method on the ABC. Whatever signal you put in `hard` (0/1, or a float
  in [0, 1] for smoothed reward) and `soft` (float in [0, 1]) is what
  the optimizer reads.
- **Use `skillopt.model.chat_target`**, not raw OpenAI/Claude calls.
  That routes through whichever **chat** target backend the user
  configured (`openai_chat` / `claude_chat` / `qwen_chat` /
  `minimax_chat` / `openai_compatible`) without your adapter caring.
  Exec-style backends (`codex_exec`, `claude_code_exec`) need
  environment-specific rollout code —
  see `skillopt/model/codex_harness.py` together with the rollout modules in
  `skillopt/envs/searchqa/`, `skillopt/envs/docvqa/`, or
  `skillopt/envs/officeqa/` for working examples.
- **Persist a conversation for reflection.** The shared `EnvAdapter.reflect()`
  reads `conversation.json` from the directory that
  `confined_task_output_dir(out_root, id, map_unsafe=True)` in
  `skillopt/envs/task_output.py` returns:
  `<rollout_dir>/predictions/<result-id>/` for a filesystem-safe id (the case
  the example above assumes), otherwise
  `<rollout_dir>/predictions/_mapped-<sha256 of id>/`. It falls back to a
  raw-id directory only when that stays inside `predictions/`, and skips
  results whose trajectory is absent or empty. Returning `hard`/`soft` scores
  alone is sufficient for evaluation, but it cannot produce learning patches.

## Step 4 — Implement the environment adapter

`skillopt/envs/docfaithful/adapter.py`:

```python
from __future__ import annotations

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.docfaithful.dataloader import DocFaithfulDataLoader
from skillopt.envs.docfaithful.rollout import run_batch


class DocFaithfulAdapter(EnvAdapter):
    """SkillOpt adapter for the DocFaithful benchmark."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        workers: int = 4,
        analyst_workers: int = 4,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        max_completion_tokens: int = 4096,
    ) -> None:
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.max_completion_tokens = int(max_completion_tokens)
        self.dataloader = DocFaithfulDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    # ── Lifecycle ───────────────────────────────────────────────────────

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    # ── Env construction ────────────────────────────────────────────────

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        # For dataset-backed envs the "manager" is just the items list.
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(
            batch_size=batch_size, seed=seed, **kwargs
        )
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(
            env_num=env_num, split=split, seed=seed, **kwargs
        )
        return self.build_env_from_batch(batch, **kwargs)

    # ── The rollout method (reflect is inherited) ───────────────────────

    def rollout(self, env_manager, skill_content: str,
                out_dir: str, **kwargs) -> list[dict]:
        items: list[dict] = env_manager
        return run_batch(
            items=items,
            skill_content=skill_content,
            out_root=out_dir,
            workers=self.workers,
            max_completion_tokens=self.max_completion_tokens,
        )

    # reflect() is inherited from EnvAdapter — it delegates to
    # run_minibatch_reflect with your analyst_error_* / analyst_success_*
    # prompts. Override it only if you need custom reflection logic.

    def get_task_types(self) -> list[str]:
        seen: list[str] = []
        for item in (
            self.dataloader.train_items
            + self.dataloader.val_items
            + self.dataloader.test_items
        ):
            tt = str(item.get("task_type") or "docfaithful")
            if tt not in seen:
                seen.append(tt)
        return seen or ["docfaithful"]
```

### What the rollout actually does

Look back at `run_batch` from Step 3 — it sends each `item["question"]`
to the target model with `skill_content` as the system prompt, scores
the answer against `item["ground_truth"]`, and returns a list of dicts:

```python
[
    {"id": "ex_001", "hard": 1, "soft": 0.92,
     "predicted_answer": "...", "question": "...",
     "reference_text": item["reference_text"]},
    {"id": "ex_002", "hard": 0, "soft": 0.13, "fail_reason": "...", ...},
    ...
]
```

The trainer requires `id`, `hard`, and `soft` for scoring. The remaining fields
are preserved on `RolloutResult.extras` (see `skillopt/types.py`). The shared
reflection implementation combines those fields with each persisted
`conversation.json` (in `predictions/<id>/`, or
`predictions/_mapped-<sha256 of id>/` for an id that is not filesystem-safe;
see Step 3); without that file the result is omitted from reflection.

## Step 5 — Register the adapter

Edit [`scripts/train.py`](https://github.com/microsoft/SkillOpt/blob/main/scripts/train.py)
and add to `_register_builtins()`:

```python
    try:
        from skillopt.envs.docfaithful.adapter import DocFaithfulAdapter
        _ENV_REGISTRY["docfaithful"] = DocFaithfulAdapter
    except ImportError:
        pass  # docfaithful deps not installed — skip
```

Mirror the same lazy registration in
[`scripts/eval_only.py`](https://github.com/microsoft/SkillOpt/blob/main/scripts/eval_only.py)
so standalone evaluation can resolve the environment too. There is **no
`BENCHMARK_REGISTRY` dict in `skillopt/envs/__init__.py`**; both entry points
keep a small lazy registry so optional dependencies do not break `--help`.

## Step 6 — Create the YAML config

`configs/docfaithful/default.yaml`:

```yaml
_base_: ../_base_/default.yaml      # NOTE: string, not list

model:
  reasoning_effort: medium

train:
  batch_size: 16
  accumulation: 1
  num_epochs: 4

gradient:
  minibatch_size: 8
  merge_batch_size: 8

optimizer:
  learning_rate: 4

env:
  name: docfaithful
  # Point to an existing Markdown file. Use an empty file to start blank.
  skill_init: skillopt/envs/docfaithful/skills/initial.md
  split_mode: split_dir
  split_dir: data/docfaithful_split
  workers: 4
  max_completion_tokens: 4096
  limit: 0
```

> ⚠️ `_base_` is currently parsed as a **string path**, not a list. Write
> `_base_: ../_base_/default.yaml`, not `_base_: ['../_base_/default.yaml']`.
> See [`skillopt/config.py`](https://github.com/microsoft/SkillOpt/blob/main/skillopt/config.py)
> if you want to add list-form inheritance.

## Step 7 — Run

```bash
# Create the file referenced by env.skill_init before the first run:
#   mkdir -p skillopt/envs/docfaithful/skills
#   echo "# DocFaithful initial skill" > skillopt/envs/docfaithful/skills/initial.md

python scripts/train.py --config configs/docfaithful/default.yaml
```

If you get `ValueError: Unknown environment 'docfaithful'. Available: [...]`,
you forgot Step 5.

If you get `TypeError: Can't instantiate abstract class DocFaithfulAdapter`,
you forgot to implement one of the four abstract methods on `EnvAdapter`:
`build_train_env`, `build_eval_env`, `rollout`, `get_task_types`.

## Tips

- Start with `train.batch_size: 4` and `limit: 10` while debugging.
- The `evaluate` half lives **inside your `rollout`**, not as a separate
  method — there is no `evaluate()` in the `EnvAdapter` ABC. Score the
  prediction in `run_batch` and put the score on each result dict's
  `hard` / `soft`.
- Noisy scoring kills the optimizer. Spend time on `run_batch`'s scoring
  before you spend time on prompts.
- If training repeatedly reports `skip_no_patches`, first verify that every
  rollout result has a non-empty `conversation.json` in
  `rollout/predictions/<segment>/`, where `<segment>` is
  `task_output_segment(id, map_unsafe=True)` for the result's `id` string:
  the id itself when it is filesystem-safe, `_mapped-<sha256 of id>`
  otherwise.
- If your benchmark needs heavy optional deps (selenium, vllm, ...),
  wrap both registration blocks with `try / except ImportError` (Step 5)
  so people without those deps can still `--help`.
- Copy `skillopt/envs/_template/` as a starting skeleton — it now
  implements the real abstract methods.

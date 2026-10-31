# API Reference

This page documents the public Python API SkillOpt exposes for **extending the
framework** with new environments / benchmarks. For ready-made adapters,
browse [`skillopt/envs/`](https://github.com/microsoft/SkillOpt/tree/main/skillopt/envs).

> **Source of truth.** The classes below are real Python ABCs defined in
> `skillopt/envs/base.py`, `skillopt/datasets/base.py`, `skillopt/types.py`,
> and `skillopt/evaluation/gate.py`. If this page ever drifts, the code
> wins — please open an issue.

---

## Core Classes

### `EnvAdapter`

`skillopt/envs/base.py` — abstract adapter that connects the SkillOpt
trainer to an environment (benchmark, simulator, REST API, ...).
Subclasses **must** implement the four abstract methods below. Reflection has a
shared default implementation and only needs to be overridden for
environment-specific behavior.

```python
from abc import ABC, abstractmethod
from skillopt.datasets.base import BaseDataLoader, BatchSpec

class EnvAdapter(ABC):

    # ── Lifecycle hooks (have defaults; override only if needed) ────────

    def setup(self, cfg: dict) -> None: ...
    def get_dataloader(self) -> BaseDataLoader | None: ...
    def requires_ray(self) -> bool: ...                 # default False
    def reflect(self, results: list[dict], skill_content: str,
                out_dir: str, **kwargs) -> list[dict | None]:
        """Delegate to the shared minibatch reflection pipeline."""
        ...

    # ── Abstract methods (subclasses MUST implement) ────────────────────

    @abstractmethod
    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        """Return an environment-manager object to be passed to rollout()."""

    @abstractmethod
    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        """Like build_train_env() but for a fixed eval split."""

    @abstractmethod
    def rollout(self, env_manager, skill_content: str,
                out_dir: str, **kwargs) -> list[dict]:
        """Run a batch of episodes with the current skill.

        Each returned dict MUST contain:
          - "id":   str        episode/task identifier
          - "hard": int (0|1)  pass/fail (may be float 0.0-1.0 if smoothed)
          - "soft": float      partial-credit score in [0.0, 1.0]
        It MAY contain env-specific extra keys (parsed into RolloutResult.extras).
        """

    @abstractmethod
    def get_task_types(self) -> list[str]:
        """Distinct task-type strings used for stratified sampling."""
```

The default `reflect()` delegates to `run_minibatch_reflect` and returns raw
patch dicts with a `patch` payload plus a `failure` or `success` source type.
It expects each rollout to persist a non-empty trajectory at
`<rollout_dir>/predictions/<result-id>/conversation.json`; results without that
file can be scored but are skipped during reflection.
The trainer also calls several default-implemented helpers on every adapter:
`build_reference_text`, `get_reference_metadata`, `attach_reference_context`,
`select_representative_items`, and `build_env_from_batch`. Read the docstrings
in `skillopt/envs/base.py` if you need to override any of these — most
benchmarks do not.

### `BaseDataLoader` / `SplitDataLoader`

`skillopt/datasets/base.py` — episode-planning loaders.

```python
class BaseDataLoader(ABC):
    def setup(self, cfg: dict) -> None: ...
    @abstractmethod
    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec: ...
    @abstractmethod
    def build_eval_batch(self, env_num: int, split: str, seed: int, **kwargs) -> BatchSpec: ...

class SplitDataLoader(BaseDataLoader):
    """Concrete base for dataset-backed envs with on-disk train/val/test splits.

    Subclasses only need to implement load_split_items() (and optionally
    load_raw_items() if you also want ``split_mode='ratio'``).
    """
    def load_split_items(self, split_path: str) -> list[dict]: ...
    def load_raw_items(self, data_path: str) -> list[dict]: ...   # optional
```

`SplitDataLoader` handles two layout modes:

| `split_mode` | What it expects |
|---|---|
| `"split_dir"` | A directory with `train/`, `val/`, `test/` subdirs already split. |
| `"ratio"` | A raw dataset path + `split_ratio: "2:1:7"` style string. |

In either case the items returned by `load_split_items()` are plain
`dict` objects with at minimum an `"id"` key.

### `BatchSpec`

`skillopt/datasets/base.py` — a slotted dataclass describing one batch
request the trainer hands to the adapter.

```python
@dataclass(slots=True)
class BatchSpec:
    phase: str                 # "train" | "eval"
    split: str                 # "train" | "val" | "test" | "valid_seen" | ...
    seed: int
    batch_size: int
    payload: object | None = None     # what the loader produced (e.g. list[dict])
    metadata: dict = field(default_factory=dict)
```

### `Edit` / `Patch`

`skillopt/types.py` — the I/O types Reflect / Aggregate / Update produce
and consume.

```python
EditOp = Literal["append", "insert_after", "replace", "delete"]

@dataclass
class Edit:
    op: EditOp
    content: str = ""
    target: str = ""
    support_count: int | None = None
    source_type: Literal["failure", "success"] | None = None
    merge_level: int | None = None
    update_origin: str = ""
    update_target: str = ""

@dataclass
class Patch:
    edits: list[Edit] = field(default_factory=list)
    reasoning: str = ""
    ranking_details: dict[str, Any] | None = None
```

Both types support `to_dict()` / `from_dict()` for serialization.

### `RolloutResult`

`skillopt/types.py` — the normalised rollout return type. The trainer
calls `RolloutResult.from_dict(...)` on each dict returned from
`EnvAdapter.rollout()`, so the only **hard** requirement on those dicts is
the three keys above (`id`, `hard`, `soft`). Extra fields are preserved
into `RolloutResult.extras`.

### `GateResult` / `GateAction`

`skillopt/evaluation/gate.py` — the validation-gate decision types
returned for each candidate optimization step, and optionally for a separate
epoch-end slow-update candidate.

---

## Registering an environment

Environments are not registered via decorators or a `BENCHMARK_REGISTRY`
dict. The training and standalone-evaluation entry points each keep a lazy
`_ENV_REGISTRY`, populated by `_register_builtins()` in `scripts/train.py` and
`scripts/eval_only.py`. Add the environment to both entry points. See
[Add a New Benchmark](../guide/new-benchmark.md) for the full step-by-step.

---

## Backends (model layer)

The model layer lives under `skillopt.model.*`. Backends are selected
via `model.optimizer_backend` and `model.target_backend` in the config —
not via a base class subclass. Supported values (as of this writing):

| Backend | Optimizer? | Target? |
|---|---|---|
| `openai_chat` | ✓ | ✓ |
| `claude_chat` | ✓ | ✓ |
| `qwen_chat` | ✓ | ✓ |
| `minimax_chat` | ✓ | ✓ |
| `copilot_chat` | ✓ | ✓ |
| `openai_compatible` | ✓ | ✓ |
| `codex_exec` | ✓ | ✓ |
| `claude_code_exec` | — | ✓ |
| `cursor_exec` | — | ✓ |
| `copilot_exec` | — | ✓ |
| `custom` (function from `set_optimizer_chat()`) | ✓ | — |

See `skillopt/model/backend_config.py` for the live whitelist and
[`docs/reference/config.md`](./config.md) for the per-backend
configuration keys.

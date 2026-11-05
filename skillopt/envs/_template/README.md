# Benchmark Template

This directory provides scaffold files for adding a new benchmark to SkillOpt.

## Files

- `env_template.py` — Environment adapter template (subclasses
  `EnvAdapter`; implements the 4 abstract methods so the file is
  instantiable out of the box — `reflect` is inherited).
- `loader_template.py` — Data loader template (subclasses
  `SplitDataLoader`; implements `load_split_items` for `.json`/`.jsonl`).
- `config_template.yaml` — Config file template.

## Usage

1. **Copy the directory:**
   ```bash
   cp -r skillopt/envs/_template skillopt/envs/your_benchmark
   ```
2. **Rename the files** (drop the `_template` suffix):
   ```bash
   cd skillopt/envs/your_benchmark
   mv env_template.py    adapter.py
   mv loader_template.py dataloader.py
   ```
   …and inside each file rename the classes
   (`TemplateBenchmarkEnv → YourBenchmarkAdapter`,
   `TemplateBenchmarkLoader → YourBenchmarkLoader`)
   and fix the cross-import in `adapter.py`.
3. **Implement the TODO blocks** inside `adapter.py:rollout` and the
   `_normalize_item` helper in `dataloader.py`. In addition to returning
   `id`/`hard`/`soft`, persist each non-empty trajectory as
   `conversation.json` in the directory that
   `confined_task_output_dir(out_dir, id, map_unsafe=True)` from
   `skillopt/envs/task_output.py` returns (`<out_dir>/predictions/<id>/` for a
   filesystem-safe id, `<out_dir>/predictions/_mapped-<sha256 of id>/`
   otherwise); the inherited `reflect` method reads those files. Override
   `reflect` only for custom reflection logic.
4. **Register** the adapter — add matching `try / except ImportError` blocks
   to `_register_builtins()` in both `scripts/train.py` and
   `scripts/eval_only.py`, mapping the registry key to your
   `YourBenchmarkAdapter` class. There is no `BENCHMARK_REGISTRY` dict in
   `skillopt/envs/__init__.py`; each CLI keeps its own lazy `_ENV_REGISTRY`.
5. **Create the config** at `configs/your_benchmark/default.yaml`
   (start from `config_template.yaml`). `_base_` is a **string path**,
   not a list.

See the [Add a New Benchmark guide](../../../docs/guide/new-benchmark.md)
for the full step-by-step with a worked `docfaithful` example.

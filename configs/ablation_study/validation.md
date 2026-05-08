# Ablation Validation Checklist

Use this checklist before launch, during monitoring, and before filling
`docs/ablation_paper_tables.md`.

## Before Launch

Run from repo root:

```bash
cd /home/azureuser/workspace-gzy/SkillReflection
export ALFWORLD_DATA=/home/azureuser/.cache/alfworld
```

Verify syntax for edited files:

```bash
/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python -m py_compile \
  scripts/run_ablation_matrix.py \
  scripts/train.py \
  skillopt/model/azure_openai.py \
  skillopt/envs/searchqa/rollout.py \
  skillopt/envs/spreadsheetbench/rollout.py \
  skillopt/envs/livemathematicianbench/rollout.py \
  skillopt/envs/alfworld/rollout.py \
  skillopt/envs/docvqa/rollout.py
```

Check active runs and duplicate `env.out_root` before starting more:

```bash
/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python - <<'PY'
import subprocess, re, collections
try:
    raw = subprocess.check_output(["pgrep", "-af", "scripts/train.py"], text=True)
except subprocess.CalledProcessError:
    raw = ""
roots = []
for line in raw.splitlines():
    m = re.search(r"env\.out_root=([^\s]+)", line)
    if m:
        roots.append(m.group(1))
ctr = collections.Counter(roots)
print("train_count", len(roots))
print("duplicate_roots", [r.rsplit("/", 1)[-1] for r, c in ctr.items() if c > 1])
for root in sorted(roots):
    print(root.rsplit("/", 1)[-1])
PY
```

## During Monitoring

Check launchers:

```bash
pgrep -af 'scripts/run_ablation_matrix.py' || true
tail -80 outputs/ablation_docvqa_20260503_160225_run/launcher_parallel8.log 2>/dev/null || true
tail -80 outputs/ablation_livemath_alfworld_clean_20260503_155155_run/launcher_livemath_parallel8.log 2>/dev/null || true
tail -80 outputs/ablation_livemath_alfworld_clean_20260503_155155_run/launcher_alfworld_parallel1.log 2>/dev/null || true
```

Scan current logs for new hard failures:

```bash
rg -n "Traceback|ERROR|Error code|AuthenticationError|BadRequest|RateLimit|content_filter|Killed|OutOfMemory|\\[FAIL\\]|\\[RETRY\\]" \
  outputs/ablation_docvqa_20260503_160225_run/logs \
  outputs/ablation_livemath_alfworld_clean_20260503_155155_run/logs \
  outputs/ablation_batch_searchqa_spreadsheet_20260503_153902_run/logs \
  -g '*.log' | tail -160 || true
```

Check resource pressure:

```bash
df -h /tmp
du -sh /tmp/ray 2>/dev/null || true
free -h | sed -n '1,3p'
```

## Quality Checks

LiveMathBench current valid runs should not look like old 768/512 runs:

```bash
/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python - <<'PY'
import json, pathlib
root = pathlib.Path("outputs/ablation_livemath_alfworld_clean_20260503_155155_run")
for run in sorted(root.glob("*livemathematicianbench*")):
    if not run.is_dir() or "archive" in str(run):
        continue
    for rel in ["test_eval_baseline/results.jsonl", "test_eval/results.jsonl"]:
        p = run / rel
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.open(errors="ignore") if l.strip()]
        empty = sum(1 for r in rows if not str(r.get("response", "")).strip())
        answer = sum(1 for r in rows if "<answer>" in str(r.get("response", "")).lower())
        if empty:
            print(run.name, rel, "empty", empty, "answer", answer, "n", len(rows))
PY
```

ALFWorld valid runs must not contain empty action or missing action:

```bash
/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python - <<'PY'
import json, pathlib
root = pathlib.Path("outputs/ablation_livemath_alfworld_clean_20260503_155155_run")
for run in sorted(root.glob("*alfworld*")):
    if not run.is_dir() or "archive" in str(run):
        continue
    bad = []
    fallback = 0
    for c in run.glob("**/conversation.json"):
        data = json.load(c.open(errors="ignore"))
        for step in data:
            if step.get("step") is None:
                continue
            if not step.get("action"):
                bad.append(str(c.relative_to(run)))
                break
            mr = str(step.get("model_response", ""))
            if "empty model response" in mr or "missing action tag" in mr:
                fallback += 1
    print(run.name, "bad_action_files", len(bad), "fallback", fallback)
PY
```

## Filling Tables

Use only `summary.json` fields:

- `best_selection_hard` -> Best Sel
- `baseline_test_hard` -> Base Test
- `test_hard` -> Best Test
- `test_delta_hard` -> Delta
- `total_accepts` -> Accept
- `total_rejects` -> Reject
- `token_summary._total.total_tokens` -> Tokens

Do not fill table rows from logs alone.

# skillopt-eval: controlled baseline-vs-candidate experiment harness

Implements the reproducible experiment requested in
[issue #132](https://github.com/microsoft/SkillOpt/issues/132): test whether
a SkillOpt-optimized `SKILL.md` measurably improves agent workflows over the
installed baseline, under identical harness / model / settings / tasks /
seeds / permissions / budgets.

## Design

```
split scenarios (seeded, disjoint train/validation/test)
  └─ optimization loop: propose bounded edits → score on validation
        gate: STRICT improvement over current best (fail closed on error)
  └─ paired final eval on held-out test split (baseline, candidate, interleaved)
        └─ evalkit: exact McNemar + bootstrap CI on the pass-rate delta
  └─ artifacts: raw_results.csv, results.json, RESULTS.md, trajectories/
```

* **Identical settings**: every run records the same model, model_settings,
  pinned SHA, seed and budgets in `results.json` provenance.
* **Quality is the gate**: cost deltas (tokens, tool calls, turns, latency)
  are reported but can never compensate for a quality regression.
* **Fail closed**: timeout, non-zero exit, malformed output, missing score,
  or incomplete trajectory all count as failures - never silently dropped.
* **Isolated injection**: the live runner reuses
  `skillopt_sleep.adapters.superpowers`, which clones Superpowers at the
  pinned SHA into a throwaway workspace and overlays only the selected
  `SKILL.md` through the normal plugin bootstrap
  (`claude --plugin-dir`). The installed tree and the source skill file are
  never modified; adoption stays an explicit human decision.
* **No general claim**: one skill, one harness, one model. `RESULTS.md`
  states this verbatim; a null result (no candidate beats baseline) is a
  valid, reportable outcome.

## Run it

### Offline dry-run (mock runner — no keys, no network, any OS)

```bash
cd experiments/skillopt-eval
python run.py --config config.mock.yaml
```

Proves the full pipeline end-to-end (splits → gated optimization → paired
comparison → artifacts) deterministically. The mock simulates the agent:
a skill "works" on a scenario iff it contains directives for the behaviors
that scenario's rule judges require. **It says nothing about real model
behavior.**

### Real experiment (live harness)

Requires: POSIX host (the adapter's pytest shims are bash), `claude` CLI
(`SKILLOPT_CLAUDE_BIN` to override), `ANTHROPIC_API_KEY` — or
`SKILLOPT_HOST_AUTH=1` for trusted candidates only — and network access to
fetch the pinned Superpowers SHA.

1. Copy `config.superpowers.example.yaml`, set `superpowers_sha`,
   `model`, and your split.
2. Prepare bounded-edit candidates as standalone `SKILL.md` files and list
   them under `candidates:` (one per optimization step). The harness stages
   each into the throwaway clone; it never installs them.
3. `python run.py --config your.config.yaml`

Before substantial runs, comment on issue #132 with the target skill +
pinned SHA, harness + model, task/eval source, and whether anything beyond
this runner/docs/tests is needed — per the issue's "Interested?" section.

## Artifacts

| file | contents |
|---|---|
| `raw_results.csv` | one row per run: condition, scenario, split, seed, passed, score, tokens, tool_calls, turns, latency_ms, failure kind, model, skill_hash, pinned_sha, trajectory path, per-check detail |
| `results.json` | full provenance, split assignment, every optimization attempt (accepted/rejected + reason), test-split comparison (McNemar + bootstrap CI), aggregate metrics, verdict |
| `RESULTS.md` | human-readable report; leads with the verdict and the no-general-claim caveat |
| `trajectories/` | raw agent output + evidence per run (gitignored convention — may embed local paths; do not commit) |

Note: the live adapter reports estimated tokens and wall latency; it does not
expose turn/tool-call counts, so those fields are `n/a` in live runs rather
than fabricated. The mock runner fills them in.

## Tests

```bash
python -m pytest tests/test_skillopt_eval.py   # from repo root
```

Deterministic and offline: split integrity, gate accept/reject, fail-closed
on runner error, learning-rate bounding, non-mutation of the source skill,
artifact emission, A/A no-claim, mock determinism.

## Security scope

Same scope as the adapter (`docs/superpowers/SECURITY.md`): trusted,
locally-authored candidates only. The evaluated agent gets Bash and runs as
the harness's OS user — evidence collection is tamper-evident, not
tamper-proof. Candidates never run in public CI with live credentials.

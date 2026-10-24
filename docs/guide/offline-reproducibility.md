# Offline reproducibility checker

The checker reads local split manifests, saved per-item scores, and evaluation
summaries without importing model backends or running benchmark evaluators.
It supports the remaining offline audit work in [Issue #158](https://github.com/microsoft/SkillOpt/issues/158).
It does not change the LiveMath metadata correction from PR #176.

## Minimal example

From the repository root, with Python 3.10+:

```bash
python -m scripts.check_reproducibility tests/fixtures/reproducibility/split_manifest.json --results tests/fixtures/reproducibility/results.jsonl --summary tests/fixtures/reproducibility/eval_summary.json --evidence tests/fixtures/reproducibility/evidence.json
```

This tiny **synthetic** fixture has one train item, one validation item, and
two test items. Hard scores 0.25 and 0.75 average to 0.5; soft scores 0.4 and
0.8 average to approximately 0.6. Artifact and aggregation checks pass, but
the overall status is **MISSING** (exit 2): reconstructed data cannot establish
a historical experiment. The fixture's `source_revision` is explicitly a
synthetic label, not a historical Git commit. Do not relabel it as RECORDED.

Use `--format json` for machine-readable output. Exit codes are 0 for PASS,
1 for FAIL, and 2 for MISSING. FAIL takes precedence when both failures and
missing evidence exist. PASS means the supplied evidence is internally
consistent; provenance declarations are not independent authentication.

To inspect a released split without original run artifacts:

```bash
python -m scripts.check_reproducibility data/docvqa_id_split/split_manifest.json
```

This reports split counts, IDs, declared dataset revision and local SHA-256
digests, with MISSING for historical results. Manifests are lookup lists, not
full runnable benchmark payloads; the checker does not download raw data.

## Supported inputs

* Manifest: the repository's `split_manifest.json` with `counts.train`,
  `counts.val`, `counts.test`, and `source_revision`. Adjacent
  `train/items.json`, `val/items.json`, and `test/items.json` must be arrays
  of objects with nonempty string or integer `id` values. Integer IDs are
  compared as strings, matching the loaders' ID normalization.
* Results: the actual `results.jsonl` format written by benchmark rollouts,
  one object per line with `id`, `hard`, `soft`, and optional other fields.
* Summary: `eval_summary.json` written by `scripts/eval_only.py`, containing
  `split`, `n_items`, `hard`, and `soft`. Use `--split val` or `--split train`
  when checking those canonical splits. The runtime aliases `valid_seen`
  (val) and `valid_unseen` (test) are accepted in both CLI and summary.
  Use `--split all` for `eval_only.py`'s default concatenation of train,
  validation and test. Other aliases/subsets are unsupported.
* Evidence: an explicit JSON sidecar describing the origin and aggregation
  of the supplied artifacts. Existing evaluation outputs do not contain all
  this metadata; missing fields remain MISSING, never inferred.

The checker calls `skillopt.utils.scoring.compute_score`: each metric is the
unweighted mean of **all** result rows. It preserves continuous hard scores.
Failed requests and timeouts are not implicitly excluded. Runtime missing
score keys default to zero, but an audit with missing/null scores withholds
the comparison rather than guessing whether the omission was intentional.
Nonfinite or nonnumeric scores are invalid. Empty results cannot establish
an evaluation, even though the runtime returns zero for an empty list.
An empty results file is MISSING, not a failed numerical comparison. With an
unknown filtering or subset protocol, ID coverage differences remain MISSING;
the checker does not assume that every manifest item was scheduled. Duplicate
result IDs still fail independently. With an explicitly declared all-items
protocol, incomplete coverage fails.

## Evidence sidecar

Use the fixture's `evidence.json` as a schema example, not historical evidence.
For actual artifacts provide:

| Field | Meaning |
|---|---|
| `classification` | `RECORDED`, `RECONSTRUCTED`, or `MISSING` |
| `origin` | RECORDED: `historical` or `new_run`; RECONSTRUCTED: `synthetic` or `reconstruction` |
| `run_id` | Identifies the supplied run or explicitly synthetic example |
| `source_revision` | Actual run's code revision, distinct from the checker's current checkout SHA |
| `aggregation` | `single_run_unweighted_all_items` for the supported comparison protocol |
| `manifest_sha256` | Exact SHA-256 of the supplied manifest |
| `train_sha256`, `val_sha256`, `test_sha256` | Exact SHA-256 of each split item file |
| `results_sha256`, `summary_sha256` | Exact SHA-256 of the corresponding saved artifacts |
| `scorer_sha256` | Optional digest of the scorer; checked against the current scorer when supplied |

Hashes must come from the artifacts whose relationship is being attested.
They bind the declaration to bytes, not to an independently verified timestamp
or author. For example, Python's `hashlib.sha256(Path(path).read_bytes()).hexdigest()`
computes a local file digest. Line-ending changes intentionally change the
digest. The checker never writes or repairs supplied evidence.
Available artifact bindings are checked even when results or summary evidence
is missing or empty. A missing summary withholds the numerical comparison but
does not hide a known hash mismatch in the supplied results or manifest.
Each available artifact is also parsed and validated independently. A missing
or malformed manifest does not skip the other supplied files, and a summary
with invalid fields fails even when results are unavailable or empty. Missing
inputs withhold the aggregate comparison; known failures still take precedence
over MISSING in the overall report.

The report always includes the current Git HEAD and dirty state, the current
scorer digest, the manifest's declared dataset revision, and artifact digests.
A dirty checkout's HEAD alone does not identify all local code. Raw dataset
payloads and the actual historical source snapshot are not verified here.
Recorded new runs stay distinct from recorded historical runs. Synthetic or
new runs can pass a local aggregate comparison while historical evidence
remains MISSING. Even a RECORDED historical declaration cannot prove its own
authenticity or that a paper used those artifacts.

## Aggregation and precision

Numerical comparisons require exact result-ID coverage of the selected split,
unique result IDs, complete finite scores, matching artifact digests, a run ID
and run source revision, summary metadata, and the explicit supported protocol.
The report shows the row count, denominator, excluded count, and error-row
count. Unknown protocols have no claimed effective denominator or exclusions;
the all-row mean may still be shown as a local calculation.

Row-level `weight`, `excluded`, or `run_id` fields require additional protocol
support and withhold comparison. Do not concatenate runs or silently filter
rows. Multi-run, weighted, category-averaged and subset results remain MISSING
until their actual protocol is supplied and supported. A mismatch between
supplied artifacts does not establish that a paper value is wrong.

By default, summary values are fractions and the absolute comparison tolerance
is 1e-9. For a percentage reported to one decimal place, explicitly pass:

```bash
python -m scripts.check_reproducibility path/to/split_manifest.json --results path/to/results.jsonl --summary path/to/eval_summary.json --evidence path/to/evidence.json --scale 100 --decimals 1
```

This uses a conservative half-last-digit interval (0.05 percentage points,
plus four units in the last place of the recomputed floating-point value).
The slack follows the value's representable precision rather than imposing a
fixed 1e-9 floor on high-precision reports. It accepts either tie-rounding convention at
the boundary; it does not infer precision or scale from the reported number.
Metadata containing nonfinite numbers, including overflowing exponents such
as `1e309`, produces a structured FAIL report. Aggregation or scale overflow
also fails without emitting nonfinite JSON numbers.

## Validation and limitations

```bash
python -m pytest -q tests/test_reproducibility.py tests/test_scoring.py tests/test_data_manifests.py
python -m pytest -q
python -m mkdocs build --strict
```

The tests are offline and use tiny synthetic artifacts, including explicitly
mocked historical declarations to test classification branches. No real
historical run is bundled. The checker recomputes saved score aggregation,
not the correctness of model answers, evaluator judgments, or model settings.
It cannot prove missing historical results and cannot automatically resolve
[Issue #326](https://github.com/microsoft/SkillOpt/issues/326)'s request for
original per-item records, code snapshots, skill artifacts and runtime config.

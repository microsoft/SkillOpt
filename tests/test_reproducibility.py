"""Small offline fixtures; no original experiment records or model calls."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.check_reproducibility import main
from skillopt.utils.reproducibility import PROTOCOL, audit, format_text, sha256
from skillopt.utils.scoring import compute_score

FIXTURE = Path(__file__).parent / "fixtures" / "reproducibility"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURE, tmp_path / "sample")
    return tmp_path / "sample"


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def bind(sample: Path, **updates: object) -> None:
    """Bind mutated test artifacts, never label these synthetic bytes historical."""
    evidence = json.loads((sample / "evidence.json").read_text(encoding="utf-8"))
    for label, path in {
        "manifest": sample / "split_manifest.json",
        "results": sample / "results.jsonl",
        "summary": sample / "eval_summary.json",
        "scorer": ROOT / "skillopt" / "utils" / "scoring.py",
        **{s: sample / s / "items.json" for s in ("train", "val", "test")},
    }.items():
        evidence[f"{label}_sha256"] = sha256(path)
    evidence.update(updates)
    write(sample / "evidence.json", evidence)


def run(sample: Path, **kwargs: object) -> dict:
    return audit(
        sample / "split_manifest.json", source_root=ROOT,
        results_path=sample / "results.jsonl", summary_path=sample / "eval_summary.json",
        evidence_path=sample / "evidence.json", **kwargs,
    )


def status(report: dict, name: str) -> str:
    return next(c["status"] for c in report["checks"] if c["name"] == name)


def test_synthetic_fixture_is_consistent_but_not_historical(sample: Path) -> None:
    report = run(sample)
    assert report["status"] == "MISSING"
    assert report["evidence"]["classification"] == "RECONSTRUCTED"
    assert report["aggregation"]["comparison_status"] == "PASS"
    assert report["aggregation"]["hard"] == 0.5
    assert not [c for c in report["checks"] if c["status"] == "FAIL"]
    assert status(report, "historical_evidence") == "MISSING"


def test_recorded_classification_is_a_declaration_only(sample: Path) -> None:
    # Exercise the recorded branch with explicitly MOCK metadata, not a fixture
    # presented as an actual historical run. Hashes cannot authenticate origin.
    bind(sample, classification="RECORDED", origin="historical", run_id="MOCK-ONLY")
    report = run(sample)
    assert report["status"] == "PASS"
    assert report["evidence"]["run_id"] == "MOCK-ONLY"
    assert any("not independent authentication" in text for text in report["limitations"])


def test_duplicate_split_id(sample: Path) -> None:
    write(sample / "test/items.json", [{"id": "a"}, {"id": "a"}])
    bind(sample)
    report = run(sample)
    assert status(report, "duplicates:test") == "FAIL"
    assert report["duplicates"]["test"] == ["a"]


def test_cross_split_overlap(sample: Path) -> None:
    write(sample / "train/items.json", [{"id": "a"}])
    bind(sample)
    report = run(sample)
    assert status(report, "overlap:train/test") == "FAIL"
    assert report["overlaps"]["train/test"] == ["a"]


def test_manifest_count_mismatch(sample: Path) -> None:
    manifest = json.loads((sample / "split_manifest.json").read_text())
    manifest["counts"]["test"] = 3
    write(sample / "split_manifest.json", manifest)
    bind(sample)
    assert status(run(sample), "count:test") == "FAIL"


@pytest.mark.parametrize("metric", ["hard", "soft"])
def test_score_mismatch_with_complete_protocol(sample: Path, metric: str) -> None:
    summary = json.loads((sample / "eval_summary.json").read_text())
    summary[metric] = 0.9
    write(sample / "eval_summary.json", summary)
    bind(sample)
    report = run(sample)
    assert status(report, f"aggregate:{metric}") == "FAIL"
    assert report["aggregation"]["comparison_status"] == "FAIL"


def test_nonbinary_scores_reuse_runtime_mean(sample: Path) -> None:
    rows = [{"id": "a", "hard": 0.25, "soft": 0.4}, {"id": "b", "hard": 0.75, "soft": 0.8}]
    (sample / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    bind(sample)
    report = run(sample)
    assert report["aggregation"]["hard"] == compute_score(rows)[0]
    assert report["aggregation"]["comparison_status"] == "PASS"


def test_missing_per_item_results(sample: Path) -> None:
    (sample / "results.jsonl").unlink()
    report = run(sample)
    assert status(report, "per_item_results") == "MISSING"
    assert report["aggregation"] == {}


@pytest.mark.parametrize("label,relative", [("results", "results.jsonl"), ("summary", "eval_summary.json")])
def test_missing_manifest_preserves_available_digest_failure(sample: Path, label: str, relative: str) -> None:
    bind(sample)
    (sample / "split_manifest.json").unlink()
    with (sample / relative).open("a", encoding="utf-8") as stream:
        stream.write("\n")
    report = run(sample)
    assert status(report, f"binding:{label}") == "FAIL"
    assert report["status"] == "FAIL"
    assert report["aggregation"].get("comparison_status") != "PASS"


@pytest.mark.parametrize("contents", ["[]", "{broken"])
def test_missing_results_still_validates_bound_summary(sample: Path, contents: str) -> None:
    (sample / "eval_summary.json").write_text(contents, encoding="utf-8")
    bind(sample)
    (sample / "results.jsonl").unlink()
    report = run(sample)
    assert status(report, "binding:summary") == "PASS"
    assert status(report, "summary") == "FAIL"
    assert report["status"] == "FAIL"
    assert report["aggregation"] == {}


@pytest.mark.parametrize("field,value", [
    ("hard", "0.5"), ("soft", True), ("n_items", "2"),
    ("n_items", True), ("n_items", -1), ("split", []),
])
@pytest.mark.parametrize("results_mode", ["missing", "omitted", "empty"])
def test_partial_results_do_not_hide_invalid_summary_fields(
    sample: Path, field: str, value: object, results_mode: str,
) -> None:
    summary = {"split": "test", "n_items": 2, "hard": 0.5, "soft": 0.6, field: value}
    write(sample / "eval_summary.json", summary)
    if results_mode == "empty":
        (sample / "results.jsonl").write_text("", encoding="utf-8")
    bind(sample)
    if results_mode == "missing":
        (sample / "results.jsonl").unlink()
    report = audit(
        sample / "split_manifest.json", source_root=ROOT,
        results_path=None if results_mode == "omitted" else sample / "results.jsonl",
        summary_path=sample / "eval_summary.json", evidence_path=sample / "evidence.json",
    )
    assert status(report, "binding:summary") == "PASS"
    check = f"summary_values:{field}" if field in {"hard", "soft"} else "summary_metadata"
    assert status(report, check) == "FAIL"
    assert report["status"] == "FAIL"
    assert report["aggregation"].get("comparison_status") != "PASS"


@pytest.mark.parametrize("field", ["hard", "soft", "n_items", "split"])
def test_incomplete_summary_fields_remain_missing(sample: Path, field: str) -> None:
    summary = {"split": "test", "n_items": 2, "hard": 0.5, "soft": 0.6}
    summary[field] = None
    write(sample / "eval_summary.json", summary)
    bind(sample)
    report = run(sample)
    check = f"summary_values:{field}" if field in {"hard", "soft"} else "summary_metadata"
    assert status(report, check) == "MISSING"
    assert report["status"] == "MISSING"
    assert report["aggregation"]["comparison_status"] == "MISSING"


@pytest.mark.parametrize("target", ["split_manifest.json", "evidence.json", "train/items.json"])
def test_malformed_artifact_does_not_skip_other_artifacts(sample: Path, target: str) -> None:
    (sample / target).write_text("{broken", encoding="utf-8")
    (sample / "results.jsonl").write_text("{broken", encoding="utf-8")
    write(sample / "eval_summary.json", [])
    report = run(sample)
    label = {"split_manifest.json": "manifest", "evidence.json": "evidence", "train/items.json": "train"}[target]
    assert status(report, label) == "FAIL"
    assert status(report, "results") == "FAIL"
    assert status(report, "summary") == "FAIL"
    assert report["status"] == "FAIL"
    assert "test" in report["split_counts"]


def test_invalid_evidence_metadata_preserves_supplied_bindings(sample: Path) -> None:
    bind(sample, classification="invalid")
    with (sample / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    report = run(sample)
    assert status(report, "evidence") == "FAIL"
    assert status(report, "binding:results") == "FAIL"
    assert status(report, "summary_metadata") == "PASS"
    assert report["aggregation"]["comparison_status"] == "MISSING"


@pytest.mark.parametrize("target", ["split_manifest.json", "results.jsonl", "eval_summary.json"])
def test_missing_artifact_preserves_valid_available_checks(sample: Path, target: str) -> None:
    bind(sample)
    (sample / target).unlink()
    report = run(sample)
    assert report["status"] == "MISSING"
    assert not [check for check in report["checks"] if check["status"] == "FAIL"]
    assert report["aggregation"].get("comparison_status") != "PASS"
    assert status(report, "duplicates:test") == "PASS"
    assert status(report, "binding:train") == "PASS"


@pytest.mark.parametrize("case", ["missing_manifest", "missing_results", "only_missing"])
def test_partial_input_cli_keeps_failures_and_input_bytes(
    sample: Path, capsys: pytest.CaptureFixture, case: str,
) -> None:
    if case == "missing_results":
        write(sample / "eval_summary.json", [])
    bind(sample)
    if case == "missing_manifest":
        (sample / "split_manifest.json").unlink()
        with (sample / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    else:
        (sample / "results.jsonl").unlink()
    before = {path: path.read_bytes() for path in sample.rglob("*") if path.is_file()}
    args = [str(sample / "split_manifest.json"), "--results", str(sample / "results.jsonl"),
            "--summary", str(sample / "eval_summary.json"), "--evidence", str(sample / "evidence.json")]
    expected = "MISSING" if case == "only_missing" else "FAIL"
    code = 2 if expected == "MISSING" else 1
    assert main(args + ["--format", "json"]) == code
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == expected
    assert report["aggregation"].get("comparison_status") != "PASS"
    names = [check["name"] for check in report["checks"]]
    assert len(names) == len(set(names))
    assert main(args) == code
    assert f"Check status: {expected}" in capsys.readouterr().out
    assert before == {path: path.read_bytes() for path in sample.rglob("*") if path.is_file()}


@pytest.mark.parametrize("value,expected", [(66.7, "PASS"), (66.8, "FAIL")])
def test_explicit_percentage_precision(sample: Path, value: float, expected: str) -> None:
    rows = [{"id": "a", "hard": 1 / 3, "soft": 1 / 3}, {"id": "b", "hard": 1, "soft": 1}]
    (sample / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    write(sample / "eval_summary.json", {"n_items": 2, "split": "test", "hard": value, "soft": value})
    bind(sample)
    assert status(run(sample, decimals=1, scale=100), "aggregate:hard") == expected


def test_new_recorded_run_does_not_prove_history(sample: Path) -> None:
    bind(sample, classification="RECORDED", origin="new_run")
    report = run(sample)
    assert status(report, "historical_evidence") == "MISSING"
    assert report["evidence"]["origin"] == "new_run"


@pytest.mark.parametrize("field", ["aggregation", "run_id", "source_revision", "results_sha256"])
def test_missing_protocol_or_provenance(sample: Path, field: str) -> None:
    evidence = json.loads((sample / "evidence.json").read_text())
    evidence.pop(field)
    write(sample / "evidence.json", evidence)
    report = run(sample)
    assert report["status"] == "MISSING"
    if field in {"aggregation", "results_sha256"}:
        assert report["aggregation"]["comparison_status"] == "MISSING"


@pytest.mark.parametrize("metric_value", [None, "0.5", float("nan"), float("inf")])
def test_missing_or_invalid_score_never_becomes_zero(sample: Path, metric_value: object) -> None:
    rows = [{"id": "a", "hard": metric_value, "soft": 0.4}, {"id": "b", "hard": 1, "soft": 0.8}]
    (sample / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    bind(sample)
    report = run(sample)
    assert status(report, "values:hard") == ("MISSING" if metric_value is None else "FAIL")
    assert report["aggregation"]["comparison_status"] == "MISSING"


def test_missing_key_and_failed_row_are_not_filtered(sample: Path) -> None:
    rows = [{"id": "a", "soft": 0.4, "agent_ok": False}, {"id": "b", "hard": 1, "soft": 0.8}]
    (sample / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    bind(sample)
    report = run(sample)
    assert report["aggregation"]["effective_denominator"] == 2
    assert report["aggregation"]["error_rows"] == 1
    assert status(report, "values:hard") == "MISSING"
    rows[0]["hard"] = 0
    (sample / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    bind(sample)
    assert run(sample)["aggregation"]["comparison_status"] == "PASS"


@pytest.mark.parametrize("extra", [{"weight": 2}, {"excluded": True}, {"run_id": "second-run"}])
def test_unknown_aggregation_is_missing(sample: Path, extra: dict) -> None:
    rows = [{"id": "a", "hard": 0, "soft": 0.4, **extra}, {"id": "b", "hard": 1, "soft": 0.8}]
    (sample / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    bind(sample)
    report = run(sample)
    assert status(report, "protocol") == "MISSING"
    assert report["aggregation"]["comparison_status"] == "MISSING"


def test_duplicate_and_missing_result_ids(sample: Path) -> None:
    (sample / "results.jsonl").write_text('{"id":"a","hard":0,"soft":0.4}\n' * 2)
    bind(sample)
    report = run(sample)
    assert status(report, "result_duplicates") == "FAIL"
    assert status(report, "result_coverage") == "FAIL"
    assert report["aggregation"]["comparison_status"] == "MISSING"


def test_reported_denominator(sample: Path) -> None:
    write(sample / "eval_summary.json", {"n_items": 1, "split": "test", "hard": 0.5, "soft": 0.6})
    bind(sample)
    assert status(run(sample), "denominator") == "FAIL"


def test_changed_artifact_is_not_silently_trusted(sample: Path) -> None:
    with (sample / "results.jsonl").open("a") as stream:
        stream.write("\n")
    report = run(sample)
    assert status(report, "binding:results") == "FAIL"
    assert report["aggregation"]["comparison_status"] == "MISSING"


@pytest.mark.parametrize("target", ["split_manifest.json", "test/items.json", "results.jsonl", "eval_summary.json", "evidence.json"])
def test_malformed_json_is_readable_failure(sample: Path, target: str) -> None:
    for invalid in ("{broken", '{"source_revision":NaN}', '{"source_revision":1e309}'):
        (sample / target).write_text(invalid)
        report = run(sample)
        assert report["status"] == "FAIL"
        assert "[FAIL]" in format_text(report)
        json.dumps(report, allow_nan=False)


def test_unavailable_revision_and_manifest(sample: Path) -> None:
    manifest = json.loads((sample / "split_manifest.json").read_text())
    manifest.pop("source_revision")
    write(sample / "split_manifest.json", manifest)
    bind(sample)
    assert status(run(sample), "dataset_revision") == "MISSING"
    report = audit(sample / "absent.json", source_root=sample)
    assert report["source_revision"] is None
    assert report["status"] == "MISSING"


def test_empty_results_are_missing(sample: Path) -> None:
    (sample / "results.jsonl").write_text("")
    bind(sample)
    report = run(sample)
    assert status(report, "nonempty_results") == "MISSING"
    assert status(report, "result_coverage") == "MISSING"
    assert report["status"] == "MISSING"


def test_cli_text_json_and_exit_codes(sample: Path, capsys: pytest.CaptureFixture) -> None:
    args = [str(sample / "split_manifest.json"), "--results", str(sample / "results.jsonl"), "--summary", str(sample / "eval_summary.json"), "--evidence", str(sample / "evidence.json")]
    assert main(args) == 2
    output = capsys.readouterr().out
    assert "Check status: MISSING" in output
    assert "Reproduction limitations:" in output
    assert main(args + ["--format", "json"]) == 2
    assert json.loads(capsys.readouterr().out)["aggregation"]["comparison_status"] == "PASS"
    bind(sample, classification="RECORDED", origin="historical", run_id="MOCK-ONLY")
    assert main(args) == 0
    capsys.readouterr()
    (sample / "results.jsonl").write_text("bad")
    assert main(args) == 1


def test_protocol_constant_matches_fixture() -> None:
    evidence = json.loads((FIXTURE / "evidence.json").read_text())
    assert evidence["aggregation"] == PROTOCOL
    paths = {"manifest": "split_manifest.json", "results": "results.jsonl", "summary": "eval_summary.json"}
    paths.update({split: f"{split}/items.json" for split in ("train", "val", "test")})
    for label, relative in paths.items():
        path = FIXTURE / relative
        assert b"\r\n" not in path.read_bytes()
        assert sha256(path) == evidence[f"{label}_sha256"]


def test_runtime_split_alias(sample: Path) -> None:
    write(sample / "eval_summary.json", {"split": "valid_unseen", "n_items": 2, "hard": 0.5, "soft": 0.6})
    bind(sample)
    assert run(sample, split="valid_unseen")["aggregation"]["comparison_status"] == "PASS"


def test_runtime_all_splits(sample: Path) -> None:
    rows = [{"id": key, "hard": 0.5, "soft": 0.6} for key in ("train-1", "val-1", "a", "b")]
    (sample / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    write(sample / "eval_summary.json", {"split": "all", "n_items": 4, "hard": 0.5, "soft": 0.6})
    bind(sample)
    assert run(sample, split="all")["aggregation"]["comparison_status"] == "PASS"


def test_unknown_summary_split_is_missing(sample: Path) -> None:
    write(sample / "eval_summary.json", {"split": "some-filtered-subset", "n_items": 2, "hard": 0.5, "soft": 0.6})
    bind(sample)
    assert status(run(sample), "summary_metadata") == "MISSING"


@pytest.mark.parametrize("protocol", [None, "filtered_protocol_unavailable", "weighted_mean"])
def test_unknown_subset_protocol_withholds_failure(sample: Path, protocol: str | None) -> None:
    (sample / "results.jsonl").write_text('{"id":"a","hard":0.25,"soft":0.4}\n', encoding="utf-8")
    write(sample / "eval_summary.json", {"split": "test", "n_items": 1, "hard": 0.25, "soft": 0.4})
    bind(sample, aggregation=protocol)
    report = run(sample)
    assert status(report, "result_coverage") == "MISSING"
    assert report["status"] == "MISSING"
    assert report["aggregation"]["comparison_status"] == "MISSING"


def test_declared_all_items_protocol_rejects_incomplete_coverage(sample: Path) -> None:
    (sample / "results.jsonl").write_text('{"id":"a","hard":0.25,"soft":0.4}\n', encoding="utf-8")
    bind(sample)
    assert status(run(sample), "result_coverage") == "FAIL"


def test_unknown_protocol_still_rejects_duplicate_results(sample: Path) -> None:
    (sample / "results.jsonl").write_text('{"id":"a","hard":0,"soft":0.4}\n' * 2, encoding="utf-8")
    bind(sample, aggregation="unknown")
    report = run(sample)
    assert status(report, "result_duplicates") == "FAIL"
    assert report["status"] == "FAIL"


@pytest.mark.parametrize("omit_argument", [False, True])
def test_results_binding_checked_without_summary(sample: Path, omit_argument: bool) -> None:
    with (sample / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    (sample / "eval_summary.json").unlink()
    report = audit(
        sample / "split_manifest.json", source_root=ROOT, results_path=sample / "results.jsonl",
        summary_path=None if omit_argument else sample / "eval_summary.json", evidence_path=sample / "evidence.json",
    )
    assert status(report, "binding:results") == "FAIL"
    assert report["status"] == "FAIL"
    assert report["aggregation"]["comparison_status"] == "MISSING"


@pytest.mark.parametrize("empty_results", [False, True])
def test_manifest_binding_checked_without_usable_results(sample: Path, empty_results: bool) -> None:
    # Keep valid JSON and consistent counts while changing its attested bytes.
    with (sample / "split_manifest.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    if empty_results:
        (sample / "results.jsonl").write_text("", encoding="utf-8")
    else:
        (sample / "results.jsonl").unlink()
    assert status(run(sample), "binding:manifest") == "FAIL"


@pytest.mark.parametrize("decimals", [9, 10, 11, 12])
def test_high_precision_rejects_distinct_rounded_scores(sample: Path, decimals: int) -> None:
    summary = json.loads((sample / "eval_summary.json").read_text(encoding="utf-8"))
    summary["hard"] = 0.5 + 2 * 10 ** -decimals
    write(sample / "eval_summary.json", summary)
    bind(sample)
    assert status(run(sample, decimals=decimals), "aggregate:hard") == "FAIL"


@pytest.mark.parametrize("reported,expected", [(0.62, "PASS"), (0.63, "PASS"), (0.64, "FAIL")])
def test_rounding_boundary_accepts_either_tie_convention(sample: Path, reported: float, expected: str) -> None:
    rows = [{"id": key, "hard": 0.625, "soft": 0.6} for key in ("a", "b")]
    (sample / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    write(sample / "eval_summary.json", {"split": "test", "n_items": 2, "hard": reported, "soft": 0.6})
    bind(sample)
    assert status(run(sample, decimals=2), "aggregate:hard") == expected


@pytest.mark.parametrize("target", ["split_manifest.json", "evidence.json", "eval_summary.json"])
def test_exponent_overflow_emits_structured_cli_failure(sample: Path, target: str, capsys: pytest.CaptureFixture) -> None:
    path = sample / target
    value = json.loads(path.read_text(encoding="utf-8"))
    if target == "split_manifest.json":
        value["counts"]["train"] = float("inf")
    elif target == "evidence.json":
        value["source_revision"] = float("inf")
    else:
        value["hard"] = float("inf")
    path.write_text(json.dumps(value).replace("Infinity", "1e309"), encoding="utf-8")
    args = [str(sample / "split_manifest.json"), "--results", str(sample / "results.jsonl"),
            "--summary", str(sample / "eval_summary.json"), "--evidence", str(sample / "evidence.json"), "--format", "json"]
    assert main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("scale", [1, 100])
def test_finite_scores_cannot_overflow_report(sample: Path, scale: int) -> None:
    value = 1e308 if scale == 1 else 1e307
    rows = [{"id": key, "hard": value, "soft": 0.6} for key in ("a", "b")]
    (sample / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    bind(sample)
    report = run(sample, scale=scale)
    assert report["status"] == "FAIL"
    assert report["aggregation"]["comparison_status"] != "PASS"
    json.dumps(report, allow_nan=False)

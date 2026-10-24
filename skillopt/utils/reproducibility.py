"""Read-only audits of released split manifests and saved evaluation scores.

No benchmark adapter is imported and no model or evaluator is executed.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from skillopt.datasets.base import SPLIT_NAMES
from skillopt.utils.scoring import compute_score

PROTOCOL = "single_run_unweighted_all_items"
SPLITS = {"train": "train", "val": "val", "test": "test", "valid_seen": "val", "valid_unseen": "test", "all": "all"}
LIMITATIONS = [
    "Recomputes saved scores, not model outputs or benchmark judgments.",
    "Hashes identify supplied bytes; provenance declarations are not independent authentication.",
    "Current checkout and dataset metadata do not establish the original experiment versions.",
    "Cannot recover missing historical artifacts or resolve Issue #326's artifact request.",
    "Multi-run, weighted, filtered, and subset protocols require separate evidence and are not compared.",
]


def sha256(path: Path) -> str:
    """Hash the exact local artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: nonfinite JSON literal {value}")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path}: nonfinite JSON number {value}")
        return number

    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant, parse_float=finite_float,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _ids(rows: list[Any], label: str) -> list[str]:
    ids = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), (str, int)) or isinstance(row["id"], bool):
            raise ValueError(f"{label}[{index}]: expected a string or integer id")
        value = str(row["id"])
        if not value.strip():
            raise ValueError(f"{label}[{index}]: empty id")
        ids.append(value)
    return ids


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _present(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def audit(
    manifest_path: Path,
    *,
    source_root: Path,
    results_path: Path | None = None,
    summary_path: Path | None = None,
    evidence_path: Path | None = None,
    split: str = "test",
    decimals: int | None = None,
    scale: int = 1,
) -> dict[str, Any]:
    """Audit canonical train/val/test manifests and eval_only JSONL artifacts.

    Rounded reports are compared within half a unit in the explicitly supplied
    last decimal place. Without precision, use absolute tolerance 1e-9.
    Malformed input is FAIL; unavailable evidence is MISSING.
    """
    if split not in SPLITS or scale not in (1, 100) or (decimals is not None and not 0 <= decimals <= 12):
        raise ValueError("Expected supported split, scale 1 or 100, and decimals between 0 and 12")
    split = SPLITS[split]
    report: dict[str, Any] = {
        "source_revision": _git(source_root, "rev-parse", "HEAD"),
        "source_dirty": None,
        "dataset_revision": None,
        "artifacts": {},
        "evidence": {"classification": "MISSING", "origin": None},
        "split_counts": {},
        "duplicates": {},
        "overlaps": {},
        "aggregation": {},
        "checks": [],
        "limitations": list(LIMITATIONS),
    }
    evidence: dict[str, Any] = {}
    bindings_valid: bool | None = None

    def check(name: str, status: str, detail: str) -> None:
        report["checks"].append({"name": name, "status": status, "detail": detail})

    def artifact(path: Path, label: str) -> bool:
        if not path.is_file():
            check(label, "MISSING", f"Artifact unavailable: {path}")
            return False
        try:
            report["artifacts"][label] = {"path": str(path), "sha256": sha256(path)}
        except OSError as exc:
            check(label, "FAIL", f"Cannot read artifact {path}: {exc}")
            return False
        return True

    def check_bindings() -> bool:
        nonlocal bindings_valid
        if bindings_valid is None:
            bindings_valid = True
            labels = ["manifest", "train", "val", "test", "results", "summary"]
            if "scorer_sha256" in evidence:
                labels.append("scorer")
            for label in labels:
                expected_hash = evidence.get(f"{label}_sha256")
                actual_hash = report["artifacts"].get(label, {}).get("sha256")
                status = "MISSING" if not expected_hash or not actual_hash else "PASS" if expected_hash == actual_hash else "FAIL"
                check(f"binding:{label}", status, f"{label}: supplied provenance digest compared with current bytes")
                bindings_valid &= status == "PASS"
        return bindings_valid

    def finish() -> dict[str, Any]:
        # Integrity checks are independent of whether aggregation is possible.
        check_bindings()
        states = {item["status"] for item in report["checks"]}
        report["status"] = "FAIL" if "FAIL" in states else "MISSING" if "MISSING" in states else "PASS"
        report["missing_evidence"] = [c["detail"] for c in report["checks"] if c["status"] == "MISSING"]
        return report

    dirty = _git(source_root, "status", "--porcelain", "--untracked-files=normal")
    report["source_dirty"] = None if dirty is None else bool(dirty)
    check("source", "PASS" if report["source_revision"] else "MISSING", "Current checkout SHA; not historical run SHA")
    scorer = Path(__file__).with_name("scoring.py")
    artifact(scorer, "scorer")
    paths = {
        "manifest": manifest_path, "evidence": evidence_path,
        "results": results_path, "summary": summary_path,
        **{name: manifest_path.parent / name / "items.json" for name in SPLIT_NAMES},
    }
    available = {label: path is not None and artifact(path, label) for label, path in paths.items()}
    split_ids: dict[str, list[str]] = {}
    counts: dict[str, Any] = {}
    manifest_valid = False
    try:
        if available["manifest"]:
            manifest = _object(manifest_path)
            report["dataset_revision"] = manifest.get("source_revision")
            check("dataset_revision", "PASS" if _present(manifest.get("source_revision")) else "MISSING", "Dataset revision declared by manifest; raw payload not verified")
            if not isinstance(manifest.get("counts", {}), dict):
                raise ValueError("manifest counts must be an object")
            counts = manifest.get("counts", {})
            manifest_valid = True
    except (ValueError, OSError) as exc:
        check("manifest", "FAIL", str(exc))
    for name in SPLIT_NAMES:
        if not available[name]:
            continue
        path = manifest_path.parent / name / "items.json"
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"{path}: expected a JSON array")
            ids = _ids(rows, name)
            split_ids[name] = ids
            declared = counts.get(name)
            report["split_counts"][name] = {"actual": len(rows), "declared": declared}
            valid_count = isinstance(declared, int) and not isinstance(declared, bool) and declared >= 0
            check(f"count:{name}", "MISSING" if declared is None else "PASS" if valid_count and declared == len(rows) else "FAIL", f"{name}: declared={declared}, actual={len(rows)}")
            duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
            report["duplicates"][name] = duplicates
            check(f"duplicates:{name}", "FAIL" if duplicates else "PASS", f"{name}: {duplicates}")
        except (ValueError, OSError) as exc:
            check(name, "FAIL", str(exc))
    for left, right in combinations(split_ids, 2):
        overlap = sorted(set(split_ids[left]) & set(split_ids[right]))
        report["overlaps"][f"{left}/{right}"] = overlap
        check(f"overlap:{left}/{right}", "FAIL" if overlap else "PASS", f"{left}/{right}: {overlap}")

    run_metadata = False
    try:
        if available["evidence"]:
            evidence = _object(evidence_path)
        kind = evidence.get("classification", "MISSING")
        origin = evidence.get("origin")
        if not isinstance(kind, str) or kind not in {"RECORDED", "RECONSTRUCTED", "MISSING"}:
            raise ValueError("classification must be RECORDED, RECONSTRUCTED, or MISSING")
        if kind == "RECORDED" and origin not in ("historical", "new_run"):
            raise ValueError("RECORDED requires origin historical or new_run; synthetic data is RECONSTRUCTED")
        if kind == "RECONSTRUCTED" and origin not in ("synthetic", "reconstruction"):
            raise ValueError("RECONSTRUCTED requires origin synthetic or reconstruction")
        report["evidence"] = {"classification": kind, "origin": origin, "run_id": evidence.get("run_id"), "run_source_revision": evidence.get("source_revision")}
        run_metadata = kind != "MISSING" and _present(evidence.get("run_id")) and _present(evidence.get("source_revision"))
        historical = kind == "RECORDED" and origin == "historical" and run_metadata
        check("historical_evidence", "PASS" if historical else "MISSING", "Historical origin, run ID and source revision are user declarations; absent or reconstructed artifacts cannot verify a historical result")
        if kind != "MISSING":
            check("run_metadata", "PASS" if run_metadata else "MISSING", "Run ID and actual run source revision required")
    except (ValueError, OSError) as exc:
        check("evidence", "FAIL", str(exc))

    summary: dict[str, Any] = {}
    summary_metadata = False
    summary_values = False
    if available["summary"]:
        try:
            summary = _object(summary_path)
            n_items = summary.get("n_items")
            summary_split = summary.get("split")
            normalized_split = SPLITS.get(summary_split) if isinstance(summary_split, str) else None
            valid_count = isinstance(n_items, int) and not isinstance(n_items, bool) and n_items >= 0
            invalid_metadata = (
                n_items is not None and not valid_count
                or summary_split is not None and not isinstance(summary_split, str)
                or normalized_split is not None and normalized_split != split
            )
            summary_metadata = valid_count and normalized_split == split
            check("summary_metadata", "FAIL" if invalid_metadata else "PASS" if summary_metadata else "MISSING", f"n_items={n_items}, split={summary_split}")
            summary_values = True
            for metric in ("hard", "soft"):
                value = summary.get(metric)
                state = "MISSING" if value is None else "PASS" if _number(value) else "FAIL"
                check(f"summary_values:{metric}", state, f"{metric}: reported={value}")
                summary_values &= state == "PASS"
        except (ValueError, OSError, OverflowError) as exc:
            check("summary", "FAIL", str(exc))
    else:
        check("summary", "MISSING", "eval_summary.json unavailable; no reported-score comparison")

    # All available bytes and provenance have been read before caching bindings.
    bindings = check_bindings()

    try:
        if not available["results"]:
            check("per_item_results", "MISSING", "Per-item results.jsonl unavailable; no aggregate verdict")
            return finish()
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = _ids(rows, "results")
        duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
        check("result_duplicates", "FAIL" if duplicates else "PASS", str(duplicates))
        expected = split_ids.get(split)
        if split == "all" and len(split_ids) == len(SPLIT_NAMES):
            expected = [item for name in SPLIT_NAMES for item in split_ids[name]]
        protocol = evidence.get("aggregation") == PROTOCOL
        # No implicit dropping of errors: compute_score includes every row.
        unsupported = any(any(key in row for key in ("weight", "excluded", "run_id")) for row in rows)
        protocol &= not unsupported
        check("protocol", "PASS" if protocol else "MISSING", f"Requires declared {PROTOCOL}; row-level run_id/weight/excluded needs separate protocol support")
        coverage = expected is not None and Counter(ids) == Counter(expected) and not duplicates
        coverage_status = "MISSING" if expected is None or not rows else "PASS" if coverage else "FAIL" if protocol else "MISSING"
        check("result_coverage", coverage_status, f"split={split}; missing={sorted(set(expected or []) - set(ids))}; unexpected={sorted(set(ids) - set(expected or []))}; results={len(rows)}")
        complete = True
        for metric in ("hard", "soft"):
            missing = [i for i, row in enumerate(rows) if metric not in row or row[metric] is None]
            invalid = [i for i, row in enumerate(rows) if metric in row and row[metric] is not None and not _number(row[metric])]
            check(f"values:{metric}", "FAIL" if invalid else "MISSING" if missing else "PASS", f"{metric}: missing/null rows={missing}, invalid rows={invalid}")
            complete &= not missing and not invalid
        report["aggregation"] = {
            "n_rows": len(rows), "row_mean_denominator": len(rows),
            "effective_denominator": len(rows) if protocol else None,
            "excluded": 0 if protocol else None,
            "missing_hard_defaults_to_zero_in_runtime": True,
            "error_rows": sum(row.get("agent_ok") is False or row.get("phase") == "timeout" for row in rows),
            "formula": "sum(metric)/len(results); all rows including errors; runtime missing keys default to zero",
            "comparison_status": "MISSING",
        }
        if not rows:
            check("nonempty_results", "MISSING", "Empty results cannot establish a reported evaluation")
            return finish()
        if complete:
            hard, soft = compute_score(rows)
            if math.isfinite(hard) and math.isfinite(soft):
                report["aggregation"].update(hard=hard, soft=soft)
            else:
                check("aggregation_values", "FAIL", "Saved-score aggregation produced a nonfinite value")
                complete = False
        n_items = summary.get("n_items")
        ready = manifest_valid and complete and coverage and protocol and bindings and summary_metadata and summary_values and run_metadata
        if ready:
            check("denominator", "PASS" if n_items == len(rows) else "FAIL", f"reported n_items={n_items}, actual denominator={len(rows)}")
            outcomes = []
            for metric in ("hard", "soft"):
                reported = summary.get(metric)
                actual = report["aggregation"][metric] * scale
                if not math.isfinite(actual):
                    check(f"aggregate:{metric}", "FAIL", f"{metric}: scaling by {scale} produced a nonfinite value")
                    outcomes.append("FAIL")
                    continue
                tolerance = 1e-9 if decimals is None else 0.5 * 10 ** -decimals + 4 * math.ulp(actual)
                status = "MISSING" if reported is None else "FAIL" if not _number(reported) else "PASS" if abs(actual - reported) <= tolerance else "FAIL"
                check(f"aggregate:{metric}", status, f"recomputed={actual}, reported={reported}, scale={scale}, absolute tolerance={tolerance}")
                outcomes.append(status)
            report["aggregation"]["comparison_status"] = "FAIL" if "FAIL" in outcomes or n_items != len(rows) else "MISSING" if "MISSING" in outcomes else "PASS"
        else:
            check("aggregate_comparison", "MISSING", "Incomplete values, ID coverage, artifact bindings, summary metadata, or aggregation protocol; no numerical inconsistency verdict")
    except (ValueError, OSError, OverflowError) as exc:
        check("results", "FAIL", str(exc))
    return finish()


def format_text(report: dict[str, Any]) -> str:
    """Render all findings, including missing evidence, without hiding failures."""
    lines = [
        f"Check status: {report['status']}",
        f"Source revision: {report['source_revision']} (dirty={report['source_dirty']})",
        f"Dataset/manifest revision: {report['dataset_revision']}",
        f"Evidence: {json.dumps(report['evidence'], sort_keys=True)}",
        f"Split counts: {json.dumps(report['split_counts'], sort_keys=True)}",
        f"Duplicates: {json.dumps(report['duplicates'], sort_keys=True)}",
        f"Overlaps: {json.dumps(report['overlaps'], sort_keys=True)}",
        f"Hard/soft aggregation: {json.dumps(report['aggregation'], sort_keys=True)}",
        "Artifact hashes: " + json.dumps(report['artifacts'], sort_keys=True),
    ]
    lines += [f"[{c['status']}] {c['name']}: {c['detail']}" for c in report["checks"]]
    lines += ["Missing evidence: " + json.dumps(report["missing_evidence"]), "Reproduction limitations:"]
    lines += [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines)

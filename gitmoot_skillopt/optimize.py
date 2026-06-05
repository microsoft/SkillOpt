"""Gitmoot SkillOpt optimize command implementation."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from gitmoot_skillopt.artifacts import OutputArtifactWriter, content_hash
from gitmoot_skillopt.contracts import (
    CANDIDATE_PACKAGE_KIND,
    CONTRACT_VERSION,
    CandidatePackage,
    CandidateSummary,
    CandidateTemplate,
    GateRejectionPacket,
    TrainingPackage,
)
from gitmoot_skillopt.preflight import run_optimizer_preflight
from skillopt.engine.trainer import ReflACTTrainer
from skillopt.envs.gitmoot.adapter import GitmootAdapter


def run_optimize(
    *,
    training_package: str,
    artifact_root: str,
    out_root: str,
    candidate_output: str,
    artifact_dir: str = "",
    dry_run: bool = False,
    num_epochs: int = 1,
    batch_size: int = 4,
    seed: int = 42,
    optimizer_model: str = "gpt-5.5",
    target_model: str = "gpt-5.5",
    optimizer_backend: str = "openai_chat",
    target_backend: str = "openai_chat",
    evaluator_id: str = "",
    evaluator_model: str = "",
    evaluator_backend: str = "",
    gate_metric: str = "hard",
    reasoning_effort: str = "",
    skill_update_mode: str = "patch",
    noop_retry_budget: int = 1,
    gate_reject_retry_budget: int = 3,
    wrong_artifact_retry_budget: int = 1,
    gate_reject_retry_close_gap: float = 0.03,
) -> CandidatePackage:
    package_path = _require_file(training_package, "training package")
    artifact_root_path = _require_dir(artifact_root, "artifact root")
    out_root_path = Path(out_root).expanduser()
    out_root_path.mkdir(parents=True, exist_ok=True)
    artifact_dir_path = Path(artifact_dir).expanduser() if str(artifact_dir).strip() else out_root_path / "artifacts"

    package = TrainingPackage.load(package_path)
    initial_skill_path = out_root_path / "initial_skill.md"
    initial_skill_path.write_text(package.template.content, encoding="utf-8")

    if dry_run:
        return write_candidate_package(
            package=package,
            candidate_content=package.template.content,
            summary=_dry_run_summary(package),
            out_root=out_root_path,
            artifact_dir=artifact_dir_path,
            candidate_output=Path(candidate_output).expanduser(),
            dry_run=True,
        )

    preflight = run_optimizer_preflight(
        package,
        optimizer_backend=optimizer_backend,
        target_backend=target_backend,
        optimizer_model=optimizer_model,
        target_model=target_model,
        evaluator_id=evaluator_id,
        evaluator_backend=evaluator_backend,
        evaluator_model=evaluator_model,
    )
    evaluator_config = preflight.evaluator_config
    cfg = build_trainer_config(
        package_path=package_path,
        artifact_root=artifact_root_path,
        out_root=out_root_path,
        initial_skill_path=initial_skill_path,
        dry_run=dry_run,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
        optimizer_model=preflight.optimizer_model,
        target_model=preflight.target_model,
        optimizer_backend=preflight.optimizer_backend,
        target_backend=preflight.target_backend,
        evaluator_config=evaluator_config,
        gate_metric=gate_metric,
        reasoning_effort=reasoning_effort,
        skill_update_mode=skill_update_mode,
        noop_retry_budget=noop_retry_budget,
        gate_reject_retry_budget=gate_reject_retry_budget,
        wrong_artifact_retry_budget=wrong_artifact_retry_budget,
        gate_reject_retry_close_gap=gate_reject_retry_close_gap,
    )
    adapter = GitmootAdapter(
        training_package=str(package_path),
        artifact_root=str(artifact_root_path),
        seed=seed,
        minibatch_size=cfg["minibatch_size"],
        edit_budget=cfg["edit_budget"],
        analyst_workers=cfg["analyst_workers"],
    )
    summary = ReflACTTrainer(cfg, adapter).train()
    candidate_content = _read_best_skill(out_root_path, package.template.content)
    candidate = write_candidate_package(
        package=package,
        candidate_content=candidate_content,
        summary=summary,
        out_root=out_root_path,
        artifact_dir=artifact_dir_path,
        candidate_output=Path(candidate_output).expanduser(),
        dry_run=dry_run,
    )
    return candidate


def build_trainer_config(
    *,
    package_path: Path,
    artifact_root: Path,
    out_root: Path,
    initial_skill_path: Path,
    dry_run: bool,
    num_epochs: int,
    batch_size: int,
    seed: int,
    optimizer_model: str,
    target_model: str,
    optimizer_backend: str,
    target_backend: str,
    evaluator_config: dict[str, Any],
    gate_metric: str,
    reasoning_effort: str,
    skill_update_mode: str,
    noop_retry_budget: int = 1,
    gate_reject_retry_budget: int = 3,
    wrong_artifact_retry_budget: int = 1,
    gate_reject_retry_close_gap: float = 0.03,
) -> dict[str, Any]:
    actual_epochs = 0 if dry_run else max(1, int(num_epochs))
    normalized_gate_metric = str(gate_metric or "hard").strip().lower()
    if normalized_gate_metric not in {"hard", "soft", "mixed"}:
        raise ValueError(f"unsupported gate metric: {gate_metric}")
    return {
        "env": "gitmoot",
        "training_package": str(package_path),
        "artifact_root": str(artifact_root),
        "out_root": str(out_root),
        "skill_init": str(initial_skill_path),
        "model_backend": target_backend,
        "optimizer_model": optimizer_model,
        "target_model": target_model,
        "optimizer_backend": optimizer_backend,
        "target_backend": target_backend,
        "evaluator_config": evaluator_config,
        "reasoning_effort": str(reasoning_effort or "").strip(),
        "rewrite_reasoning_effort": "",
        "rewrite_max_completion_tokens": 64000,
        "azure_openai_endpoint": "",
        "azure_openai_api_version": "2024-12-01-preview",
        "azure_openai_api_key": "",
        "azure_openai_auth_mode": "",
        "azure_openai_ad_scope": "https://cognitiveservices.azure.com/.default",
        "azure_openai_managed_identity_client_id": "",
        "optimizer_azure_openai_endpoint": "",
        "optimizer_azure_openai_api_version": "2024-12-01-preview",
        "optimizer_azure_openai_api_key": "",
        "optimizer_azure_openai_auth_mode": "",
        "optimizer_azure_openai_ad_scope": "https://cognitiveservices.azure.com/.default",
        "optimizer_azure_openai_managed_identity_client_id": "",
        "target_azure_openai_endpoint": "",
        "target_azure_openai_api_version": "2024-12-01-preview",
        "target_azure_openai_api_key": "",
        "target_azure_openai_auth_mode": "",
        "target_azure_openai_ad_scope": "https://cognitiveservices.azure.com/.default",
        "target_azure_openai_managed_identity_client_id": "",
        "codex_exec_path": "codex",
        "codex_exec_sandbox": "workspace-write",
        "codex_exec_profile": "",
        "codex_exec_full_auto": False,
        "codex_exec_reasoning_effort": "none",
        "codex_exec_use_sdk": "auto",
        "codex_exec_network_access": False,
        "codex_exec_web_search": False,
        "codex_exec_approval_policy": "never",
        "claude_code_exec_path": "claude",
        "claude_code_exec_profile": "",
        "claude_code_exec_use_sdk": "auto",
        "claude_code_exec_effort": "medium",
        "claude_code_exec_max_thinking_tokens": 16384,
        "qwen_chat_base_url": "",
        "qwen_chat_api_key": "",
        "qwen_chat_temperature": None,
        "qwen_chat_timeout_seconds": None,
        "qwen_chat_max_tokens": None,
        "qwen_chat_enable_thinking": None,
        "minimax_base_url": "",
        "minimax_api_key": "",
        "minimax_model": "",
        "minimax_temperature": None,
        "minimax_max_tokens": None,
        "minimax_enable_thinking": None,
        "codex_trace_to_optimizer": False,
        "num_epochs": actual_epochs,
        "batch_size": max(1, int(batch_size)),
        "accumulation": 1,
        "seed": int(seed),
        "minibatch_size": max(1, int(batch_size)),
        "merge_batch_size": max(1, int(batch_size)),
        "analyst_workers": 1,
        "max_analyst_rounds": 1,
        "failure_only": False,
        "edit_budget": 4,
        "min_edit_budget": 1,
        "lr_scheduler": "constant",
        "lr_control_mode": "fixed",
        "skill_update_mode": skill_update_mode,
        "use_slow_update": False,
        "slow_update_samples": 0,
        "slow_update_gate_with_selection": False,
        "longitudinal_pair_policy": "mixed",
        "use_meta_skill": False,
        "use_gate": True,
        "gate_metric": normalized_gate_metric,
        "gate_mixed_weight": 0.5,
        "sel_env_num": 0,
        "test_env_num": 0,
        "eval_test": False if dry_run else True,
        "noop_retry_budget": max(0, int(noop_retry_budget)),
        "gate_reject_retry_budget": max(0, int(gate_reject_retry_budget)),
        "wrong_artifact_retry_budget": max(0, int(wrong_artifact_retry_budget)),
        "gate_reject_retry_close_gap": max(0.0, float(gate_reject_retry_close_gap)),
    }


def write_candidate_package(
    *,
    package: TrainingPackage,
    candidate_content: str,
    summary: dict[str, Any],
    out_root: Path,
    artifact_dir: Path,
    candidate_output: Path,
    dry_run: bool,
) -> CandidatePackage:
    writer = OutputArtifactWriter(out_root, artifact_dir)
    diff_text = _diff_text(package.template.content, candidate_content)
    no_candidate_triggers = _no_candidate_triggers(summary, package.template.content, candidate_content)
    eval_report = _eval_report(summary, dry_run=dry_run, no_candidate_triggers=no_candidate_triggers)
    preference_summary = _preference_summary(summary, dry_run=dry_run, no_candidate_triggers=no_candidate_triggers)
    diff_artifact_id = _candidate_artifact_id(package, "candidate-diff")
    artifacts = [
        writer.write_bytes(
            "candidate.diff.md",
            diff_text.encode(),
            artifact_id=diff_artifact_id,
            media_type="text/markdown",
            driver="gitmoot-skillopt",
        ),
        writer.write_bytes(
            "eval-report.json",
            json.dumps(eval_report, indent=2, sort_keys=True).encode() + b"\n",
            artifact_id=_candidate_artifact_id(package, "eval-report"),
            media_type="application/json",
            driver="gitmoot-skillopt",
        ),
        writer.write_bytes(
            "preference-summary.md",
            preference_summary.encode(),
            artifact_id=_candidate_artifact_id(package, "preference-summary"),
            media_type="text/markdown",
            driver="gitmoot-skillopt",
        ),
    ]
    summary_metadata = _summary_metadata(summary, artifacts=artifacts, no_candidate_triggers=no_candidate_triggers)
    gate_rejection = _gate_rejection_packet(summary)
    candidate = CandidatePackage(
        kind=CANDIDATE_PACKAGE_KIND,
        contract_version=CONTRACT_VERSION,
        template_id=package.template.id,
        base_version_id=package.template.version_id,
        candidate=CandidateTemplate(
            content=candidate_content,
            metadata=package.template.metadata,
        ),
        artifacts=artifacts,
        eval_report=eval_report,
        summary=CandidateSummary(
            diff_artifact_id=diff_artifact_id,
            score=_summary_score(summary, no_candidate_triggers=no_candidate_triggers),
            preference_summary=preference_summary.strip(),
            metadata=summary_metadata,
            gate_rejection=gate_rejection,
        ),
    )
    candidate.validate()
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    candidate.dump(candidate_output)
    return candidate


def _candidate_artifact_id(package: TrainingPackage, suffix: str) -> str:
    run_id = str(package.eval_run.id or "").strip()
    if not run_id:
        run_id = str(package.template.id or "candidate").strip()
    return f"{run_id}/{suffix}"


def _require_file(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_dir(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _read_best_skill(out_root: Path, fallback: str) -> str:
    best_skill = out_root / "best_skill.md"
    if best_skill.is_file():
        return best_skill.read_text(encoding="utf-8")
    return fallback


def _diff_text(base: str, candidate: str) -> str:
    lines = difflib.unified_diff(
        base.splitlines(keepends=True),
        candidate.splitlines(keepends=True),
        fromfile="base-template.md",
        tofile="candidate-template.md",
    )
    diff = "".join(lines)
    return diff or "No content changes.\n"


def _eval_report(summary: dict[str, Any], *, dry_run: bool, no_candidate_triggers: list[str] | None = None) -> dict[str, Any]:
    gate_status = str(summary.get("gate_status") or "passed")
    no_candidate_triggers = no_candidate_triggers or []
    no_candidate_details = _no_candidate_details(summary, no_candidate_triggers)
    return {
        "dry_run": dry_run,
        "gate_status": gate_status,
        "gate_blocker": summary.get("gate_blocker", ""),
        "gate_blockers": summary.get("gate_blockers", []),
        "promotable": _summary_promotable(summary, no_candidate_triggers=no_candidate_triggers),
        "no_candidate_reason": _primary_no_candidate_reason(no_candidate_triggers),
        "no_candidate_triggers": no_candidate_triggers,
        "no_candidate_details": no_candidate_details,
        "next_action": _no_candidate_next_action(no_candidate_triggers),
        "best_selection_hard": summary.get("best_selection_hard"),
        "best_selection_soft": summary.get("best_selection_soft"),
        "baseline_selection_hard": summary.get("baseline_selection_hard"),
        "baseline_selection_soft": summary.get("baseline_selection_soft"),
        "best_step": summary.get("best_step"),
        "total_steps": summary.get("total_steps"),
        "total_accepts": summary.get("total_accepts"),
        "total_rejects": summary.get("total_rejects"),
        "total_blocks": summary.get("total_blocks"),
        "total_skips": summary.get("total_skips"),
        "noop_retry_attempts": summary.get("noop_retry_attempts", []),
        "gate_reject_retry_attempts": summary.get("gate_reject_retry_attempts", []),
        "wrong_artifact_retry_attempts": summary.get("wrong_artifact_retry_attempts", []),
        "gate_rejection": _gate_rejection_dict(summary),
        "final_test_skipped_reason": str(summary.get("final_test_skipped_reason") or ""),
        "token_summary": summary.get("token_summary", {}),
    }


def _dry_run_summary(package: TrainingPackage) -> dict[str, Any]:
    return {
        "best_selection_hard": None,
        "baseline_selection_hard": None,
        "best_step": 0,
        "total_steps": 0,
        "total_accepts": 0,
        "total_rejects": 0,
        "total_blocks": 0,
        "total_skips": 0,
        "gate_status": "passed",
        "gate_blocker": "",
        "gate_blockers": [],
        "promotable": True,
        "token_summary": {},
        "training_package": {
            "template_id": package.template.id,
            "items": len(package.items),
        },
    }


def _preference_summary(summary: dict[str, Any], *, dry_run: bool, no_candidate_triggers: list[str] | None = None) -> str:
    mode = "dry-run fixture" if dry_run else "optimizer"
    gate_status = str(summary.get("gate_status") or "passed")
    no_candidate_triggers = no_candidate_triggers or []
    return (
        f"# Gitmoot SkillOpt Candidate\n\n"
        f"Mode: {mode}\n\n"
        f"Best selection hard: {summary.get('best_selection_hard')}\n"
        f"Baseline selection hard: {summary.get('baseline_selection_hard')}\n"
        f"Gate status: {gate_status}\n"
        f"Promotable: {_summary_promotable(summary, no_candidate_triggers=no_candidate_triggers)}\n"
        f"No candidate reason: {_primary_no_candidate_reason(no_candidate_triggers) or 'none'}\n"
        f"Total steps: {summary.get('total_steps')}\n"
    )


def _summary_score(summary: dict[str, Any], *, no_candidate_triggers: list[str] | None = None) -> float | None:
    if not _summary_promotable(summary, no_candidate_triggers=no_candidate_triggers):
        return None
    score = summary.get("best_selection_hard")
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    return float(score)


def _summary_promotable(summary: dict[str, Any], *, no_candidate_triggers: list[str] | None = None) -> bool:
    if no_candidate_triggers:
        return False
    gate_status = str(summary.get("gate_status") or "passed")
    return bool(summary.get("promotable", gate_status != "blocked"))


def _summary_metadata(
    summary: dict[str, Any],
    *,
    artifacts: list[Any],
    no_candidate_triggers: list[str] | None = None,
) -> dict[str, Any]:
    gate_status = str(summary.get("gate_status") or "passed")
    gate_blockers = summary.get("gate_blockers")
    if not isinstance(gate_blockers, list):
        gate_blockers = []
    no_candidate_triggers = no_candidate_triggers or []
    no_candidate_details = _no_candidate_details(summary, no_candidate_triggers)
    return {
        "artifact_ids": [artifact.id for artifact in artifacts],
        "gate_status": gate_status,
        "gate_blocker": str(summary.get("gate_blocker") or ""),
        "gate_blockers": gate_blockers,
        "promotable": _summary_promotable(summary, no_candidate_triggers=no_candidate_triggers),
        "no_candidate_reason": _primary_no_candidate_reason(no_candidate_triggers),
        "no_candidate_triggers": no_candidate_triggers,
        "no_candidate_details": no_candidate_details,
        "noop_retry_attempts": summary.get("noop_retry_attempts", []),
        "gate_reject_retry_attempts": summary.get("gate_reject_retry_attempts", []),
        "wrong_artifact_retry_attempts": summary.get("wrong_artifact_retry_attempts", []),
        "gate_rejection": _gate_rejection_dict(summary),
        "final_test_skipped_reason": str(summary.get("final_test_skipped_reason") or ""),
        "next_action": _no_candidate_next_action(no_candidate_triggers),
    }


def _gate_rejection_dict(summary: dict[str, Any]) -> dict[str, Any] | None:
    value = summary.get("gate_rejection")
    if not isinstance(value, dict):
        return None
    return {key: data for key, data in value.items() if data not in (None, "", [], {})}


def _gate_rejection_packet(summary: dict[str, Any]) -> GateRejectionPacket | None:
    value = _gate_rejection_dict(summary)
    if value is None:
        return None
    return GateRejectionPacket.from_dict(value)


def _no_candidate_triggers(summary: dict[str, Any], base_content: str, candidate_content: str) -> list[str]:
    triggers = [
        str(trigger).strip()
        for trigger in summary.get("no_candidate_triggers", [])
        if str(trigger).strip()
    ] if isinstance(summary.get("no_candidate_triggers"), list) else []
    if _gate_rejection_dict(summary) is not None:
        triggers.append("gate_rejected_best_origin_initial_skill")
    reason = str(summary.get("no_candidate_reason") or "").strip()
    if reason:
        triggers.append(reason)
    if content_hash(base_content.encode()) == content_hash(candidate_content.encode()):
        triggers.append("candidate_content_unchanged")
    best_origin = str(summary.get("best_origin") or "").strip()
    if best_origin == "initial_skill":
        triggers.append("best_origin_initial_skill")
    if "total_accepts" in summary:
        total_accepts = summary.get("total_accepts")
        if isinstance(total_accepts, bool) or not isinstance(total_accepts, int | float):
            total_accepts = None
        if total_accepts == 0:
            triggers.append("optimizer_total_accepts_zero")
    return _dedupe_triggers(triggers)


def _dedupe_triggers(triggers: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for trigger in triggers:
        key = str(trigger or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _primary_no_candidate_reason(no_candidate_triggers: list[str] | None) -> str:
    triggers = no_candidate_triggers or []
    return triggers[0] if triggers else ""


def _no_candidate_next_action(no_candidate_triggers: list[str] | None) -> str:
    if not no_candidate_triggers:
        return ""
    if "gate_rejected_best_origin_initial_skill" in no_candidate_triggers:
        return (
            "Do not import or publish a candidate review; collect more feedback, "
            "rerun with gate-reject retry if budget remains, or inspect the candidate package."
        )
    return "Do not import or publish a candidate review; continue training with revised feedback or stop the run."


def _no_candidate_details(summary: dict[str, Any], no_candidate_triggers: list[str] | None) -> dict[str, Any]:
    triggers = no_candidate_triggers or []
    details: dict[str, Any] = {}
    gate_rejection = _gate_rejection_dict(summary)
    if gate_rejection is not None and "gate_rejected_best_origin_initial_skill" in triggers:
        details["attempted_patch"] = str(gate_rejection.get("attempted_patch") or "")
        details["rejection"] = {
            "baseline": gate_rejection.get("baseline") or {},
            "candidate": gate_rejection.get("candidate") or {},
            "primary_reason": str(gate_rejection.get("primary_reason") or ""),
            "human_reason": str(gate_rejection.get("human_reason") or ""),
            "failed_dimensions": gate_rejection.get("failed_dimensions") or [],
            "evidence": gate_rejection.get("evidence") or [],
        }
        details["retry_attempts"] = str(gate_rejection.get("retry_attempts") or "")
        details["next_action"] = str(gate_rejection.get("next_action") or _no_candidate_next_action(triggers))
    return {key: value for key, value in details.items() if value not in (None, "", [], {})}

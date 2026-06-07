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
    optimizer_views: int = 1,
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
    feedback_direct_mode: str = "auto",
    target_artifact_retry_budget: int = 1,
    hard_failure_retry_budget: int = 1,
    evaluator_schema_retry_budget: int = 1,
    eval_test: bool = False,
) -> CandidatePackage:
    optimizer_views = _normalize_optimizer_views(optimizer_views)
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
        optimizer_views=optimizer_views,
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
        feedback_direct_mode=feedback_direct_mode,
        target_artifact_retry_budget=target_artifact_retry_budget,
        hard_failure_retry_budget=hard_failure_retry_budget,
        evaluator_schema_retry_budget=evaluator_schema_retry_budget,
        eval_test=eval_test,
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
    optimizer_views: int = 1,
    noop_retry_budget: int = 1,
    gate_reject_retry_budget: int = 3,
    wrong_artifact_retry_budget: int = 1,
    gate_reject_retry_close_gap: float = 0.03,
    feedback_direct_mode: str = "auto",
    target_artifact_retry_budget: int = 1,
    hard_failure_retry_budget: int = 1,
    evaluator_schema_retry_budget: int = 1,
    eval_test: bool = False,
) -> dict[str, Any]:
    actual_epochs = 0 if dry_run else max(1, int(num_epochs))
    normalized_optimizer_views = _normalize_optimizer_views(optimizer_views)
    normalized_gate_metric = str(gate_metric or "hard").strip().lower()
    if normalized_gate_metric not in {"hard", "soft", "mixed"}:
        raise ValueError(f"unsupported gate metric: {gate_metric}")
    evaluator_config = dict(evaluator_config)
    evaluator_config["evaluator_schema_retry_budget"] = max(0, int(evaluator_schema_retry_budget))
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
        "optimizer_views": normalized_optimizer_views,
        "accumulation": 1,
        "seed": int(seed),
        "minibatch_size": max(1, int(batch_size)),
        "merge_batch_size": max(1, int(batch_size)),
        "analyst_workers": normalized_optimizer_views,
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
        "eval_test": bool(eval_test) and not dry_run,
        "noop_retry_budget": max(0, int(noop_retry_budget)),
        "gate_reject_retry_budget": max(0, int(gate_reject_retry_budget)),
        "wrong_artifact_retry_budget": max(0, int(wrong_artifact_retry_budget)),
        "gate_reject_retry_close_gap": max(0.0, float(gate_reject_retry_close_gap)),
        "feedback_direct_mode": _normalize_feedback_direct_mode(feedback_direct_mode),
        "target_artifact_retry_budget": max(0, int(target_artifact_retry_budget)),
        "hard_failure_retry_budget": max(0, int(hard_failure_retry_budget)),
        "evaluator_schema_retry_budget": max(0, int(evaluator_schema_retry_budget)),
    }


def _normalize_feedback_direct_mode(value: str | None) -> str:
    raw = str(value or "auto").strip().lower()
    if raw not in {"auto", "on", "off"}:
        raise ValueError("feedback_direct_mode must be one of auto, on, off")
    return raw


def _normalize_optimizer_views(value: int) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError("optimizer_views must be a positive integer")
    return normalized


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
    sample_artifact = _candidate_selection_sample_artifact(package, summary, writer)
    if sample_artifact is not None:
        artifacts.append(sample_artifact)
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


def _candidate_selection_sample_artifact(
    package: TrainingPackage,
    summary: dict[str, Any],
    writer: OutputArtifactWriter,
) -> Any | None:
    sample_path_text = str(summary.get("best_selection_sample_artifact_path") or "").strip()
    if not sample_path_text:
        return None
    sample_path = Path(sample_path_text)
    if not sample_path.is_file():
        return None
    content = sample_path.read_bytes()
    media_type = "application/json" if sample_path.suffix.lower() == ".json" else "text/plain"
    driver = "vue-vite" if media_type == "application/json" else "text"
    return writer.write_bytes(
        "candidate-selection-sample" + sample_path.suffix.lower(),
        content,
        artifact_id=_candidate_artifact_id(package, "candidate-selection-sample"),
        media_type=media_type,
        driver=driver,
    )


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
    no_candidate_diagnostics = _summary_no_candidate_diagnostics(summary, no_candidate_triggers)
    report = {
        "dry_run": dry_run,
        "gate_status": gate_status,
        "gate_blocker": summary.get("gate_blocker", ""),
        "gate_blockers": summary.get("gate_blockers", []),
        "promotable": _summary_promotable(summary, no_candidate_triggers=no_candidate_triggers),
        "no_candidate_reason": _primary_no_candidate_reason(no_candidate_triggers),
        "no_candidate_triggers": no_candidate_triggers,
        "no_candidate_details": no_candidate_details,
        "no_candidate_diagnostics": no_candidate_diagnostics,
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
        "optimizer_context_items": summary.get("optimizer_context_items", []),
        "noop_retry_attempts": summary.get("noop_retry_attempts", []),
        "gate_reject_retry_attempts": summary.get("gate_reject_retry_attempts", []),
        "wrong_artifact_retry_attempts": summary.get("wrong_artifact_retry_attempts", []),
        "gate_rejection": _gate_rejection_dict(summary),
        "final_test_skipped_reason": str(summary.get("final_test_skipped_reason") or ""),
        "final_eval_enabled": _summary_eval_test_enabled(summary),
        "final_eval_ran": summary.get("test_hard") is not None,
        "best_selection_sample_artifact_path": str(summary.get("best_selection_sample_artifact_path") or ""),
        "token_summary": summary.get("token_summary", {}),
    }
    report.update(_no_candidate_report_fields(no_candidate_details))
    return report


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
    no_candidate_diagnostics = _summary_no_candidate_diagnostics(summary, no_candidate_triggers)
    return {
        "artifact_ids": [artifact.id for artifact in artifacts],
        "gate_status": gate_status,
        "gate_blocker": str(summary.get("gate_blocker") or ""),
        "gate_blockers": gate_blockers,
        "promotable": _summary_promotable(summary, no_candidate_triggers=no_candidate_triggers),
        "no_candidate_reason": _primary_no_candidate_reason(no_candidate_triggers),
        "no_candidate_triggers": no_candidate_triggers,
        "no_candidate_details": no_candidate_details,
        "no_candidate_diagnostics": no_candidate_diagnostics,
        "optimizer_context_items": summary.get("optimizer_context_items", []),
        "noop_retry_attempts": summary.get("noop_retry_attempts", []),
        "gate_reject_retry_attempts": summary.get("gate_reject_retry_attempts", []),
        "wrong_artifact_retry_attempts": summary.get("wrong_artifact_retry_attempts", []),
        "gate_rejection": _gate_rejection_dict(summary),
        "final_test_skipped_reason": str(summary.get("final_test_skipped_reason") or ""),
        "final_eval_enabled": _summary_eval_test_enabled(summary),
        "final_eval_ran": summary.get("test_hard") is not None,
        "best_selection_sample_artifact_path": str(summary.get("best_selection_sample_artifact_path") or ""),
        "next_action": _no_candidate_next_action(no_candidate_triggers),
    }


def _summary_eval_test_enabled(summary: dict[str, Any]) -> bool:
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    value = config.get("eval_test")
    return bool(value)


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
    if "human_feedback_not_distilled" in no_candidate_triggers:
        return (
            "Do not import or publish a candidate review; rerun with stronger optimizer context "
            "or revise the skill manually using the ranked human feedback themes."
        )
    if "gate_rejected_best_origin_initial_skill" in no_candidate_triggers:
        return (
            "Do not import or publish a candidate review; collect more feedback, "
            "rerun with gate-reject retry if budget remains, or inspect the candidate package."
        )
    return "Do not import or publish a candidate review; continue training with revised feedback or stop the run."


def _no_candidate_next_actions(no_candidate_triggers: list[str] | None) -> list[str]:
    if not no_candidate_triggers:
        return []
    if "human_feedback_not_distilled" in no_candidate_triggers:
        return [
            "rerun optimizer with ranked feedback emphasized",
            "manually revise skill from unresolved feedback themes",
            "collect clearer feedback if the themes are ambiguous",
        ]
    if "gate_rejected_best_origin_initial_skill" in no_candidate_triggers:
        return [
            "collect more feedback",
            "rerun with higher retry budget",
            "manually revise skill direction",
        ]
    return [
        "revise feedback and continue training",
        "inspect the candidate package",
        "stop the run",
    ]


def _no_candidate_details(summary: dict[str, Any], no_candidate_triggers: list[str] | None) -> dict[str, Any]:
    triggers = no_candidate_triggers or []
    details: dict[str, Any] = {}
    gate_rejection = _gate_rejection_dict(summary)
    if gate_rejection is not None and "gate_rejected_best_origin_initial_skill" in triggers:
        baseline = gate_rejection.get("baseline") if isinstance(gate_rejection.get("baseline"), dict) else {}
        candidate = gate_rejection.get("candidate") if isinstance(gate_rejection.get("candidate"), dict) else {}
        diagnostics = _gate_rejection_diagnostics(summary, gate_rejection)
        details["attempted_patch"] = str(gate_rejection.get("attempted_patch") or "")
        details["baseline_gate"] = _gate_score(baseline)
        details["candidate_gate"] = _gate_score(candidate)
        details["duplicate_retry_detected"] = _duplicate_retry_detected(summary)
        details["evaluator_reason"] = _gate_evaluator_reason(gate_rejection, candidate)
        details["diagnostics"] = diagnostics
        details["diagnostic_categories"] = diagnostics.get("categories", [])
        details["selection_gate_relation"] = diagnostics.get("selection_gate_relation", "")
        details["retry_budget_exhausted"] = diagnostics.get("retry_budget_exhausted", False)
        details["feedback_themes"] = diagnostics.get("feedback_themes", [])
        details["stop_reason"] = diagnostics.get("stop_reason", "")
        details["optimizer_context_items"] = summary.get("optimizer_context_items", [])
        retry_metadata = gate_rejection.get("retry_metadata") if isinstance(gate_rejection.get("retry_metadata"), dict) else {}
        if retry_metadata:
            details["retry_metadata"] = retry_metadata
            details["score_gap"] = retry_metadata.get("score_gap")
            details["score_gap_handling"] = retry_metadata.get("score_gap_handling", "")
            details["hard_score_handling"] = retry_metadata.get("hard_score_handling", "")
        if diagnostics.get("evaluator_contract_failure"):
            details["candidate_quality_status"] = "judge_passed_but_schema_failed"
            details["raw_judge_hard"] = diagnostics.get("raw_judge_hard")
            details["raw_judge_soft"] = diagnostics.get("raw_judge_soft")
            details["normalized_hard"] = candidate.get("hard")
            details["normalized_soft"] = candidate.get("soft")
        details["rejection"] = {
            "baseline": baseline,
            "candidate": candidate,
            "primary_reason": str(gate_rejection.get("primary_reason") or ""),
            "human_reason": str(gate_rejection.get("human_reason") or ""),
            "optimizer_hint": str(gate_rejection.get("optimizer_hint") or ""),
            "failed_dimensions": gate_rejection.get("failed_dimensions") or [],
            "evidence": gate_rejection.get("evidence") or [],
            "human_feedback_context": gate_rejection.get("human_feedback_context") or {},
        }
        details["human_feedback_context"] = gate_rejection.get("human_feedback_context") or {}
        details["retry_attempts"] = str(gate_rejection.get("retry_attempts") or "")
        details["next_action"] = str(gate_rejection.get("next_action") or _no_candidate_next_action(triggers))
        details["next_actions"] = _no_candidate_next_actions(triggers)
    if "human_feedback_not_distilled" in triggers:
        failure = summary.get("failure") if isinstance(summary.get("failure"), dict) else {}
        retry_hints = summary.get("feedback_retry_hints") if isinstance(summary.get("feedback_retry_hints"), dict) else {}
        details["primary_reason"] = "human_feedback_not_distilled"
        details["human_reason"] = str(
            failure.get("human_reason")
            or "Ranked human feedback was imported, but the optimizer produced no usable skill changes."
        )
        details["optimizer_hint"] = str(failure.get("optimizer_hint") or summary.get("optimizer_hint") or "")
        details["failed_dimensions"] = failure.get("failed_dimensions") or ["human_feedback_alignment"]
        details["evidence"] = failure.get("evidence") or retry_hints.get("improve") or retry_hints.get("preserve") or []
        details["feedback_retry_hints"] = retry_hints
        details["human_feedback_context"] = failure.get("human_feedback_context") or {}
        details["next_action"] = _no_candidate_next_action(triggers)
        details["next_actions"] = _no_candidate_next_actions(triggers)
    return {key: value for key, value in details.items() if value not in (None, "", [], {})}


def _no_candidate_report_fields(details: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in (
        "attempted_patch",
        "baseline_gate",
        "candidate_gate",
        "retry_attempts",
        "duplicate_retry_detected",
        "diagnostic_categories",
        "evaluator_reason",
        "feedback_themes",
        "optimizer_hint",
        "failed_dimensions",
        "human_feedback_context",
        "retry_budget_exhausted",
        "candidate_quality_status",
        "raw_judge_hard",
        "raw_judge_soft",
        "normalized_hard",
        "normalized_soft",
        "selection_gate_relation",
        "stop_reason",
        "next_actions",
    ):
        if key in details and details[key] not in (None, "", [], {}):
            fields[key] = details[key]
    return fields


def _gate_score(scores: dict[str, Any]) -> float | None:
    value = scores.get("gate_score")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _gate_rejection_diagnostics(summary: dict[str, Any], gate_rejection: dict[str, Any]) -> dict[str, Any]:
    baseline = gate_rejection.get("baseline") if isinstance(gate_rejection.get("baseline"), dict) else {}
    candidate = gate_rejection.get("candidate") if isinstance(gate_rejection.get("candidate"), dict) else {}
    failed_dimensions = _string_list(gate_rejection.get("failed_dimensions"))
    failed_checks = gate_rejection.get("failed_checks") if isinstance(gate_rejection.get("failed_checks"), list) else []
    fields = _diagnostic_fields(gate_rejection, failed_dimensions, failed_checks)
    categories: list[str] = []
    if _contains_any(fields, ("artifact_contract", "artifact_contract_failure", "wrong_artifact_type")):
        categories.append("artifact_contract_failure")
    evaluator_contract_failure = _contains_any(
        fields,
        (
            "evaluator_contract_failure",
            "evaluator_missing_human_feedback_dimensions",
            "llm_judge_human_feedback_dimensions",
        ),
    )
    if evaluator_contract_failure:
        categories.append("evaluator_contract_failure")
    if gate_rejection.get("human_feedback_context"):
        categories.append("old_review_training_signal")
    if _contains_any(fields, ("human_feedback_not_resolved", "human_feedback_resolution", "unresolved_feedback")):
        categories.append("candidate_feedback_unresolved")
    if _contains_any(
        fields,
        (
            "animation_motion_quality",
            "brand_identity",
            "cta_clarity",
            "footer_presence_clarity",
            "hero_quality",
            "mobile_responsiveness",
            "proof_trust_content",
            "ranked_strength_preservation",
            "text_overlap_readability",
            "visual_images_relevance",
            "visual_quality",
        ),
    ):
        categories.append("candidate_specific_quality_failure")

    relation = _selection_gate_relation(_gate_score(baseline), _gate_score(candidate))
    if relation == "tie":
        categories.append("selection_gate_tie")

    retry_stop_reasons = _retry_stop_reasons(summary)
    retry_budget_exhausted = _has_retry_budget_exhausted(retry_stop_reasons)
    if retry_budget_exhausted:
        categories.append("retry_budget_exhausted")
    retry_metadata = gate_rejection.get("retry_metadata") if isinstance(gate_rejection.get("retry_metadata"), dict) else {}

    return {
        "categories": _dedupe_triggers(categories),
        "selection_gate_relation": relation,
        "retry_budget_exhausted": retry_budget_exhausted,
        "retry_stop_reasons": retry_stop_reasons,
        "stop_reason": retry_stop_reasons[-1] if retry_stop_reasons else "",
        "retry_metadata": retry_metadata,
        "score_gap": retry_metadata.get("score_gap"),
        "score_gap_handling": retry_metadata.get("score_gap_handling", ""),
        "hard_score_handling": retry_metadata.get("hard_score_handling", ""),
        "feedback_themes": _feedback_themes(gate_rejection.get("human_feedback_context")),
        "evaluator_contract_failure": evaluator_contract_failure,
        "raw_judge_hard": _raw_judge_score(gate_rejection, "hard"),
        "raw_judge_soft": _raw_judge_score(gate_rejection, "soft"),
    }


def _diagnostic_fields(gate_rejection: dict[str, Any], failed_dimensions: list[str], failed_checks: list[Any]) -> list[str]:
    values: list[Any] = [
        gate_rejection.get("primary_reason"),
        gate_rejection.get("rejection_type"),
        gate_rejection.get("human_reason"),
        *failed_dimensions,
    ]
    for check in failed_checks:
        if isinstance(check, dict):
            values.extend([check.get("check"), check.get("reason"), check.get("severity")])
    return [str(value or "").strip().lower().replace("-", "_").replace(".", "_") for value in values if str(value or "").strip()]


def _contains_any(fields: list[str], tokens: tuple[str, ...]) -> bool:
    return any(token in field for field in fields for token in tokens)


def _raw_judge_score(gate_rejection: dict[str, Any], key: str) -> float | int | None:
    evidence = gate_rejection.get("evidence")
    if not isinstance(evidence, list):
        return None
    for item in evidence:
        text = str(item or "").strip()
        if not text.startswith("{"):
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        value = parsed.get(key) if isinstance(parsed, dict) else None
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, int | float):
            return value
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            continue
    return None


def _selection_gate_relation(baseline_gate: float | None, candidate_gate: float | None) -> str:
    if baseline_gate is None or candidate_gate is None:
        return "unknown"
    if abs(candidate_gate - baseline_gate) <= 1e-9:
        return "tie"
    return "candidate_below_baseline" if candidate_gate < baseline_gate else "candidate_above_baseline"


def _retry_stop_reasons(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in ("gate_reject_retry_attempts", "wrong_artifact_retry_attempts", "noop_retry_attempts"):
        attempts = summary.get(key)
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if isinstance(attempt, dict):
                if str(attempt.get("action") or "").strip() != "stop":
                    continue
                reason = str(attempt.get("stop_reason") or "").strip()
                if reason:
                    reasons.append(reason)
    triggers = _string_list(summary.get("no_candidate_triggers"))
    if isinstance(summary.get("noop_retry_attempts"), list) and any(
        trigger in triggers
        for trigger in ("no_meaningful_skill_change", "candidate_content_unchanged", "human_feedback_not_incorporated")
    ):
        reasons.append("noop_retry_budget_exhausted")
    return _dedupe_triggers(reasons)


def _summary_no_candidate_diagnostics(summary: dict[str, Any], no_candidate_triggers: list[str]) -> dict[str, Any]:
    existing = summary.get("no_candidate_diagnostics")
    if isinstance(existing, dict) and existing:
        return existing
    retry_stop_reasons = _retry_stop_reasons(summary)
    categories: list[str] = []
    if _has_retry_budget_exhausted(retry_stop_reasons):
        categories.append("retry_budget_exhausted")
    categories.extend(no_candidate_triggers)
    return {
        "categories": _dedupe_triggers(categories),
        "selection_gate_relation": "unknown",
        "retry_budget_exhausted": _has_retry_budget_exhausted(retry_stop_reasons),
        "retry_stop_reasons": retry_stop_reasons,
    }


def _has_retry_budget_exhausted(retry_stop_reasons: list[str]) -> bool:
    return "budget_exhausted" in retry_stop_reasons or "noop_retry_budget_exhausted" in retry_stop_reasons


def _feedback_themes(context: Any) -> list[str]:
    if not isinstance(context, dict):
        return []
    themes: list[str] = []
    if isinstance(context.get("themes"), list):
        themes.extend(_string_list(context.get("themes")))
    for key in ("improve", "preserve", "avoid", "required_improvements", "reasoning", "reviewer_reasoning"):
        themes.extend(_string_list(context.get(key)))
    return _dedupe_triggers(themes)[:12]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _gate_evaluator_reason(gate_rejection: dict[str, Any], candidate_scores: dict[str, Any]) -> str:
    for source, key in (
        (candidate_scores, "evaluator_reasoning"),
        (candidate_scores, "reasoning"),
        (gate_rejection, "human_reason"),
        (gate_rejection, "optimizer_hint"),
        (gate_rejection, "primary_reason"),
    ):
        reason = str(source.get(key) or "").strip()
        if reason:
            return reason
    return ""


def _duplicate_retry_detected(summary: dict[str, Any]) -> bool:
    attempts: list[Any] = []
    for key in ("gate_reject_retry_attempts", "wrong_artifact_retry_attempts", "noop_retry_attempts"):
        value = summary.get(key)
        if isinstance(value, list):
            attempts.extend(value)
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        if attempt.get("retry_produced_duplicate_candidate") is True or attempt.get("duplicate_retry_detected") is True:
            return True
        stop_reason = str(attempt.get("stop_reason") or "")
        if "duplicate" in stop_reason:
            return True
    return False

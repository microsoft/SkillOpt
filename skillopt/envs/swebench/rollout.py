from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


_DATASET_ALIASES = {
    "lite": ("princeton-nlp/SWE-Bench_Lite", "SWE-bench/SWE-bench_Lite"),
    "verified": ("princeton-nlp/SWE-Bench_Verified", "SWE-bench/SWE-bench_Verified"),
    "full": ("princeton-nlp/SWE-Bench", "SWE-bench/SWE-bench"),
}


def _normalize_dataset_names(dataset_name: str) -> tuple[str, str]:
    key = str(dataset_name or "lite").strip()
    pair = _DATASET_ALIASES.get(key.lower())
    if pair:
        return pair
    return key, key


def _setup_litellm_env() -> None:
    mapping = {
        "AZURE_API_KEY": os.environ.get("AZURE_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY", ""),
        "AZURE_API_BASE": os.environ.get("AZURE_API_BASE") or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "AZURE_API_VERSION": os.environ.get("AZURE_API_VERSION") or os.environ.get("AZURE_OPENAI_API_VERSION", ""),
    }
    for key, value in mapping.items():
        if value and not os.environ.get(key):
            os.environ[key] = value


def _normalize_student_model(student_model: str) -> str:
    model = str(student_model or "").strip()
    if not model:
        return "azure/gpt-5.4"
    if "/" in model:
        return model
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return f"azure/{model}"
    return model


def _load_json(path: str) -> dict | list | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_agent_config(
    *,
    skill_content: str,
    student_model: str,
    step_limit: int,
    cost_limit: float,
) -> tuple[dict, str]:
    try:
        from minisweagent.config import get_config_from_spec
        from minisweagent.utils.serialize import recursive_merge
    except ImportError as exc:
        raise ImportError(
            "SWEBench rollout requires minisweagent. Install the mini-swe-agent environment first."
        ) from exc

    base_config = get_config_from_spec("swebench.yaml")
    system_template = base_config.get("agent", {}).get("system_template", "")
    rendered_system = system_template
    if skill_content.strip():
        rendered_system = (
            system_template.rstrip()
            + "\n\n## Skill Document\n"
            + "The following skill contains learned guidance for SWE-bench style bug-fixing tasks.\n\n"
            + skill_content.strip()
            + "\n"
        )

    agent_override = {
        "agent": {
            "system_template": rendered_system,
            "step_limit": int(step_limit),
            "cost_limit": float(cost_limit),
        },
        "model": {
            "model_name": _normalize_student_model(student_model),
            "cost_tracking": "ignore_errors",
        },
    }
    return recursive_merge(base_config, agent_override), rendered_system


def _load_messages_from_traj(traj_path: Path) -> list[dict]:
    traj_data = _load_json(str(traj_path))
    if not isinstance(traj_data, dict):
        return []
    messages = traj_data.get("messages")
    if not isinstance(messages, list):
        return []
    return [msg for msg in messages if isinstance(msg, dict) and msg.get("role") != "system"]


def _load_exit_status(traj_path: Path) -> str:
    traj_data = _load_json(str(traj_path))
    if not isinstance(traj_data, dict):
        return "missing_traj"
    info = traj_data.get("info")
    if isinstance(info, dict):
        return str(info.get("exit_status") or "unknown")
    return "unknown"


def _run_rollout(
    *,
    items: list[dict],
    predictions_dir: str,
    skill_content: str,
    student_model: str,
    workers: int,
    step_limit: int,
    cost_limit: float,
) -> tuple[list[dict], str]:
    try:
        from minisweagent.run.benchmarks.swebench import process_instance
        from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
    except ImportError as exc:
        raise ImportError(
            "SWEBench rollout requires minisweagent with swebench benchmark support."
        ) from exc

    _setup_litellm_env()
    config, system_prompt = _build_agent_config(
        skill_content=skill_content,
        student_model=student_model,
        step_limit=step_limit,
        cost_limit=cost_limit,
    )

    out_path = Path(predictions_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    preds_path = out_path / "preds.json"
    done_ids: set[str] = set()
    if preds_path.exists():
        data = _load_json(str(preds_path))
        if isinstance(data, dict):
            done_ids = set(data.keys())

    pending = [item for item in items if str(item.get("instance_id")) not in done_ids]
    progress_manager = RunBatchProgressManager(
        len(pending),
        out_path / f"exit_statuses_{int(time.time())}.yaml",
    )

    task_errors: dict[str, str] = {}

    def _process(instance: dict) -> None:
        process_instance(instance, out_path, config, progress_manager)

    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        futures = {
            executor.submit(_process, item): str(item.get("instance_id"))
            for item in pending
        }
        for fut in as_completed(futures):
            iid = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                task_errors[iid] = str(exc)

    preds_data = _load_json(str(preds_path))
    preds_dict = preds_data if isinstance(preds_data, dict) else {}
    results: list[dict] = []

    for item in items:
        iid = str(item.get("instance_id"))
        pred = preds_dict.get(iid, {}) if isinstance(preds_dict, dict) else {}
        traj_path = out_path / iid / f"{iid}.traj.json"
        messages = _load_messages_from_traj(traj_path)
        task_dir = out_path / iid
        task_dir.mkdir(parents=True, exist_ok=True)
        user_prompt = (
            f"Repository: {item.get('repo', '')}\n\n"
            f"Issue:\n{item.get('problem_statement', '').strip()}"
        ).strip()
        with open(task_dir / "conversation.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        with open(task_dir / "student_system_prompt.txt", "w", encoding="utf-8") as f:
            f.write(system_prompt)
        with open(task_dir / "student_user_prompt.txt", "w", encoding="utf-8") as f:
            f.write(user_prompt)

        results.append(
            {
                "id": iid,
                "instance_id": iid,
                "repo": str(item.get("repo") or "").strip(),
                "task_type": str(item.get("repo") or "swebench").strip() or "swebench",
                "task_description": str(item.get("problem_statement") or "").strip(),
                "instruction": str(item.get("problem_statement") or "").strip(),
                "hard": 0,
                "soft": 0.0,
                "response": str(pred.get("model_patch") or ""),
                "submission": str(pred.get("model_patch") or ""),
                "predicted_patch": str(pred.get("model_patch") or ""),
                "agent_ok": bool(messages),
                "n_turns": sum(1 for msg in messages if msg.get("role") == "assistant"),
                "fail_reason": task_errors.get(iid, ""),
                "exit_status": _load_exit_status(traj_path),
            }
        )

    return results, str(preds_path)


def _run_evaluation(
    *,
    preds_path: str,
    dataset_name: str,
    split: str,
    run_id: str,
    eval_workers: int,
    report_dir: str,
    instance_ids: list[str],
) -> dict:
    _, eval_dataset = _normalize_dataset_names(dataset_name)
    os.makedirs(report_dir, exist_ok=True)

    preds_data = _load_json(preds_path)
    model_name = "unknown"
    if isinstance(preds_data, dict) and preds_data:
        first_pred = next(iter(preds_data.values()))
        if isinstance(first_pred, dict):
            model_name = str(first_pred.get("model_name_or_path") or "unknown")
    expected_report = os.path.join(report_dir, f"{model_name.replace('/', '__')}.{run_id}.json")
    if os.path.exists(expected_report):
        cached = _load_json(expected_report)
        return cached if isinstance(cached, dict) else {}

    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        eval_dataset,
        "--split",
        split,
        "--predictions_path",
        preds_path,
        "--max_workers",
        str(max(int(eval_workers), 1)),
        "--run_id",
        run_id,
    ]
    if instance_ids:
        cmd.extend(["--instance_ids"] + instance_ids)

    subprocess.run(
        cmd,
        cwd=report_dir,
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )

    if os.path.exists(expected_report):
        report = _load_json(expected_report)
        return report if isinstance(report, dict) else {}

    for name in sorted(os.listdir(report_dir)):
        if name.endswith(".json") and run_id in name:
            report = _load_json(os.path.join(report_dir, name))
            if isinstance(report, dict):
                if os.path.join(report_dir, name) != expected_report:
                    shutil.move(os.path.join(report_dir, name), expected_report)
                return report
    return {"resolved_ids": [], "total_instances": len(instance_ids), "resolved_instances": 0}


def run_batch(
    *,
    items: list[dict],
    out_root: str,
    skill_content: str,
    student_model: str,
    dataset_name: str,
    hf_split: str,
    workers: int,
    eval_workers: int,
    step_limit: int,
    cost_limit: float,
    timeout_per_instance: int,
) -> list[dict]:
    os.makedirs(out_root, exist_ok=True)
    results_path = os.path.join(out_root, "results.jsonl")
    if os.path.exists(results_path):
        cached: list[dict] = []
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cached.append(json.loads(line))
        if cached:
            return cached

    predictions_dir = os.path.join(out_root, "predictions")
    results, preds_path = _run_rollout(
        items=items,
        predictions_dir=predictions_dir,
        skill_content=skill_content,
        student_model=student_model,
        workers=workers,
        step_limit=step_limit,
        cost_limit=cost_limit,
    )
    eval_report = _run_evaluation(
        preds_path=preds_path,
        dataset_name=dataset_name,
        split=hf_split,
        run_id=f"skillopt_{int(time.time())}",
        eval_workers=eval_workers,
        report_dir=os.path.join(out_root, "evaluation"),
        instance_ids=[str(item.get("instance_id")) for item in items],
    )
    resolved_ids = set(str(i) for i in eval_report.get("resolved_ids", []))
    for row in results:
        resolved = str(row["instance_id"]) in resolved_ids
        row["hard"] = int(resolved)
        row["soft"] = float(int(resolved))
        if not resolved:
            status = row.get("exit_status") or "not_resolved"
            base_reason = str(row.get("fail_reason") or "").strip()
            unresolved = f"swebench unresolved ({status})"
            row["fail_reason"] = f"{base_reason}; {unresolved}" if base_reason else unresolved
        row["timeout_per_instance"] = int(timeout_per_instance)

    with open(results_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return results

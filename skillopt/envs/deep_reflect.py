from __future__ import annotations

import json
import os
from typing import Any, Callable

from skillopt.gradient.deep_probe import generate_deep_probe_instruction
from skillopt.gradient.reflect import run_minibatch_reflect


def run_no_reference_deep_reflect(
    adapter: Any,
    results: list[dict],
    skill_content: str,
    out_dir: str,
    *,
    env_manager: Any = None,
    prediction_dir: str | None = None,
    random_seed: int | None = None,
    step_buffer_context: str = "",
    output_requirements: list[str] | None = None,
    metadata_builder: Callable[[dict], dict] | None = None,
) -> list[dict | None]:
    """Run teacher-designed diagnostic probing without hidden references."""
    if not getattr(adapter, "use_deep_reflect", False):
        return []
    if not isinstance(env_manager, list):
        return []

    prediction_dir = prediction_dir or os.path.join(out_dir, "predictions")
    selected_items = adapter.select_representative_items(
        results,
        env_manager,
        n_failures=getattr(adapter, "deep_reflect_failures", 4),
        n_successes=getattr(adapter, "deep_reflect_successes", 2),
        seed=random_seed,
    )
    if not selected_items:
        return []

    selected_ids = {str(item["id"]) for item in selected_items}
    selected_results = [row for row in results if str(row.get("id")) in selected_ids]
    if metadata_builder is None:
        selected_metadata = [
            {
                "id": str(item.get("id")),
                "task_type": str(item.get("task_type") or item.get("topic") or "unknown"),
                "question_preview": str(item.get("question") or "")[:200],
            }
            for item in selected_items
        ]
    else:
        selected_metadata = [metadata_builder(item) for item in selected_items]

    deep_dir = os.path.join(out_dir, "deep_reflect")
    rollout_dir = os.path.join(deep_dir, "rollout")
    patches_dir = os.path.join(deep_dir, "patches")
    os.makedirs(deep_dir, exist_ok=True)
    print(
        f"    [2b/6 DEEP REFLECT setup] selected={len(selected_items)} "
        "mode=no_reference_probe"
    )

    probe = generate_deep_probe_instruction(
        skill_content=skill_content,
        items=selected_results,
        prediction_dir=prediction_dir,
        system_prompt=adapter.get_deep_probe_prompt(),
        step_buffer_context=step_buffer_context,
        output_requirements=output_requirements,
    )
    if not probe:
        return []

    with open(os.path.join(deep_dir, "probe.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                **probe,
                "reference_summary": {
                    "mode": "no_reference_probe",
                    "selected_count": len(selected_items),
                },
                "selected_examples": selected_metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    deep_results = adapter.rollout(
        selected_items,
        skill_content,
        rollout_dir,
        diagnostic_mode=True,
        diagnostic_instruction=probe["probe_instruction"],
    )
    return run_minibatch_reflect(
        results=deep_results,
        skill_content=skill_content,
        prediction_dir=os.path.join(rollout_dir, "predictions"),
        patches_dir=patches_dir,
        workers=getattr(adapter, "analyst_workers", 8),
        failure_only=getattr(adapter, "failure_only", False),
        minibatch_size=getattr(adapter, "minibatch_size", 8),
        edit_budget=getattr(adapter, "edit_budget", 4),
        random_seed=random_seed,
        error_system=adapter.get_error_minibatch_prompt(),
        success_system=adapter.get_success_minibatch_prompt(),
        step_buffer_context=step_buffer_context,
        update_mode=getattr(getattr(adapter, "_cfg", {}), "get", lambda *_: "patch")(
            "skill_update_mode",
            "patch",
        ),
    )

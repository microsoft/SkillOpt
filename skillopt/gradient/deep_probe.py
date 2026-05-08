"""Teacher-written diagnostic probe generation for deep reflection."""
from __future__ import annotations

from skillopt.gradient.reflect import fmt_minibatch_trajectories
from skillopt.model import chat_teacher
from skillopt.optimizer.meta_skill import format_meta_skill_context
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


def generate_deep_probe_instruction(
    skill_content: str,
    items: list[dict],
    prediction_dir: str,
    *,
    system_prompt: str | None = None,
    step_buffer_context: str = "",
    output_requirements: list[str] | None = None,
    meta_skill_context: str = "",
) -> dict | None:
    """Generate one minimally-perturbing diagnostic probe instruction."""
    trajectories_text = fmt_minibatch_trajectories(items, prediction_dir)
    if not trajectories_text.strip():
        return None

    actual_system = system_prompt or load_prompt("deep_probe")
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        "## Probe Design Goal\n"
        "Design one short diagnostic instruction to append to the student prompt.\n"
        "The instruction should expose the student's current intermediate judgment\n"
        "without materially changing the original scaffold.\n\n"
    )
    if step_buffer_context.strip():
        user += f"## Previous Steps in This Epoch\n{step_buffer_context}\n\n"
    teacher_ctx = format_meta_skill_context(meta_skill_context)
    if teacher_ctx:
        user += teacher_ctx + "\n\n"
    requirements = output_requirements or [
        "- Some trajectories may include a hidden Reference block. Use it to identify what intermediate conclusion matters, but do not reveal or paraphrase that reference directly to the student.",
        "- The instruction must explicitly request a short <analysis>...</analysis> block before the final <answer>...</answer>.",
        "- Keep the readout concise and structured.",
        "- Do not ask for exhaustive listing, full derivation, or a new solving protocol.",
        "- The instruction text should be ready to append directly to the student's prompt.",
    ]
    user += (
        f"## Representative Trajectories ({len(items)} total)\n{trajectories_text}\n\n"
        "## Output Requirements\n"
        + "\n".join(requirements)
        + "\n"
    )

    try:
        response, _ = chat_teacher(
            system=actual_system,
            user=user,
            max_completion_tokens=1024,
            retries=3,
            stage="deep_probe",
        )
        result = extract_json(response)
        if result and str(result.get("probe_instruction", "")).strip():
            parsed = {
                "reasoning": str(result.get("reasoning", "")).strip(),
                "probe_instruction": str(result.get("probe_instruction", "")).strip(),
            }
            if str(result.get("probe_target_id", "")).strip():
                parsed["probe_target_id"] = str(result.get("probe_target_id", "")).strip()
            try:
                if result.get("probe_after_step") is not None:
                    parsed["probe_after_step"] = int(result.get("probe_after_step"))
            except Exception:  # noqa: BLE001
                pass
            return parsed
    except Exception:  # noqa: BLE001
        return None
    return None

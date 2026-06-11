"""ReflACT Aggregate stage — hierarchical patch merging.

The Aggregate stage takes independently-generated patches from the Reflect
stage and merges them into a single coherent patch via hierarchical LLM calls.
Failure-driven patches take priority over success-driven ones.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.model import chat_optimizer
from skillopt.optimizer.meta_skill import format_meta_skill_context
from skillopt.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    is_rewrite_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
)
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


# ── Internal helpers ──────────────────────────────────────────────────────────

def _merge_batch(
    skill_content: str,
    patches: list[dict],
    system_prompt: str,
    update_mode: str,
    meta_skill_context: str = "",
    level: int = 1,
) -> dict:
    """Call optimizer LLM to merge a batch of patches into one."""
    patches_text = json.dumps(patches, ensure_ascii=False, indent=2)
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Patches to merge ({len(patches)} total, merge level {level})\n{patches_text}"
    )
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    try:
        response, _ = chat_optimizer(
            system=system_prompt,
            user=user,
            max_completion_tokens=64000 if is_full_rewrite_minibatch_mode(update_mode) else 16384,
            retries=3,
            stage="merge",
        )
        merged = extract_json(response)
        key = payload_key(update_mode)
        if merged and key in merged:
            for e in merged.get(key, []):
                e["merge_level"] = level
            return merged
    except Exception:  # noqa: BLE001
        pass
    # Fallback: concatenate all edits
    all_edits = []
    for p in patches:
        for e in get_payload_items(p, update_mode):
            e.setdefault("merge_level", level)
            all_edits.append(e)
    return {"reasoning": "fallback concatenation", payload_key(update_mode): all_edits}


def _hierarchical_merge(
    skill_content: str,
    patches: list[dict],
    system_prompt: str,
    update_mode: str,
    batch_size: int,
    verbose: bool,
    label: str = "",
    workers: int = 16,
    meta_skill_context: str = "",
) -> dict:
    """Hierarchically merge N patches using the given system prompt.

    Same-level batches are executed in PARALLEL via ThreadPoolExecutor.
    """
    if not patches:
        return {"reasoning": "no patches", payload_key(update_mode): []}
    if len(patches) == 1:
        return patches[0]

    current = list(patches)
    level = 0
    while len(current) > 1:
        level += 1
        batches: list[tuple[int, list[dict]]] = []
        for i in range(0, len(current), batch_size):
            batch = current[i : i + batch_size]
            batches.append((i, batch))

        if verbose:
            print(
                f"    [aggregate {label}] level={level}  "
                f"{len(current)} patches → {len(batches)} batches "
                f"(parallel, batch_size={batch_size})"
            )

        next_level: list[dict | None] = [None] * len(batches)

        to_merge: list[tuple[int, list[dict]]] = []
        for idx, (i, batch) in enumerate(batches):
            if len(batch) == 1:
                next_level[idx] = batch[0]
            else:
                to_merge.append((idx, batch))

        if to_merge:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(
                        _merge_batch, skill_content, batch, system_prompt, update_mode,
                        meta_skill_context, level,
                    ): idx
                    for idx, batch in to_merge
                }
                for fut in as_completed(futs):
                    idx = futs[fut]
                    next_level[idx] = fut.result()
                    if verbose:
                        batch_i, batch_data = batches[idx]
                        n_edits = len(get_payload_items(next_level[idx], update_mode))
                        print(
                            f"      [aggregate {label}] level={level} "
                            f"batch [{batch_i}:{batch_i+len(batch_data)}] "
                            f"→ 1 patch ({n_edits} {payload_label(update_mode)})"
                        )

        current = [x for x in next_level if x is not None]

    return current[0]


# ── Public API ────────────────────────────────────────────────────────────────

def merge_patches(
    skill_content: str,
    failure_patches: list[dict],
    success_patches: list[dict],
    batch_size: int = 8,
    verbose: bool = True,
    workers: int = 16,
    update_mode: str = "patch",
    meta_skill_context: str = "",
) -> dict:
    """Failure-first hierarchical merge with support count tracking.

    1. Merge failure patches independently (parallel)
    2. Merge success patches independently (parallel)
    3. Final merge: combine both groups with failure priority

    Returns a merged :class:`~skillopt.types.Patch` dict (``edits`` + ``reasoning``).
    """
    if verbose:
        print(
            f"    [3/6 AGGREGATE] "
            f"failure={len(failure_patches)} success={len(success_patches)} "
            f"(parallel, workers={workers})"
        )

    update_mode = normalize_update_mode(update_mode)
    if is_full_rewrite_minibatch_mode(update_mode):
        merge_failure_prompt = load_prompt("merge_failure_full_rewrite")
        merge_success_prompt = load_prompt("merge_success_full_rewrite")
        merge_final_prompt = load_prompt("merge_final_full_rewrite")
    elif is_rewrite_mode(update_mode):
        merge_failure_prompt = load_prompt("merge_failure_rewrite")
        merge_success_prompt = load_prompt("merge_success_rewrite")
        merge_final_prompt = load_prompt("merge_final_rewrite")
    else:
        merge_failure_prompt = load_prompt("merge_failure")
        merge_success_prompt = load_prompt("merge_success")
        merge_final_prompt = load_prompt("merge_final")

    failure_merged = _hierarchical_merge(
        skill_content, failure_patches, merge_failure_prompt, update_mode,
        batch_size, verbose, label="failure", workers=workers,
        meta_skill_context=meta_skill_context,
    )

    success_merged = _hierarchical_merge(
        skill_content, success_patches, merge_success_prompt, update_mode,
        batch_size, verbose, label="success", workers=workers,
        meta_skill_context=meta_skill_context,
    )

    f_edits = get_payload_items(failure_merged, update_mode)
    s_edits = get_payload_items(success_merged, update_mode)

    if not f_edits and not s_edits:
        return {"reasoning": "no updates from either group", payload_key(update_mode): []}
    if not s_edits:
        return failure_merged
    if not f_edits:
        return success_merged

    combined_patches = [failure_merged, success_merged]
    combined_text = json.dumps(combined_patches, ensure_ascii=False, indent=2)
    if is_full_rewrite_minibatch_mode(update_mode):
        item_label = payload_label(update_mode)
        user = (
            f"## Current Skill\n{skill_content}\n\n"
            f"## Two pre-merged candidate groups to combine\n"
            f"Group 1 (from failed trajectories): "
            f"{len(f_edits)} {item_label}\n"
            f"Group 2 (from successful trajectories): "
            f"{len(s_edits)} {item_label}\n\n"
            f"{combined_text}"
        )
    else:
        user = (
            f"## Current Skill\n{skill_content}\n\n"
            f"## Two pre-merged patch groups to combine\n"
            f"Group 1 (failure-driven, HIGH priority): "
            f"{len(f_edits)} edits\n"
            f"Group 2 (success-driven, lower priority): "
            f"{len(s_edits)} edits\n\n"
            f"{combined_text}"
        )
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    try:
        response, _ = chat_optimizer(
            system=merge_final_prompt,
            user=user,
            max_completion_tokens=64000 if is_full_rewrite_minibatch_mode(update_mode) else 16384,
            retries=3,
            stage="merge",
        )
        final = extract_json(response)
        key = payload_key(update_mode)
        if final and key in final:
            if verbose:
                print(
                    f"    [aggregate final] "
                    f"{len(f_edits)}+{len(s_edits)} → {len(final[key])} {payload_label(update_mode)}"
                )
            return final
    except Exception:  # noqa: BLE001
        pass

    return {
        "reasoning": "fallback: failure first, then success",
        payload_key(update_mode): f_edits + s_edits,
    }

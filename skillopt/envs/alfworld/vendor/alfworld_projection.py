# Vendored from SkillRL (Apache-2.0 License)
# Original: agent_system/environments/env_package/alfworld/projection.py

from typing import List
import re


def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """Process raw model outputs into valid ALFWorld actions.

    Extracts text from ``<action>...</action>`` tags and validates that
    the response also contains ``<think>...</think>`` tags.

    Parameters
    ----------
    actions : list[str]
        Raw model outputs, one per environment.
    action_pools : list[list[str]]
        Admissible action lists per environment (unused but kept for API compat).

    Returns
    -------
    actions : list[str]
        Cleaned action strings.
    valids : list[int]
        1 if the action was successfully parsed, 0 otherwise.
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        actions[i] = actions[i].lower()

        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                actions[i] = actions[i][-30:]
                continue

            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            actions[i] = extracted_action
            valids[i] = 1

        except Exception:
            actions[i] = actions[i][-30:]

        # Require <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # Reject responses containing Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids

# Vendored from SkillRL (Apache-2.0 License)
# Original: agent_system/environments/env_package/alfworld/projection.py

import re
from typing import List

from skillopt.envs.alfworld.actions import extract_action


def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """Process raw model outputs into valid ALFWorld actions.

    Extracts text from the last ``<action>...</action>`` tag pair (see
    :func:`skillopt.envs.alfworld.actions.extract_action`) and validates that
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

        # Same parser as the rollout record, so the executed action is the
        # recorded one.
        extracted_action = extract_action(actions[i])
        if extracted_action is None:
            actions[i] = actions[i][-30:]
            continue

        actions[i] = extracted_action
        valids[i] = 1

        # Require <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # Reject responses containing Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids

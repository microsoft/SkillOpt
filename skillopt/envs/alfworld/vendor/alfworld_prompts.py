# Vendored from SkillRL (Apache-2.0 License)
# Original: agent_system/environments/prompts/alfworld.py

from skillopt.prompts import load_prompt

ALFWORLD_TEMPLATE_NO_HIS = load_prompt("rollout_no_history", env="alfworld")
ALFWORLD_TEMPLATE = load_prompt("rollout_with_history", env="alfworld")
ALFWORLD_TEMPLATE_WITH_MEMORY = load_prompt("rollout_with_memory", env="alfworld")

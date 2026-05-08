"""Vendored ALFWorld environment runtime.

Minimal subset of SkillRL's agent_system package needed to run
ALFWorld environments with ReflACT. Original source:
https://github.com/NTU-LANTERN/SkillRL (Apache-2.0 License)
"""
from .alfworld_envs import AlfworldEnvs, build_alfworld_envs
from .alfworld_projection import alfworld_projection
from .env_manager import AlfWorldEnvironmentManager

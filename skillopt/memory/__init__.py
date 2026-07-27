"""skillopt.memory — optional, opt-in mem0-backed persistent memory.

Disabled unless a config sets ``mem0_enabled: true``; see
:mod:`skillopt.memory.settings` for the resolution rules and
:mod:`skillopt.memory.redaction` for what is stripped before anything is sent.
"""
from skillopt.memory.mem0_backend import SkillMemory, mem0_available
from skillopt.memory.redaction import redact_for_upload
from skillopt.memory.settings import Mem0Settings, resolve_settings

__all__ = [
    "Mem0Settings",
    "SkillMemory",
    "mem0_available",
    "redact_for_upload",
    "resolve_settings",
]

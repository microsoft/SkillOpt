"""ReflACT config loading engine — structured YAML with inheritance.

Supports two config formats:
  1. **Structured** (new): sections like ``model``, ``train``, ``gradient``,
     ``optimizer``, ``evaluation``, ``env`` — with ``_base_`` inheritance.
  2. **Flat** (legacy): all keys at top level — fully backward compatible.

Usage::

    from skillopt.config import load_config, flatten_config

    cfg = load_config("configs/searchqa_default.yaml")
    flat = flatten_config(cfg)  # always returns flat dict for trainer
"""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

# ── Section names that indicate a structured config ──────────────────────

_STRUCTURED_SECTIONS = frozenset({
    "model", "train", "gradient", "optimizer", "evaluation", "env",
})

# ── Structured → flat key mapping ────────────────────────────────────────

_FLATTEN_MAP: dict[str, str] = {
    "model.backend": "model_backend",
    "model.teacher": "teacher_model",
    "model.student": "student_model",
    "model.teacher_backend": "teacher_backend",
    "model.student_backend": "student_backend",
    "model.reasoning_effort": "reasoning_effort",
    "model.rewrite_reasoning_effort": "rewrite_reasoning_effort",
    "model.rewrite_max_completion_tokens": "rewrite_max_completion_tokens",
    "model.codex_exec_path": "codex_exec_path",
    "model.codex_exec_sandbox": "codex_exec_sandbox",
    "model.codex_exec_profile": "codex_exec_profile",
    "model.codex_exec_full_auto": "codex_exec_full_auto",
    "model.codex_exec_reasoning_effort": "codex_exec_reasoning_effort",
    "model.codex_exec_use_sdk": "codex_exec_use_sdk",
    "model.codex_exec_network_access": "codex_exec_network_access",
    "model.codex_exec_web_search": "codex_exec_web_search",
    "model.codex_exec_approval_policy": "codex_exec_approval_policy",
    "model.claude_code_exec_path": "claude_code_exec_path",
    "model.claude_code_exec_profile": "claude_code_exec_profile",
    "model.claude_code_exec_use_sdk": "claude_code_exec_use_sdk",
    "model.claude_code_exec_effort": "claude_code_exec_effort",
    "model.claude_code_exec_max_thinking_tokens": "claude_code_exec_max_thinking_tokens",
    "model.codex_trace_to_teacher": "codex_trace_to_teacher",
    "model.azure_endpoint": "azure_endpoint",
    "model.azure_api_version": "azure_api_version",
    "model.azure_api_key": "azure_api_key",
    "model.azure_openai_endpoint": "azure_openai_endpoint",
    "model.azure_openai_api_version": "azure_openai_api_version",
    "model.azure_openai_api_key": "azure_openai_api_key",
    "model.azure_openai_auth_mode": "azure_openai_auth_mode",
    "model.azure_openai_ad_scope": "azure_openai_ad_scope",
    "model.azure_openai_managed_identity_client_id": "azure_openai_managed_identity_client_id",
    "model.teacher_azure_openai_endpoint": "teacher_azure_openai_endpoint",
    "model.teacher_azure_openai_api_version": "teacher_azure_openai_api_version",
    "model.teacher_azure_openai_api_key": "teacher_azure_openai_api_key",
    "model.teacher_azure_openai_auth_mode": "teacher_azure_openai_auth_mode",
    "model.teacher_azure_openai_ad_scope": "teacher_azure_openai_ad_scope",
    "model.teacher_azure_openai_managed_identity_client_id": "teacher_azure_openai_managed_identity_client_id",
    "model.student_azure_openai_endpoint": "student_azure_openai_endpoint",
    "model.student_azure_openai_api_version": "student_azure_openai_api_version",
    "model.student_azure_openai_api_key": "student_azure_openai_api_key",
    "model.student_azure_openai_auth_mode": "student_azure_openai_auth_mode",
    "model.student_azure_openai_ad_scope": "student_azure_openai_ad_scope",
    "model.student_azure_openai_managed_identity_client_id": "student_azure_openai_managed_identity_client_id",
    "train.num_epochs": "num_epochs",
    "train.train_size": "train_size",
    "train.steps_per_epoch": "steps_per_epoch",
    "train.batch_size": "batch_size",
    "train.accumulation": "accumulation",
    "train.seed": "seed",
    "gradient.minibatch_size": "minibatch_size",
    "gradient.merge_batch_size": "merge_batch_size",
    "gradient.analyst_workers": "analyst_workers",
    "gradient.failure_only": "failure_only",
    "gradient.use_deep_reflect": "use_deep_reflect",
    "gradient.deep_reflect_failures": "deep_reflect_failures",
    "gradient.deep_reflect_successes": "deep_reflect_successes",
    "gradient.max_analyst_rounds": "max_analyst_rounds",
    "optimizer.learning_rate": "edit_budget",
    "optimizer.min_learning_rate": "min_edit_budget",
    "optimizer.lr_scheduler": "lr_scheduler",
    "optimizer.lr_control_mode": "lr_control_mode",
    "optimizer.skill_update_mode": "skill_update_mode",
    "optimizer.use_meta_reflect": "use_meta_reflect",
    "optimizer.meta_learning_rate": "meta_edit_budget",
    "optimizer.use_slow_update": "use_slow_update",
    "optimizer.slow_update_samples": "slow_update_samples",
    "optimizer.longitudinal_pair_policy": "longitudinal_pair_policy",
    "optimizer.use_meta_skill": "use_meta_skill",
    "evaluation.use_gate": "use_gate",
    "evaluation.sel_env_num": "sel_env_num",
    "evaluation.test_env_num": "test_env_num",
    "evaluation.eval_test": "eval_test",
    "env.name": "env",
    "env.skill_init": "skill_init",
    "env.out_root": "out_root",
}


# ── Deep merge ───────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns new dict)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


# ── YAML loading with _base_ inheritance ─────────────────────────────────

def _load_yaml(path: str, _visited: set[str] | None = None) -> dict:
    """Load a YAML file, resolving ``_base_`` inheritance recursively."""
    abs_path = os.path.abspath(path)
    if _visited is None:
        _visited = set()
    if abs_path in _visited:
        raise ValueError(f"Circular _base_ inheritance: {abs_path}")
    _visited.add(abs_path)

    with open(abs_path) as f:
        cfg = yaml.safe_load(f) or {}

    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = os.path.join(os.path.dirname(abs_path), base_ref)
        base_cfg = _load_yaml(base_path, _visited)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


# ── Format detection ─────────────────────────────────────────────────────

def is_structured(cfg: dict) -> bool:
    """Return True if *cfg* uses the new structured section format."""
    return any(
        key in _STRUCTURED_SECTIONS and isinstance(cfg.get(key), dict)
        for key in cfg
    )


# ── Flatten ──────────────────────────────────────────────────────────────

def flatten_config(cfg: dict) -> dict:
    """Convert a structured config to the flat dict expected by the trainer.

    If *cfg* is already flat, returns a shallow copy unchanged.
    """
    if not is_structured(cfg):
        return dict(cfg)

    flat: dict[str, Any] = {}

    evaluation_section = cfg.get("evaluation", {})
    if isinstance(evaluation_section, dict) and evaluation_section.get("use_gate") is False:
        raise ValueError(
            "Gate validation is mandatory in this branch. Remove "
            "`evaluation.use_gate: false` from the config."
        )

    # Apply the explicit mapping
    for dotted, flat_key in _FLATTEN_MAP.items():
        section, key = dotted.split(".", 1)
        section_dict = cfg.get(section, {})
        if isinstance(section_dict, dict) and key in section_dict:
            flat[flat_key] = section_dict[key]

    # Pass through env-specific keys not in the explicit mapping
    env_section = cfg.get("env", {})
    if isinstance(env_section, dict):
        mapped_env_keys = {
            k.split(".", 1)[1]
            for k in _FLATTEN_MAP
            if k.startswith("env.")
        }
        for key, val in env_section.items():
            if key not in mapped_env_keys:
                flat[key] = val

    return flat


# ── Override application ─────────────────────────────────────────────────

def _cast_value(val_str: str) -> Any:
    """Auto-cast a CLI string value to int / float / bool / str."""
    if val_str.lower() in ("true", "yes"):
        return True
    if val_str.lower() in ("false", "no"):
        return False
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply ``key=value`` overrides to a structured config (in place).

    Supports both ``section.key=value`` (for structured configs) and
    ``key=value`` (for flat configs or flat keys in env section).
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item!r}")
        key, val_str = item.split("=", 1)
        val = _cast_value(val_str)

        if "." in key:
            section, subkey = key.split(".", 1)
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section][subkey] = val
            else:
                cfg.setdefault(section, {})[subkey] = val
        else:
            # Flat key — apply to top level (for legacy compat)
            cfg[key] = val


# ── Public API ───────────────────────────────────────────────────────────

def load_config(
    path: str,
    overrides: list[str] | None = None,
) -> dict:
    """Load a config file with ``_base_`` inheritance and optional overrides.

    Parameters
    ----------
    path : str
        Path to the YAML config file.
    overrides : list[str] | None
        ``key=value`` strings from ``--cfg-options``.

    Returns
    -------
    dict
        The merged config (structured or flat depending on the YAML).
    """
    cfg = _load_yaml(path)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg

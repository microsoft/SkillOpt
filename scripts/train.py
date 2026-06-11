#!/usr/bin/env python3
"""SkillOpt unified training entry point.

Usage
-----
    python scripts/train.py --config configs/alfworld/default.yaml

Any YAML key can be overridden from the command line::

    python scripts/train.py --config configs/alfworld/default.yaml \\
        --batch_size 40 --num_epochs 2 --seed 123

Run ``python scripts/train.py --help`` for a full list of options.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

# Ensure the project root is on sys.path so ``import skillopt`` works
# regardless of where the script is invoked from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from skillopt.model.common import default_model_for_backend, normalize_backend_name

_OPENAI_DEFAULT_MODEL_SENTINELS = {"gpt-5.4", "gpt-5.5"}


# ── Environment registry ────────────────────────────────────────────────────

_ENV_REGISTRY: dict[str, type] = {}


def _register_builtins() -> None:
    """Lazy-import built-in adapters so we don't pull heavy deps at CLI parse time."""
    try:
        from skillopt.envs.alfworld.adapter import ALFWorldAdapter
        _ENV_REGISTRY["alfworld"] = ALFWorldAdapter
    except ImportError:
        pass  # ALFWorld deps not installed — skip
    try:
        from skillopt.envs.searchqa.adapter import SearchQAAdapter
        _ENV_REGISTRY["searchqa"] = SearchQAAdapter
    except ImportError:
        pass
    try:
        from skillopt.envs.livemathematicianbench.adapter import LiveMathematicianBenchAdapter
        _ENV_REGISTRY["livemathematicianbench"] = LiveMathematicianBenchAdapter
    except ImportError:
        pass
    try:
        from skillopt.envs.babyvision.adapter import BabyVisionAdapter
        _ENV_REGISTRY["babyvision"] = BabyVisionAdapter
    except ImportError:
        pass
    try:
        from skillopt.envs.spreadsheetbench.adapter import SpreadsheetBenchAdapter
        _ENV_REGISTRY["spreadsheetbench"] = SpreadsheetBenchAdapter
    except ImportError:
        pass
    try:
        from skillopt.envs.mmrb.adapter import MMRBAdapter
        _ENV_REGISTRY["mmrb"] = MMRBAdapter
    except ImportError:
        pass
    try:
        from skillopt.envs.docvqa.adapter import DocVQAAdapter
        _ENV_REGISTRY["docvqa"] = DocVQAAdapter
    except ImportError:
        pass
    try:
        from skillopt.envs.mathverse.adapter import MathVerseAdapter
        _ENV_REGISTRY["mathverse"] = MathVerseAdapter
    except ImportError:
        pass
    try:
        from skillopt.envs.officeqa.adapter import OfficeQAAdapter
        _ENV_REGISTRY["officeqa"] = OfficeQAAdapter
    except ImportError:
        pass
    try:
        from skillopt.envs.sealqa.adapter import SealQAAdapter
        _ENV_REGISTRY["sealqa"] = SealQAAdapter
    except ImportError:
        pass
    try:
        from skillopt.envs.swebench.adapter import SWEBenchAdapter
        _ENV_REGISTRY["swebench"] = SWEBenchAdapter
    except ImportError:
        pass


def get_adapter(cfg: dict):
    """Instantiate the environment adapter specified in ``cfg["env"]``."""
    _register_builtins()
    env_name = cfg.get("env", "alfworld")
    if env_name not in _ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_name}'. "
            f"Available: {list(_ENV_REGISTRY.keys())}"
        )
    adapter_cls = _ENV_REGISTRY[env_name]

    # Inspect adapter __init__ signature and only pass accepted kwargs
    import inspect
    sig = inspect.signature(adapter_cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    adapter_kwargs: dict = {}
    for key in accepted:
        if key in cfg:
            adapter_kwargs[key] = cfg[key]

    return adapter_cls(**adapter_kwargs)


# ── CLI ──────────────────────────────────────────────────────────────────────

_BOOL = lambda x: x.lower() in ("true", "1", "yes")  # noqa: E731


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SkillOpt: Executive Strategy for Self-Evolving Agent Skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    p.add_argument("--cfg-options", nargs="+", default=[],
                   help="Override config: section.key=value (e.g. train.batch_size=40)")

    # Legacy flat CLI overrides (still work, prefer --cfg-options for new usage)
    p.add_argument("--env", type=str)
    p.add_argument("--backend", type=str,
                   choices=["azure_openai", "codex", "codex_exec", "claude", "claude_chat", "claude_code_exec", "qwen", "qwen_chat", "minimax", "minimax_chat"])
    p.add_argument("--optimizer_model", type=str)
    p.add_argument("--target_model", type=str)
    p.add_argument("--optimizer_backend", type=str)
    p.add_argument("--target_backend", type=str)
    p.add_argument("--reasoning_effort", type=str,
                   choices=["", "low", "medium", "high", "xhigh", "max"])
    p.add_argument("--rewrite_reasoning_effort", type=str)
    p.add_argument("--rewrite_max_completion_tokens", type=int)
    p.add_argument("--azure_endpoint", type=str)
    p.add_argument("--azure_api_version", type=str)
    p.add_argument("--azure_api_key", type=str)
    p.add_argument("--azure_openai_endpoint", type=str)
    p.add_argument("--azure_openai_api_version", type=str)
    p.add_argument("--azure_openai_api_key", type=str)
    p.add_argument("--azure_openai_auth_mode", type=str)
    p.add_argument("--azure_openai_ad_scope", type=str)
    p.add_argument("--azure_openai_managed_identity_client_id", type=str)
    p.add_argument("--optimizer_azure_openai_endpoint", type=str)
    p.add_argument("--optimizer_azure_openai_api_version", type=str)
    p.add_argument("--optimizer_azure_openai_api_key", type=str)
    p.add_argument("--optimizer_azure_openai_auth_mode", type=str)
    p.add_argument("--optimizer_azure_openai_ad_scope", type=str)
    p.add_argument("--optimizer_azure_openai_managed_identity_client_id", type=str)
    p.add_argument("--target_azure_openai_endpoint", type=str)
    p.add_argument("--target_azure_openai_api_version", type=str)
    p.add_argument("--target_azure_openai_api_key", type=str)
    p.add_argument("--target_azure_openai_auth_mode", type=str)
    p.add_argument("--target_azure_openai_ad_scope", type=str)
    p.add_argument("--target_azure_openai_managed_identity_client_id", type=str)
    p.add_argument("--qwen_chat_base_url", type=str)
    p.add_argument("--qwen_chat_api_key", type=str)
    p.add_argument("--qwen_chat_temperature", type=float)
    p.add_argument("--qwen_chat_timeout_seconds", type=float)
    p.add_argument("--qwen_chat_max_tokens", type=int)
    p.add_argument("--qwen_chat_enable_thinking", type=_BOOL)
    p.add_argument("--optimizer_qwen_chat_base_url", type=str)
    p.add_argument("--optimizer_qwen_chat_api_key", type=str)
    p.add_argument("--optimizer_qwen_chat_temperature", type=float)
    p.add_argument("--optimizer_qwen_chat_timeout_seconds", type=float)
    p.add_argument("--optimizer_qwen_chat_max_tokens", type=int)
    p.add_argument("--optimizer_qwen_chat_enable_thinking", type=_BOOL)
    p.add_argument("--target_qwen_chat_base_url", type=str)
    p.add_argument("--target_qwen_chat_api_key", type=str)
    p.add_argument("--target_qwen_chat_temperature", type=float)
    p.add_argument("--target_qwen_chat_timeout_seconds", type=float)
    p.add_argument("--target_qwen_chat_max_tokens", type=int)
    p.add_argument("--target_qwen_chat_enable_thinking", type=_BOOL)
    p.add_argument("--minimax_base_url", type=str)
    p.add_argument("--minimax_api_key", type=str)
    p.add_argument("--minimax_model", type=str)
    p.add_argument("--minimax_temperature", type=float)
    p.add_argument("--minimax_max_tokens", type=int)
    p.add_argument("--minimax_enable_thinking", type=_BOOL)
    p.add_argument("--codex_exec_path", type=str)
    p.add_argument("--codex_exec_sandbox", type=str)
    p.add_argument("--codex_exec_profile", type=str)
    p.add_argument("--codex_exec_full_auto", type=_BOOL)
    p.add_argument("--codex_exec_reasoning_effort", type=str)
    p.add_argument("--codex_exec_use_sdk", type=str)
    p.add_argument("--codex_exec_network_access", type=_BOOL)
    p.add_argument("--codex_exec_web_search", type=_BOOL)
    p.add_argument("--codex_exec_approval_policy", type=str)
    p.add_argument("--claude_code_exec_path", type=str)
    p.add_argument("--claude_code_exec_profile", type=str)
    p.add_argument("--claude_code_exec_use_sdk", type=str)
    p.add_argument("--claude_code_exec_effort", type=str)
    p.add_argument("--claude_code_exec_max_thinking_tokens", type=int)
    p.add_argument("--codex_trace_to_optimizer", type=_BOOL)
    p.add_argument("--skill_init", type=str)
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--train_size", type=int)
    p.add_argument("--steps_per_epoch", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--accumulation", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--edit_budget", type=int)
    p.add_argument("--min_edit_budget", type=int)
    p.add_argument("--lr_scheduler", type=str,
                   choices=["constant", "linear", "cosine", "autonomous"])
    p.add_argument("--lr_control_mode", type=str,
                   choices=["fixed", "autonomous", "none"])
    p.add_argument("--merge_batch_size", type=int)
    p.add_argument("--max_analyst_rounds", type=int)
    p.add_argument("--sel_env_num", type=int)
    p.add_argument("--test_env_num", type=int)
    p.add_argument("--eval_test", type=_BOOL)
    p.add_argument("--use_gate", type=_BOOL)
    p.add_argument("--max_steps", type=int)
    p.add_argument("--max_api_workers", type=int)
    p.add_argument("--analyst_workers", type=int)
    p.add_argument("--failure_only", type=_BOOL)
    p.add_argument("--minibatch_size", type=int)
    p.add_argument("--skill_update_mode", type=str,
                   choices=[
                       "patch",
                       "rewrite_from_suggestions",
                       "rewrite",
                       "suggestions",
                       "full_rewrite",
                       "full_rewrite_minibatch",
                       "minibatch_full_rewrite",
                   ])
    p.add_argument("--use_slow_update", type=_BOOL)
    p.add_argument("--slow_update_samples", type=int)
    p.add_argument("--longitudinal_pair_policy", type=str,
                   choices=["mixed", "changed", "unchanged"])
    p.add_argument("--use_meta_skill", type=_BOOL)
    p.add_argument("--use_skill_aware_reflection", type=_BOOL)
    p.add_argument("--skill_aware_appendix_source", type=str,
                   choices=["both", "failure_only"])
    p.add_argument("--skill_aware_consolidate_threshold", type=int)
    p.add_argument("--data_path", type=str)
    p.add_argument("--split_mode", type=str,
                   choices=["ratio", "split_dir"])
    p.add_argument("--split_ratio", type=str)
    p.add_argument("--split_seed", type=int)
    p.add_argument("--split_dir", type=str)
    p.add_argument("--split_output_dir", type=str)
    p.add_argument("--data_root", type=str)
    p.add_argument("--max_turns", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--shuffle_choices", type=_BOOL)
    p.add_argument("--use_theorem", type=_BOOL)
    p.add_argument("--use_sketch", type=_BOOL)
    p.add_argument("--image_detail", type=str)
    p.add_argument("--judge_model", type=str)
    p.add_argument("--judge_max_completion_tokens", type=int)
    p.add_argument("--judge_retries", type=int)
    p.add_argument("--out_root", type=str)
    p.add_argument("--mode", type=str)

    return p.parse_args()


# ── Flat key → structured path mapping (for legacy CLI → structured config) ──

_LEGACY_TO_STRUCTURED: dict[str, str] = {
    "backend": "model.backend",
    "optimizer_model": "model.optimizer",
    "target_model": "model.target",
    "optimizer_backend": "model.optimizer_backend",
    "target_backend": "model.target_backend",
    "reasoning_effort": "model.reasoning_effort",
    "rewrite_reasoning_effort": "model.rewrite_reasoning_effort",
    "rewrite_max_completion_tokens": "model.rewrite_max_completion_tokens",
    "azure_endpoint": "model.azure_endpoint",
    "azure_api_version": "model.azure_api_version",
    "azure_api_key": "model.azure_api_key",
    "azure_openai_endpoint": "model.azure_openai_endpoint",
    "azure_openai_api_version": "model.azure_openai_api_version",
    "azure_openai_api_key": "model.azure_openai_api_key",
    "azure_openai_auth_mode": "model.azure_openai_auth_mode",
    "azure_openai_ad_scope": "model.azure_openai_ad_scope",
    "azure_openai_managed_identity_client_id": "model.azure_openai_managed_identity_client_id",
    "optimizer_azure_openai_endpoint": "model.optimizer_azure_openai_endpoint",
    "optimizer_azure_openai_api_version": "model.optimizer_azure_openai_api_version",
    "optimizer_azure_openai_api_key": "model.optimizer_azure_openai_api_key",
    "optimizer_azure_openai_auth_mode": "model.optimizer_azure_openai_auth_mode",
    "optimizer_azure_openai_ad_scope": "model.optimizer_azure_openai_ad_scope",
    "optimizer_azure_openai_managed_identity_client_id": "model.optimizer_azure_openai_managed_identity_client_id",
    "target_azure_openai_endpoint": "model.target_azure_openai_endpoint",
    "target_azure_openai_api_version": "model.target_azure_openai_api_version",
    "target_azure_openai_api_key": "model.target_azure_openai_api_key",
    "target_azure_openai_auth_mode": "model.target_azure_openai_auth_mode",
    "target_azure_openai_ad_scope": "model.target_azure_openai_ad_scope",
    "target_azure_openai_managed_identity_client_id": "model.target_azure_openai_managed_identity_client_id",
    "qwen_chat_base_url": "model.qwen_chat_base_url",
    "qwen_chat_api_key": "model.qwen_chat_api_key",
    "qwen_chat_temperature": "model.qwen_chat_temperature",
    "qwen_chat_timeout_seconds": "model.qwen_chat_timeout_seconds",
    "qwen_chat_max_tokens": "model.qwen_chat_max_tokens",
    "qwen_chat_enable_thinking": "model.qwen_chat_enable_thinking",
    "optimizer_qwen_chat_base_url": "model.optimizer_qwen_chat_base_url",
    "optimizer_qwen_chat_api_key": "model.optimizer_qwen_chat_api_key",
    "optimizer_qwen_chat_temperature": "model.optimizer_qwen_chat_temperature",
    "optimizer_qwen_chat_timeout_seconds": "model.optimizer_qwen_chat_timeout_seconds",
    "optimizer_qwen_chat_max_tokens": "model.optimizer_qwen_chat_max_tokens",
    "optimizer_qwen_chat_enable_thinking": "model.optimizer_qwen_chat_enable_thinking",
    "target_qwen_chat_base_url": "model.target_qwen_chat_base_url",
    "target_qwen_chat_api_key": "model.target_qwen_chat_api_key",
    "target_qwen_chat_temperature": "model.target_qwen_chat_temperature",
    "target_qwen_chat_timeout_seconds": "model.target_qwen_chat_timeout_seconds",
    "target_qwen_chat_max_tokens": "model.target_qwen_chat_max_tokens",
    "target_qwen_chat_enable_thinking": "model.target_qwen_chat_enable_thinking",
    "minimax_base_url": "model.minimax_base_url",
    "minimax_api_key": "model.minimax_api_key",
    "minimax_model": "model.minimax_model",
    "minimax_temperature": "model.minimax_temperature",
    "minimax_max_tokens": "model.minimax_max_tokens",
    "minimax_enable_thinking": "model.minimax_enable_thinking",
    "codex_exec_path": "model.codex_exec_path",
    "codex_exec_sandbox": "model.codex_exec_sandbox",
    "codex_exec_profile": "model.codex_exec_profile",
    "codex_exec_full_auto": "model.codex_exec_full_auto",
    "codex_exec_reasoning_effort": "model.codex_exec_reasoning_effort",
    "codex_exec_use_sdk": "model.codex_exec_use_sdk",
    "codex_exec_network_access": "model.codex_exec_network_access",
    "codex_exec_web_search": "model.codex_exec_web_search",
    "codex_exec_approval_policy": "model.codex_exec_approval_policy",
    "claude_code_exec_path": "model.claude_code_exec_path",
    "claude_code_exec_profile": "model.claude_code_exec_profile",
    "claude_code_exec_use_sdk": "model.claude_code_exec_use_sdk",
    "claude_code_exec_effort": "model.claude_code_exec_effort",
    "claude_code_exec_max_thinking_tokens": "model.claude_code_exec_max_thinking_tokens",
    "codex_trace_to_optimizer": "model.codex_trace_to_optimizer",
    "num_epochs": "train.num_epochs",
    "train_size": "train.train_size",
    "steps_per_epoch": "train.steps_per_epoch",
    "batch_size": "train.batch_size",
    "accumulation": "train.accumulation",
    "seed": "train.seed",
    "minibatch_size": "gradient.minibatch_size",
    "merge_batch_size": "gradient.merge_batch_size",
    "analyst_workers": "gradient.analyst_workers",
    "max_analyst_rounds": "gradient.max_analyst_rounds",
    "failure_only": "gradient.failure_only",
    "edit_budget": "optimizer.learning_rate",
    "min_edit_budget": "optimizer.min_learning_rate",
    "lr_scheduler": "optimizer.lr_scheduler",
    "lr_control_mode": "optimizer.lr_control_mode",
    "skill_update_mode": "optimizer.skill_update_mode",
    "use_slow_update": "optimizer.use_slow_update",
    "slow_update_samples": "optimizer.slow_update_samples",
    "longitudinal_pair_policy": "optimizer.longitudinal_pair_policy",
    "use_meta_skill": "optimizer.use_meta_skill",
    "use_skill_aware_reflection": "optimizer.use_skill_aware_reflection",
    "skill_aware_appendix_source": "optimizer.skill_aware_appendix_source",
    "skill_aware_consolidate_threshold": "optimizer.skill_aware_consolidate_threshold",
    "use_gate": "evaluation.use_gate",
    "sel_env_num": "evaluation.sel_env_num",
    "test_env_num": "evaluation.test_env_num",
    "eval_test": "evaluation.eval_test",
    "env": "env.name",
    "skill_init": "env.skill_init",
    "out_root": "env.out_root",
}


def load_config(args: argparse.Namespace) -> dict:
    """Load config with _base_ inheritance, then apply CLI overrides."""
    from skillopt.config import load_config as _load, flatten_config, is_structured

    cfg = _load(args.config, overrides=args.cfg_options)
    structured = is_structured(cfg)

    # Apply legacy --key value overrides
    cli = {k: v for k, v in vars(args).items()
           if v is not None and k not in ("config", "cfg_options")}
    if cli:
        if structured:
            from skillopt.config import apply_overrides
            mapped = []
            for k, v in cli.items():
                dotted = _LEGACY_TO_STRUCTURED.get(k)
                if dotted:
                    mapped.append(f"{dotted}={v}")
                else:
                    mapped.append(f"env.{k}={v}")
            apply_overrides(cfg, mapped)
        else:
            cfg.update(cli)

    # Flatten structured config → flat dict for trainer/adapter
    flat = flatten_config(cfg) if structured else cfg

    for new_key, old_key in (
        ("azure_openai_endpoint", "azure_endpoint"),
        ("azure_openai_api_version", "azure_api_version"),
        ("azure_openai_api_key", "azure_api_key"),
    ):
        if flat.get(new_key) in (None, "") and flat.get(old_key) not in (None, ""):
            flat[new_key] = flat[old_key]

    explicit_backend = getattr(args, "backend", None)
    if explicit_backend is None:
        for option in args.cfg_options or []:
            key = str(option).split("=", 1)[0].strip()
            if key == "model.backend":
                explicit_backend = str(option).split("=", 1)[1].strip()
                break

    backend = normalize_backend_name(flat.get("model_backend") or flat.get("target_backend") or "azure_openai")

    def _has_model_override(dotted_key: str, legacy_key: str) -> bool:
        if getattr(args, legacy_key, None) is not None:
            return True
        for option in args.cfg_options or []:
            key = str(option).split("=", 1)[0].strip()
            if key == dotted_key:
                return True
        return False

    if explicit_backend is not None:
        backend = normalize_backend_name(explicit_backend)
        flat["model_backend"] = backend
        if backend in {"claude", "claude_chat"}:
            flat.setdefault("optimizer_backend", "claude_chat")
            flat.setdefault("target_backend", "claude_chat")
        elif backend in {"codex", "codex_exec"}:
            flat.setdefault("optimizer_backend", "openai_chat")
            flat.setdefault("target_backend", "codex_exec")
        elif backend == "claude_code_exec":
            flat.setdefault("optimizer_backend", "openai_chat")
            flat.setdefault("target_backend", "claude_code_exec")
        elif backend in {"qwen", "qwen_chat"}:
            flat.setdefault("optimizer_backend", "openai_chat")
            flat.setdefault("target_backend", "qwen_chat")
        elif backend in {"minimax", "minimax_chat"}:
            flat.setdefault("optimizer_backend", "openai_chat")
            flat.setdefault("target_backend", "minimax_chat")
        else:
            flat.setdefault("optimizer_backend", "openai_chat")
            flat.setdefault("target_backend", "openai_chat")
    else:
        flat.setdefault("optimizer_backend", "openai_chat")
        flat.setdefault("target_backend", "openai_chat")

    if flat.get("optimizer_backend") == "claude_chat":
        if (
            str(flat.get("optimizer_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.optimizer", "optimizer_model")
        ):
            flat["optimizer_model"] = default_model_for_backend("claude_chat")
    if flat.get("optimizer_backend") == "qwen_chat":
        if (
            str(flat.get("optimizer_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.optimizer", "optimizer_model")
        ):
            flat["optimizer_model"] = default_model_for_backend("qwen_chat")
    if flat.get("target_backend") == "claude_chat":
        if (
            str(flat.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            flat["target_model"] = default_model_for_backend("claude_chat")
    if flat.get("target_backend") == "claude_code_exec":
        if (
            str(flat.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            flat["target_model"] = default_model_for_backend("claude_chat")
    if flat.get("target_backend") == "qwen_chat":
        if (
            str(flat.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            flat["target_model"] = default_model_for_backend("qwen_chat")
    if flat.get("target_backend") == "minimax_chat":
        if (
            str(flat.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            flat["target_model"] = (
                flat.get("minimax_model")
                or default_model_for_backend("minimax_chat")
            )

    # Auto-generate output root
    if not flat.get("out_root"):
        env = flat.get("env", "unknown")
        model = flat.get("optimizer_model", "unknown").replace("/", "-")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        flat["out_root"] = os.path.join("outputs", f"skillopt_{env}_{model}_{ts}")

    flat["out_root"] = os.path.abspath(flat["out_root"])
    return flat


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    print(f"\n{'='*60}")
    print(f"  SkillOpt — Executive Strategy for Self-Evolving Agent Skills")
    print(f"{'='*60}")
    print(f"  env:            {cfg.get('env')}")
    print(f"  optimizer_model:  {cfg.get('optimizer_model')}")
    print(f"  target_model:  {cfg.get('target_model')}")
    print(f"  optimizer_backend:{cfg.get('optimizer_backend', 'openai_chat')}")
    print(f"  target_backend:{cfg.get('target_backend', 'openai_chat')}")
    print(f"  reasoning:      {cfg.get('reasoning_effort') or 'off'}")
    print(f"  rewrite_effort: {cfg.get('rewrite_reasoning_effort') or 'off'}")
    print(f"  epochs:         {cfg.get('num_epochs')}")
    print(f"  train_size:     {cfg.get('train_size') or 'from dataset'}")
    print(f"  steps/epoch:    auto")
    print(f"  batch_size:     {cfg.get('batch_size')}")
    print(f"  edit_budget:    {cfg.get('edit_budget')}")
    print(f"  lr_scheduler:   {cfg.get('lr_scheduler', 'constant')}")
    print(f"  update_mode:    {cfg.get('skill_update_mode', 'patch')}")
    print(f"  min_edit_budget:{cfg.get('min_edit_budget', 2)}")
    print(f"  minibatch_size: {cfg.get('minibatch_size')}")
    print(f"  seed:           {cfg.get('seed')}")
    print(f"  meta_skill:     {cfg.get('use_meta_skill', False)}")
    print(f"  skill_aware_reflection: {cfg.get('use_skill_aware_reflection', False)}")
    print(f"  slow_update:    {cfg.get('use_slow_update', False)}")
    print(f"  out_root:       {cfg.get('out_root')}")
    print(f"{'='*60}\n")

    # Build adapter
    adapter = get_adapter(cfg)

    # Build trainer and run
    from skillopt.engine.trainer import ReflACTTrainer
    trainer = ReflACTTrainer(cfg, adapter)
    summary = trainer.train()

    print(f"\n  Output saved to: {cfg['out_root']}")
    if summary.get("test_hard") is not None:
        print(f"  Final test: {summary['test_hard']:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SkillOpt eval-only: run a single skill on a dataset without training.

Usage
-----
    python scripts/eval_only.py \
        --config configs/spreadsheetbench/default.yaml \
        --skill skillopt/envs/spreadsheetbench/skills/initial.md \
        --split_dir /path/to/split \
        --out_root outputs/eval_skill0

All YAML keys can be overridden from the CLI, same as train.py.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Progress output uses box-drawing and arrow characters that crash a cp1252
# console/redirection mid-run; force UTF-8 before any of it is written.
from skillopt.utils.console import force_utf8_stdout_stderr

force_utf8_stdout_stderr()

from skillopt.model import (
    configure_azure_openai,
    configure_claude_code_exec,
    configure_codex_exec_from_config,
    configure_copilot_chat,
    configure_copilot_exec,
    configure_cursor_exec,
    configure_minimax_chat,
    configure_qwen_chat,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
)
from skillopt.model.common import default_model_for_backend, normalize_backend_name
from skillopt.utils import compute_score

_OPENAI_DEFAULT_MODEL_SENTINELS = {"gpt-5.4", "gpt-5.5"}
_ROLE_BACKEND_DEFAULTS = (None, "", "openai_chat")


def _set_role_if_default(
    cfg: dict,
    key: str,
    value: str,
    *,
    explicitly_overridden: bool,
) -> None:
    """Apply a high-level backend mapping without clobbering a named role."""
    if not explicitly_overridden and cfg.get(key) in _ROLE_BACKEND_DEFAULTS:
        cfg[key] = value


# ── Reuse registry from train.py ───────────────────────────────────────────

_ENV_REGISTRY: dict[str, type] = {}


def _register_builtins() -> None:
    try:
        from skillopt.envs.alfworld.adapter import ALFWorldAdapter
        _ENV_REGISTRY["alfworld"] = ALFWorldAdapter
    except ImportError:
        pass
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
    _register_builtins()
    env_name = cfg.get("env", "alfworld")
    if env_name not in _ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_name}'. "
            f"Available: {list(_ENV_REGISTRY.keys())}"
        )
    adapter_cls = _ENV_REGISTRY[env_name]

    import inspect
    sig = inspect.signature(adapter_cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    adapter_kwargs = {k: cfg[k] for k in accepted if k in cfg}
    return adapter_cls(**adapter_kwargs)


# ── CLI ────────────────────────────────────────────────────────────────────

_BOOL = lambda x: str(x).lower() in ("true", "1", "yes")  # noqa: E731


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SkillOpt eval-only")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--skill", type=str, required=True,
                   help="Path to skill .md file to evaluate")
    p.add_argument("--split", type=str, default="all",
                   help="Which split to eval: train/valid_seen/valid_unseen/all (default: all)")
    p.add_argument("--cfg-options", nargs="+", default=[],
                   help="Override config: section.key=value")
    # Legacy flat overrides
    p.add_argument("--env", type=str)
    p.add_argument("--backend", type=str,
                   choices=["azure_openai", "codex", "codex_exec", "claude", "claude_chat", "claude_code_exec", "cursor", "cursor_exec", "copilot", "copilot_chat", "copilot_exec", "minimax", "minimax_chat"])
    p.add_argument("--optimizer_model", type=str)
    p.add_argument("--target_model", type=str)
    p.add_argument("--optimizer_backend", type=str)
    p.add_argument("--target_backend", type=str)
    p.add_argument("--reasoning_effort", type=str,
                   choices=["", "low", "medium", "high", "xhigh", "max"])
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
    p.add_argument("--codex_exec_path", type=str)
    p.add_argument("--codex_exec_sandbox", type=str)
    p.add_argument("--codex_exec_profile", type=str)
    p.add_argument(
        "--codex_exec_full_auto",
        type=_BOOL,
        help=(
            "Deprecated and ignored; use --codex_exec_sandbox and "
            "--codex_exec_approval_policy"
        ),
    )
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
    p.add_argument("--cursor_exec_path", type=str)
    p.add_argument("--cursor_exec_sandbox", type=str)
    p.add_argument("--copilot_exec_path", type=str)
    p.add_argument("--copilot_exec_home", type=str)
    p.add_argument("--copilot_exec_allow_all_tools", type=_BOOL)
    p.add_argument("--copilot_chat_optimizer_model", type=str)
    p.add_argument("--copilot_chat_target_model", type=str)
    p.add_argument("--copilot_chat_timeout", type=int)
    p.add_argument("--minimax_region", type=str)
    p.add_argument("--minimax_base_url", type=str)
    p.add_argument("--minimax_api_key", type=str)
    p.add_argument("--minimax_model", type=str)
    p.add_argument("--minimax_temperature", type=float)
    p.add_argument("--minimax_max_tokens", type=int)
    p.add_argument("--minimax_enable_thinking", type=_BOOL)
    p.add_argument("--out_root", type=str)
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
    p.add_argument("--max_api_workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--test_env_num", type=int)
    p.add_argument("--mode", type=str,
                   help="SpreadsheetBench: single/multi/react (default comes from config)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from skillopt.config import load_config as _load, flatten_config, is_structured

    cfg = _load(args.config, overrides=args.cfg_options)
    structured = is_structured(cfg)

    # Apply legacy --key value overrides
    cli = {k: v for k, v in vars(args).items()
           if v is not None and k not in ("config", "skill", "split", "cfg_options")}
    if cli:
        if structured:
            from skillopt.config import apply_overrides
            _MAP = {
                "backend": "model.backend",
                "optimizer_model": "model.optimizer",
                "target_model": "model.target",
                "optimizer_backend": "model.optimizer_backend",
                "target_backend": "model.target_backend",
                "reasoning_effort": "model.reasoning_effort",
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
                "cursor_exec_path": "model.cursor_exec_path",
                "cursor_exec_sandbox": "model.cursor_exec_sandbox",
                "copilot_exec_path": "model.copilot_exec_path",
                "copilot_exec_home": "model.copilot_exec_home",
                "copilot_exec_allow_all_tools": "model.copilot_exec_allow_all_tools",
                "copilot_chat_optimizer_model": "model.copilot_chat_optimizer_model",
                "copilot_chat_target_model": "model.copilot_chat_target_model",
                "copilot_chat_timeout": "model.copilot_chat_timeout",
                "minimax_region": "model.minimax_region",
                "minimax_base_url": "model.minimax_base_url",
                "minimax_api_key": "model.minimax_api_key",
                "minimax_model": "model.minimax_model",
                "minimax_temperature": "model.minimax_temperature",
                "minimax_max_tokens": "model.minimax_max_tokens",
                "minimax_enable_thinking": "model.minimax_enable_thinking",
                "seed": "train.seed",
                "test_env_num": "evaluation.test_env_num",
                "env": "env.name",
                "out_root": "env.out_root",
            }
            mapped = []
            for k, v in cli.items():
                dotted = _MAP.get(k)
                if dotted:
                    mapped.append(f"{dotted}={v}")
                else:
                    mapped.append(f"env.{k}={v}")
            apply_overrides(cfg, mapped)
        else:
            cfg.update(cli)

    cfg = flatten_config(cfg) if structured else cfg

    for new_key, old_key in (
        ("azure_openai_endpoint", "azure_endpoint"),
        ("azure_openai_api_version", "azure_api_version"),
        ("azure_openai_api_key", "azure_api_key"),
    ):
        if cfg.get(new_key) in (None, "") and cfg.get(old_key) not in (None, ""):
            cfg[new_key] = cfg[old_key]

    explicit_backend = getattr(args, "backend", None)
    if explicit_backend is None:
        for option in args.cfg_options or []:
            key = str(option).split("=", 1)[0].strip()
            if key == "model.backend":
                explicit_backend = str(option).split("=", 1)[1].strip()
                break
    if explicit_backend is None:
        # ``model.backend`` from a structured YAML file is flattened to
        # ``model_backend``.  Treat it like the equivalent CLI selection so
        # its role mapping (and backend-specific model defaults) is applied.
        explicit_backend = cfg.get("model_backend") or cfg.get("backend")

    backend = normalize_backend_name(cfg.get("model_backend") or cfg.get("target_backend") or "azure_openai")

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
        cfg["model_backend"] = backend

        def _set_role(key: str, value: str) -> None:
            _set_role_if_default(
                cfg,
                key,
                value,
                explicitly_overridden=_has_model_override(f"model.{key}", key),
            )

        if backend == "claude_chat":
            _set_role("optimizer_backend", "claude_chat")
            _set_role("target_backend", "claude_chat")
        elif backend in {"codex", "codex_exec"}:
            _set_role("optimizer_backend", "codex_exec")
            _set_role("target_backend", "codex_exec")
        elif backend == "claude_code_exec":
            # Both roles default to Claude Code so reflection sees the full
            # trajectory. A role pinned to a non-default value (e.g. minimax_chat)
            # still overrides; an explicit --optimizer_backend openai_chat does
            # not, because openai_chat is one of the base-config defaults.
            _set_role("optimizer_backend", "claude_code_exec")
            _set_role("target_backend", "claude_code_exec")
        elif backend == "cursor_exec":
            _set_role("optimizer_backend", "openai_chat")
            _set_role("target_backend", "cursor_exec")
        elif backend == "copilot_chat":
            # Both roles use the locally installed, CLI-authenticated backend.
            _set_role("optimizer_backend", "copilot_chat")
            _set_role("target_backend", "copilot_chat")
        elif backend == "copilot_exec":
            _set_role("optimizer_backend", "openai_chat")
            _set_role("target_backend", "copilot_exec")
        elif backend == "minimax_chat":
            _set_role("optimizer_backend", "openai_chat")
            _set_role("target_backend", "minimax_chat")
        elif backend == "openai_compatible":
            _set_role("optimizer_backend", "openai_compatible")
            _set_role("target_backend", "openai_compatible")
        else:
            _set_role("optimizer_backend", "openai_chat")
            _set_role("target_backend", "openai_chat")
    else:
        cfg.setdefault("optimizer_backend", "openai_chat")
        cfg.setdefault("target_backend", "openai_chat")

    if cfg.get("optimizer_backend") == "claude_chat":
        if (
            str(cfg.get("optimizer_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.optimizer", "optimizer_model")
        ):
            cfg["optimizer_model"] = default_model_for_backend("claude_chat")
    if cfg.get("target_backend") == "claude_chat":
        if (
            str(cfg.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            cfg["target_model"] = default_model_for_backend("claude_chat")
    if cfg.get("target_backend") == "claude_code_exec":
        if (
            str(cfg.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            cfg["target_model"] = default_model_for_backend("claude_chat")
    if cfg.get("target_backend") == "cursor_exec":
        if (
            str(cfg.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            cfg["target_model"] = default_model_for_backend("cursor_exec")
    if cfg.get("target_backend") == "copilot_exec":
        if (
            str(cfg.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            # Copilot CLI model IDs are independent of Azure deployment names.
            cfg["target_model"] = ""
    if cfg.get("target_backend") == "minimax_chat":
        if (
            str(cfg.get("target_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.target", "target_model")
        ):
            cfg["target_model"] = (
                cfg.get("minimax_model")
                or default_model_for_backend("minimax_chat")
            )

    if not cfg.get("out_root"):
        env = cfg.get("env", "unknown")
        model = cfg.get("target_model", "unknown").replace("/", "-")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["out_root"] = os.path.join("outputs", f"eval_{env}_{model}_{ts}")

    cfg["out_root"] = os.path.abspath(cfg["out_root"])

    out_root = cfg["out_root"]
    os.makedirs(out_root, exist_ok=True)

    # Load skill
    skill_path = os.path.abspath(args.skill)
    with open(skill_path) as f:
        skill_content = f.read()
    print(f"  [skill] {skill_path} ({len(skill_content)} chars)")

    # Configure models
    configure_azure_openai(
        endpoint=(cfg.get("azure_openai_endpoint") or cfg.get("azure_endpoint") or None),
        api_version=(cfg.get("azure_openai_api_version") or cfg.get("azure_api_version") or None),
        api_key=(cfg.get("azure_openai_api_key") or cfg.get("azure_api_key") or None),
        auth_mode=cfg.get("azure_openai_auth_mode") or None,
        ad_scope=cfg.get("azure_openai_ad_scope") or None,
        managed_identity_client_id=cfg.get("azure_openai_managed_identity_client_id") or None,
        optimizer_endpoint=cfg.get("optimizer_azure_openai_endpoint") or None,
        optimizer_api_version=cfg.get("optimizer_azure_openai_api_version") or None,
        optimizer_api_key=cfg.get("optimizer_azure_openai_api_key") or None,
        optimizer_auth_mode=cfg.get("optimizer_azure_openai_auth_mode") or None,
        optimizer_ad_scope=cfg.get("optimizer_azure_openai_ad_scope") or None,
        optimizer_managed_identity_client_id=(
            cfg.get("optimizer_azure_openai_managed_identity_client_id") or None
        ),
        target_endpoint=cfg.get("target_azure_openai_endpoint") or None,
        target_api_version=cfg.get("target_azure_openai_api_version") or None,
        target_api_key=cfg.get("target_azure_openai_api_key") or None,
        target_auth_mode=cfg.get("target_azure_openai_auth_mode") or None,
        target_ad_scope=cfg.get("target_azure_openai_ad_scope") or None,
        target_managed_identity_client_id=(
            cfg.get("target_azure_openai_managed_identity_client_id") or None
        ),
    )
    set_optimizer_backend(cfg.get("optimizer_backend", "openai_chat"))
    set_target_backend(cfg.get("target_backend", "openai_chat"))
    set_optimizer_deployment(cfg.get("optimizer_model", default_model_for_backend(backend)))
    set_target_deployment(cfg.get("target_model", default_model_for_backend(backend)))
    configure_codex_exec_from_config(cfg)
    configure_claude_code_exec(
        path=cfg.get("claude_code_exec_path", "claude"),
        profile=cfg.get("claude_code_exec_profile", ""),
        use_sdk=cfg.get("claude_code_exec_use_sdk", None),
        effort=cfg.get("claude_code_exec_effort", cfg.get("reasoning_effort", "medium")),
        max_thinking_tokens=cfg.get("claude_code_exec_max_thinking_tokens", 16384),
    )
    configure_cursor_exec(
        path=cfg.get("cursor_exec_path") or None,
        sandbox=cfg.get("cursor_exec_sandbox") or None,
    )
    configure_copilot_exec(
        path=cfg.get("copilot_exec_path") or None,
        home=cfg.get("copilot_exec_home") or None,
        allow_all_tools=cfg.get("copilot_exec_allow_all_tools"),
    )
    configure_copilot_chat(
        optimizer_model=cfg.get("copilot_chat_optimizer_model") or None,
        target_model=cfg.get("copilot_chat_target_model") or None,
        timeout=cfg.get("copilot_chat_timeout") or None,
    )
    configure_qwen_chat(
        base_url=cfg.get("qwen_chat_base_url") or None,
        api_key=cfg.get("qwen_chat_api_key") or None,
        temperature=cfg.get("qwen_chat_temperature"),
        timeout_seconds=cfg.get("qwen_chat_timeout_seconds"),
        max_tokens=cfg.get("qwen_chat_max_tokens"),
        enable_thinking=cfg.get("qwen_chat_enable_thinking"),
        target_base_url=cfg.get("target_qwen_chat_base_url") or None,
        target_api_key=cfg.get("target_qwen_chat_api_key") or None,
        target_temperature=cfg.get("target_qwen_chat_temperature"),
        target_timeout_seconds=cfg.get("target_qwen_chat_timeout_seconds"),
        target_max_tokens=cfg.get("target_qwen_chat_max_tokens"),
        target_enable_thinking=cfg.get("target_qwen_chat_enable_thinking"),
    )
    configure_minimax_chat(
        region=cfg.get("minimax_region") or None,
        base_url=cfg.get("minimax_base_url") or None,
        api_key=cfg.get("minimax_api_key") or None,
        temperature=cfg.get("minimax_temperature"),
        max_tokens=cfg.get("minimax_max_tokens"),
        enable_thinking=cfg.get("minimax_enable_thinking"),
    )
    minimax_model_cfg = cfg.get("minimax_model")
    if minimax_model_cfg and cfg.get("target_backend") == "minimax_chat":
        set_target_deployment(str(minimax_model_cfg))
    set_reasoning_effort(cfg.get("reasoning_effort", "") or None)

    # Build adapter
    adapter = get_adapter(cfg)
    adapter.setup(cfg)

    seed = cfg.get("seed", 42)
    split = args.split or "all"

    if split == "all":
        items = (
            adapter.build_eval_env(0, "train", seed)
            + adapter.build_eval_env(0, "valid_seen", seed)
            + adapter.build_eval_env(0, "valid_unseen", seed)
        )
    else:
        env_num = cfg.get("test_env_num", 0)
        items = adapter.build_eval_env(env_num, split, seed)

    print(f"\n  [eval] split={split}  items={len(items)}")
    print(f"  [eval] out_root={out_root}")
    print(f"{'='*60}")

    # Run rollout
    results = adapter.rollout(items, skill_content, out_root)

    # Score
    hard, soft = compute_score(results)
    print(f"\n{'='*60}")
    print(f"  Results: hard={hard:.4f}  soft={soft:.4f}  (n={len(results)})")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "skill": skill_path,
        "split": split,
        "n_items": len(results),
        "hard": hard,
        "soft": soft,
    }
    with open(os.path.join(out_root, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {out_root}")


if __name__ == "__main__":
    main()

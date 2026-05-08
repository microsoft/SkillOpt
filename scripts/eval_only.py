#!/usr/bin/env python3
"""ReflACT eval-only: run a single skill on a dataset without training.

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

from skillopt.model import (
    configure_azure_openai,
    configure_claude_code_exec,
    configure_codex_exec,
    set_reasoning_effort,
    set_student_backend,
    set_student_deployment,
    set_teacher_backend,
    set_teacher_deployment,
)
from skillopt.model.common import default_model_for_backend, normalize_backend_name

_OPENAI_DEFAULT_MODEL_SENTINELS = {"gpt-5.4", "gpt-5.5"}
from skillopt.utils import compute_score


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
    p = argparse.ArgumentParser(description="ReflACT eval-only")
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
                   choices=["azure_openai", "codex", "codex_exec", "claude", "claude_chat", "claude_code_exec"])
    p.add_argument("--teacher_model", type=str)
    p.add_argument("--student_model", type=str)
    p.add_argument("--teacher_backend", type=str)
    p.add_argument("--student_backend", type=str)
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
    p.add_argument("--teacher_azure_openai_endpoint", type=str)
    p.add_argument("--teacher_azure_openai_api_version", type=str)
    p.add_argument("--teacher_azure_openai_api_key", type=str)
    p.add_argument("--teacher_azure_openai_auth_mode", type=str)
    p.add_argument("--teacher_azure_openai_ad_scope", type=str)
    p.add_argument("--teacher_azure_openai_managed_identity_client_id", type=str)
    p.add_argument("--student_azure_openai_endpoint", type=str)
    p.add_argument("--student_azure_openai_api_version", type=str)
    p.add_argument("--student_azure_openai_api_key", type=str)
    p.add_argument("--student_azure_openai_auth_mode", type=str)
    p.add_argument("--student_azure_openai_ad_scope", type=str)
    p.add_argument("--student_azure_openai_managed_identity_client_id", type=str)
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
                "teacher_model": "model.teacher",
                "student_model": "model.student",
                "teacher_backend": "model.teacher_backend",
                "student_backend": "model.student_backend",
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
                "teacher_azure_openai_endpoint": "model.teacher_azure_openai_endpoint",
                "teacher_azure_openai_api_version": "model.teacher_azure_openai_api_version",
                "teacher_azure_openai_api_key": "model.teacher_azure_openai_api_key",
                "teacher_azure_openai_auth_mode": "model.teacher_azure_openai_auth_mode",
                "teacher_azure_openai_ad_scope": "model.teacher_azure_openai_ad_scope",
                "teacher_azure_openai_managed_identity_client_id": "model.teacher_azure_openai_managed_identity_client_id",
                "student_azure_openai_endpoint": "model.student_azure_openai_endpoint",
                "student_azure_openai_api_version": "model.student_azure_openai_api_version",
                "student_azure_openai_api_key": "model.student_azure_openai_api_key",
                "student_azure_openai_auth_mode": "model.student_azure_openai_auth_mode",
                "student_azure_openai_ad_scope": "model.student_azure_openai_ad_scope",
                "student_azure_openai_managed_identity_client_id": "model.student_azure_openai_managed_identity_client_id",
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

    backend = normalize_backend_name(cfg.get("model_backend") or cfg.get("student_backend") or "azure_openai")

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
        if backend in {"claude", "claude_chat"}:
            cfg.setdefault("teacher_backend", "claude_chat")
            cfg.setdefault("student_backend", "claude_chat")
        elif backend in {"codex", "codex_exec"}:
            cfg.setdefault("teacher_backend", "openai_chat")
            cfg.setdefault("student_backend", "codex_exec")
        elif backend == "claude_code_exec":
            cfg.setdefault("teacher_backend", "openai_chat")
            cfg.setdefault("student_backend", "claude_code_exec")
        else:
            cfg.setdefault("teacher_backend", "openai_chat")
            cfg.setdefault("student_backend", "openai_chat")
    else:
        cfg.setdefault("teacher_backend", "openai_chat")
        cfg.setdefault("student_backend", "openai_chat")

    if cfg.get("teacher_backend") == "claude_chat":
        if (
            str(cfg.get("teacher_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.teacher", "teacher_model")
        ):
            cfg["teacher_model"] = default_model_for_backend("claude_chat")
    if cfg.get("student_backend") == "claude_chat":
        if (
            str(cfg.get("student_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.student", "student_model")
        ):
            cfg["student_model"] = default_model_for_backend("claude_chat")
    if cfg.get("student_backend") == "claude_code_exec":
        if (
            str(cfg.get("student_model", "") or "").strip() in _OPENAI_DEFAULT_MODEL_SENTINELS
            and not _has_model_override("model.student", "student_model")
        ):
            cfg["student_model"] = default_model_for_backend("claude_chat")

    if not cfg.get("out_root"):
        env = cfg.get("env", "unknown")
        model = cfg.get("student_model", "unknown").replace("/", "-")
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
        teacher_endpoint=cfg.get("teacher_azure_openai_endpoint") or None,
        teacher_api_version=cfg.get("teacher_azure_openai_api_version") or None,
        teacher_api_key=cfg.get("teacher_azure_openai_api_key") or None,
        teacher_auth_mode=cfg.get("teacher_azure_openai_auth_mode") or None,
        teacher_ad_scope=cfg.get("teacher_azure_openai_ad_scope") or None,
        teacher_managed_identity_client_id=(
            cfg.get("teacher_azure_openai_managed_identity_client_id") or None
        ),
        student_endpoint=cfg.get("student_azure_openai_endpoint") or None,
        student_api_version=cfg.get("student_azure_openai_api_version") or None,
        student_api_key=cfg.get("student_azure_openai_api_key") or None,
        student_auth_mode=cfg.get("student_azure_openai_auth_mode") or None,
        student_ad_scope=cfg.get("student_azure_openai_ad_scope") or None,
        student_managed_identity_client_id=(
            cfg.get("student_azure_openai_managed_identity_client_id") or None
        ),
    )
    set_teacher_backend(cfg.get("teacher_backend", "openai_chat"))
    set_student_backend(cfg.get("student_backend", "openai_chat"))
    set_teacher_deployment(cfg.get("teacher_model", default_model_for_backend(backend)))
    set_student_deployment(cfg.get("student_model", default_model_for_backend(backend)))
    configure_codex_exec(
        path=cfg.get("codex_exec_path", "codex"),
        sandbox=cfg.get("codex_exec_sandbox", "workspace-write"),
        profile=cfg.get("codex_exec_profile", ""),
        full_auto=cfg.get("codex_exec_full_auto", False),
        reasoning_effort=cfg.get("codex_exec_reasoning_effort", "none"),
        use_sdk=cfg.get("codex_exec_use_sdk", None),
        network_access=cfg.get("codex_exec_network_access", False),
        web_search=cfg.get("codex_exec_web_search", False),
        approval_policy=cfg.get("codex_exec_approval_policy", "never"),
    )
    configure_claude_code_exec(
        path=cfg.get("claude_code_exec_path", "claude"),
        profile=cfg.get("claude_code_exec_profile", ""),
        use_sdk=cfg.get("claude_code_exec_use_sdk", None),
        effort=cfg.get("claude_code_exec_effort", cfg.get("reasoning_effort", "medium")),
        max_thinking_tokens=cfg.get("claude_code_exec_max_thinking_tokens", 16384),
    )
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

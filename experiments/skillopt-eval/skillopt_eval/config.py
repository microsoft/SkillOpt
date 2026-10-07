"""Experiment configuration loading and validation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

RUNNERS = ("mock", "superpowers")


class ConfigError(ValueError):
    """Raised when the experiment config is missing or invalid."""


@dataclass
class ExperimentConfig:
    """One controlled baseline-vs-candidate experiment."""

    name: str
    skill: str
    runner: str = "mock"

    # Provenance / identical-settings contract. Recorded verbatim and passed to
    # the runner; both conditions MUST run under the same values.
    model: str = "mock-deterministic"
    model_settings: Dict[str, Any] = field(default_factory=dict)
    superpowers_sha: str = ""
    seed: int = 0
    timeout_seconds: int = 120
    token_cap: int = 0  # 0 = no cap

    # Scenario splitting. Counts must cover the available scenario ids.
    train: int = 0
    validation: int = 0
    test: int = 0
    seeds_per_task: int = 1

    # Optimization loop.
    optimize_steps: int = 3
    learning_rate: float = 1.0  # max fraction of skill text a single edit may touch
    candidates: List[str] = field(default_factory=list)  # optional pre-authored candidate files

    baseline_skill: str = ""  # path to baseline SKILL.md; required for mock
    output_dir: str = "results"

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {p}: {exc}") from None
        if not isinstance(raw, dict):
            raise ConfigError(f"config {p} must be a YAML mapping")
        cfg = ExperimentConfig._from_dict(raw)
        cfg._resolve_paths(p.parent)
        return cfg

    @staticmethod
    def _from_dict(raw: Dict[str, Any]) -> "ExperimentConfig":
        def req(key: str) -> Any:
            if key not in raw or raw[key] is None:
                raise ConfigError(f"missing required config key: {key}")
            return raw[key]

        cfg = ExperimentConfig(name=str(req("name")), skill=str(req("skill")))
        cfg.runner = str(raw.get("runner", cfg.runner))
        if cfg.runner not in RUNNERS:
            raise ConfigError(f"runner must be one of {RUNNERS}, got {cfg.runner!r}")
        cfg.model = str(raw.get("model", cfg.model))
        settings = raw.get("model_settings", {})
        if not isinstance(settings, dict):
            raise ConfigError("model_settings must be a mapping")
        cfg.model_settings = settings
        cfg.superpowers_sha = str(raw.get("superpowers_sha", ""))
        if cfg.runner == "superpowers":
            import re
            if not re.fullmatch(r"[0-9a-f]{40}", cfg.superpowers_sha):
                raise ConfigError(
                    "runner=superpowers requires superpowers_sha (full 40-char commit)"
                )
        for key in ("seed", "timeout_seconds", "token_cap", "train",
                    "validation", "test", "seeds_per_task", "optimize_steps"):
            value = raw.get(key, getattr(cfg, key))
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{key} must be an integer")
            setattr(cfg, key, value)
        if cfg.seeds_per_task < 1:
            raise ConfigError("seeds_per_task must be >= 1")
        if cfg.test < 1:
            raise ConfigError("test split must contain at least one scenario")
        if cfg.validation < 1 and cfg.optimize_steps > 0:
            raise ConfigError(
                "optimize_steps > 0 requires a non-empty validation split"
            )
        lr = raw.get("learning_rate", cfg.learning_rate)
        if isinstance(lr, bool) or not isinstance(lr, (int, float)):
            raise ConfigError("learning_rate must be a number")
        cfg.learning_rate = float(lr)
        if not math.isfinite(cfg.learning_rate) or not 0 < cfg.learning_rate <= 1:
            raise ConfigError("learning_rate must be in (0, 1]")
        candidates = raw.get("candidates", [])
        if not isinstance(candidates, list) or not all(
            isinstance(c, str) for c in candidates
        ):
            raise ConfigError("candidates must be a list of file paths")
        cfg.candidates = candidates
        cfg.baseline_skill = str(raw.get("baseline_skill", ""))
        if cfg.runner == "mock" and not cfg.baseline_skill:
            raise ConfigError("runner=mock requires baseline_skill (path to SKILL.md)")
        cfg.output_dir = str(raw.get("output_dir", cfg.output_dir))
        return cfg

    def _resolve_paths(self, base: Path) -> None:
        for attr in ("baseline_skill", "output_dir"):
            value = getattr(self, attr)
            if value and not Path(value).is_absolute():
                setattr(self, attr, str((base / value).resolve()))
        self.candidates = [
            str((base / c).resolve()) if not Path(c).is_absolute() else c
            for c in self.candidates
        ]

"""Regression tests for ``env:`` section survival through _base_ inheritance.

``_FLATTEN_MAP`` maps ``env.name`` to the flat key ``env``, which collides with
the name of the ``env:`` section itself.  Pruning that "duplicate" flat key
blindly deleted the whole section, so every shipped benchmark config lost
``env.name``, ``env.skill_init``, and ``env.split_dir`` at load time -- and
``scripts/train.py`` silently fell back to its ``alfworld`` default.
"""
from pathlib import Path

import pytest

from skillopt.config import flatten_config, load_config

_CONFIGS_DIR = Path(__file__).parents[1] / "configs"

_EXPECTED_ENV_NAMES = {
    "alfworld": "alfworld",
    "docvqa": "docvqa",
    "livemathematicianbench": "livemathematicianbench",
    "officeqa": "officeqa",
    "searchqa": "searchqa",
    "spreadsheetbench": "spreadsheetbench",
}


@pytest.mark.parametrize("bench,expected", sorted(_EXPECTED_ENV_NAMES.items()))
def test_shipped_config_keeps_env_section(bench: str, expected: str) -> None:
    flat = flatten_config(load_config(str(_CONFIGS_DIR / bench / "default.yaml")))
    assert flat["env"] == expected
    assert flat["skill_init"], f"{bench}: skill_init was dropped"
    assert flat["skill_init"].endswith(".md")


def test_child_section_omitting_a_key_does_not_delete_it(tmp_path: Path) -> None:
    """A child ``env:`` block without ``name`` must inherit the base's name."""
    base = tmp_path / "base.yaml"
    base.write_text(
        "env:\n  name: searchqa\n  skill_init: seed.md\n  workers: 24\n"
        "train:\n  batch_size: 40\n"
    )
    child = tmp_path / "child.yaml"
    child.write_text("_base_: base.yaml\nenv:\n  workers: 8\n")

    flat = flatten_config(load_config(str(child)))
    assert flat["env"] == "searchqa"
    assert flat["skill_init"] == "seed.md"
    assert flat["workers"] == 8


def test_structured_child_still_overrides_base_flat_key(tmp_path: Path) -> None:
    """Cross-format precedence: a structured child value wins over a flat base."""
    base = tmp_path / "base.yaml"
    base.write_text("train:\n  seed: 1\nbatch_size: 40\n")
    child = tmp_path / "child.yaml"
    child.write_text("_base_: base.yaml\ntrain:\n  batch_size: 8\n")

    flat = flatten_config(load_config(str(child)))
    assert flat["batch_size"] == 8

"""Regression: a flat ``--cfg-options`` key for an ``env`` setting must win.

``apply_overrides`` stored an unmapped flat key such as ``workers=4`` at the
top level of a structured config. ``flatten_config`` copies top-level keys
first and the ``env`` section's keys after them, so the YAML's ``env.workers``
silently replaced the command-line value.
"""
from __future__ import annotations

from skillopt.config import flatten_config, load_config


def _write_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "train:\n"
        "  batch_size: 8\n"
        "env:\n"
        "  name: searchqa\n"
        "  workers: 24\n"
        "  max_turns: 1\n",
        encoding="utf-8",
    )
    return str(config)


def test_flat_override_of_env_key_wins_over_yaml(tmp_path):
    cfg = load_config(_write_config(tmp_path), overrides=["workers=4", "max_turns=3"])
    flat = flatten_config(cfg)
    assert flat["workers"] == 4
    assert flat["max_turns"] == 3


def test_flat_and_dotted_env_overrides_agree(tmp_path):
    path = _write_config(tmp_path)
    flat_form = flatten_config(load_config(path, overrides=["workers=4"]))
    dotted_form = flatten_config(load_config(path, overrides=["env.workers=4"]))
    assert flat_form == dotted_form


def test_new_flat_key_still_reaches_the_trainer(tmp_path):
    cfg = load_config(_write_config(tmp_path), overrides=["limit=16"])
    assert flatten_config(cfg)["limit"] == 16

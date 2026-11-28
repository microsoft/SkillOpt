"""`--slow_update_gate_with_selection` reaches the optimizer setting, not env."""
from __future__ import annotations

import sys

import pytest

import scripts.train as train_script


@pytest.mark.parametrize("value, expected", [("true", True), ("false", False)])
def test_flag_maps_to_optimizer_setting(monkeypatch, value, expected) -> None:
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--config", "configs/_base_/default.yaml",
        "--slow_update_gate_with_selection", value,
    ])
    cfg = train_script.load_config(train_script.parse_args())
    assert cfg["slow_update_gate_with_selection"] is expected


def test_flag_absent_keeps_config_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "configs/_base_/default.yaml"])
    cfg = train_script.load_config(train_script.parse_args())
    assert cfg["slow_update_gate_with_selection"] is False

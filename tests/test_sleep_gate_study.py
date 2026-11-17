"""The gate study harness, offline: the committed fixture, a deterministic
stand-in model, and the arithmetic the report prints."""
from __future__ import annotations

import json

import pytest

from skillopt_sleep.experiments import gate_study


def _run(path, *argv):
    return gate_study.main([*argv, "--offline", "--store", str(path / "study.sqlite")])


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    path = tmp_path_factory.mktemp("gate_study")
    _run(path, "effects")
    _run(path, "faults", "--arm", "off")
    _run(path, "faults", "--arm", "check", "--feature", '{"gate_train_check": true}')
    store = gate_study.Store(str(path / "study.sqlite"))
    return gate_study.summarize(store, "fixture-model", gate_study.RULES["fixture"])


def test_fixture_effects_are_the_ones_the_fixture_encodes(study):
    assert study["starting_em"] == 75.0
    assert study["faults"]["F1 harmful"]["effect"] == -50.0
    assert study["faults"]["F2 helpful"]["effect"] == 25.0
    assert study["faults"]["F3 placebo"]["effect"] == 0.0


def test_shipped_gate_accepts_the_harmful_rule_in_some_contexts(study):
    off = study["faults"]["F1 harmful"]["arms"]["off"]
    assert off == EXPECTED["harmful_off"]
    assert off["accepted"] > 0


def test_train_check_removes_those_accepts_and_keeps_the_helpful_ones(study):
    faults = study["faults"]
    assert faults["F1 harmful"]["arms"]["check"] == EXPECTED["harmful_check"]
    assert faults["F2 helpful"]["arms"]["check"] == faults["F2 helpful"]["arms"]["off"]
    assert faults["F2 helpful"]["arms"]["off"] == EXPECTED["helpful"]
    assert faults["F3 placebo"]["arms"]["off"]["accepted"] == 0
    assert faults["F3 placebo"]["arms"]["check"]["accepted"] == 0


def test_the_check_costs_one_training_replay_per_passing_candidate(study):
    assert study["calls_per_decision"] == EXPECTED["calls_per_decision"]
    assert study["calls_per_decision"]["check"] > study["calls_per_decision"]["off"]


def test_a_model_command_needs_live_or_offline(tmp_path):
    with pytest.raises(SystemExit, match="--live"):
        gate_study.main(["faults", "--store", str(tmp_path / "s.sqlite")])


def test_report_arithmetic_on_known_rows(tmp_path, capsys):
    store = gate_study.Store(str(tmp_path / "rows.sqlite"))
    store.run("INSERT INTO effects VALUES ('m', '(starting skill)', 0.5)")
    store.run("INSERT INTO effects VALUES ('m', 'F1 harmful', 0.25)")
    for seed, accepted_off, accepted_on in ((1, 1, 0), (2, 1, 0), (3, 0, 0), (4, 0, 0)):
        store.run("INSERT INTO decisions VALUES ('off', 'm', ?, 1, 'F1 harmful', ?, 20)", (seed, accepted_off))
        store.run("INSERT INTO decisions VALUES ('on', 'm', ?, 1, 'F1 harmful', ?, 22)", (seed, accepted_on))
    for seed, ems in ((1, (0.5, 0.6, 0.7)), (2, (0.5, 0.5, 0.6))):
        for night, em in enumerate(ems):
            store.run("INSERT INTO curves VALUES ('base', 'm', ?, ?, ?, ?, ?)",
                      (seed, night, int(night > 0 and em > ems[night - 1]), em, 200 if night else 0))
    data = gate_study.summarize(store, "m", gate_study.RULES["fixture"])
    assert data["starting_em"] == 50.0
    assert data["faults"]["F1 harmful"] == {
        "effect": -25.0,
        "arms": {"off": {"accepted": 2, "decisions": 4}, "on": {"accepted": 0, "decisions": 4}},
    }
    assert data["faults"]["F2 helpful"] == {"effect": None, "arms": {}}
    assert data["calls_per_decision"] == {"off": 20.0, "on": 22.0}
    assert data["curves"]["base"] == {
        "seeds": 2, "curve": [50.0, 55.0, 65.0], "accepted_nights": 3,
        "first_accepted_night": 1.5, "target_calls_per_night": 200,
    }
    args = type("A", (), {"model": "m", "benchmark": "fixture", "json": True})()
    gate_study.report(args, store)
    assert json.loads(capsys.readouterr().out) == data


EXPECTED = {
    "harmful_off": {"accepted": 1, "decisions": 16},
    "harmful_check": {"accepted": 0, "decisions": 16},
    "helpful": {"accepted": 10, "decisions": 16},
    "calls_per_decision": {"check": 19.4, "off": 16.8},
}

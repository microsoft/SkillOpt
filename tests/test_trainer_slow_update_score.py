"""The force-written epoch-end slow update is re-measured on selection (#404).

These tests drive ``ReflACTTrainer.train()`` for three one-step epochs with a
mock adapter and stubbed optimizer calls, so no model or network is used.

Selection scores are keyed on skill content (``_score``): the step-2 skill
scores 0.8, the epoch-2 slow-update injection (``SLOW-A``) scores 0.3/0.9, so
a step-3 candidate at 0.55 is accepted only when the gate compares it against
the injected skill's real score (0.3) and not the stale pre-injection 0.8.
"""
import json
import os
from unittest import mock

import pytest

import skillopt.engine.trainer as trainer_mod
from skillopt.envs.base import EnvAdapter
from skillopt.evaluation.gate import select_gate_score

# (marker, (hard, soft)), checked in order; a skill with no marker is S_0.
_SCORES = (
    ("SLOW-B", (0.45, 0.2)),  # epoch-3 slow update injected
    ("rule 3", (0.55, 0.55)),  # step-3 candidate
    ("SLOW-A", (0.3, 0.9)),  # epoch-2 slow update injected
    ("rule 2", (0.8, 0.8)),  # step-2 candidate
    ("rule 1", (0.6, 0.6)),  # step-1 candidate
)
_INIT_SCORE = (0.5, 0.5)


def _score(skill, overrides=None):
    for marker, score in _SCORES:
        if marker in skill:
            return (overrides or {}).get(marker, score)
    return _INIT_SCORE


def _make_adapter(out_root, calls, *, overrides=None, crash_dir=None):
    adapter = mock.create_autospec(EnvAdapter, instance=True)
    adapter.requires_ray.return_value = False
    adapter.get_dataloader.return_value = None
    adapter.build_train_env.return_value = [{"id": "t0"}]
    adapter.build_eval_env.return_value = [{"id": "v0"}]
    adapter.reflect.return_value = [
        {"patch": {"edits": [{"op": "append", "content": "x"}]},
         "source_type": "failure"},
    ]

    def rollout(env, skill, out_dir, **kwargs):
        rel = os.path.relpath(out_dir, out_root).replace(os.sep, "/")
        if rel == crash_dir:
            raise RuntimeError("simulated crash")
        calls.append(rel)
        hard, soft = _score(skill, overrides)
        return [{"id": "t0", "hard": hard, "soft": soft}]

    adapter.rollout.side_effect = rollout
    return adapter


def _stub_optimizer(monkeypatch, *, slow_contents, unchanged_after="",
                    slow_calls=None):
    """Replace every optimizer-side call the loop makes with a stub.

    Each step appends ``- rule N`` (N = rules already present + 1), except
    that a skill containing ``unchanged_after`` is returned byte-identical.
    """
    monkeypatch.setattr(
        trainer_mod, "merge_patches",
        lambda *a, **k: {"edits": [{"op": "append", "content": "x"}]},
    )
    monkeypatch.setattr(
        trainer_mod, "rank_and_select", lambda skill, patch, **k: patch,
    )

    def apply_patch(skill, patch):
        if unchanged_after and unchanged_after in skill:
            return skill, []
        n = skill.count("- rule ") + 1
        return skill.rstrip() + f"\n- rule {n}\n", []

    monkeypatch.setattr(trainer_mod, "apply_patch_with_report", apply_patch)

    def run_slow_update(skill, results_prev, results_curr, items, **kwargs):
        epoch = os.path.basename(os.path.dirname(kwargs["prev_rollout_dir"]))
        if slow_calls is not None:
            slow_calls.append(epoch)
        return {"slow_update_content": slow_contents[epoch]}

    monkeypatch.setattr(trainer_mod, "run_slow_update", run_slow_update)
    monkeypatch.setattr(
        trainer_mod, "run_meta_skill",
        mock.Mock(side_effect=AssertionError("meta skill is off")),
    )


def _cfg(tmp_path, **overrides):
    skill_init = tmp_path / "skill_init.md"
    if not skill_init.exists():
        skill_init.write_text("# Skill\n", encoding="utf-8")
    cfg = {
        "out_root": str(tmp_path / "run"),
        "backend": "openai_chat",
        "optimizer_backend": "openai_chat",
        "target_backend": "openai_chat",
        "skill_init": str(skill_init),
        "num_epochs": 3,
        "train_size": 1,
        "batch_size": 1,
        "accumulation": 1,
        "merge_batch_size": 2,
        "edit_budget": 2,
        "seed": 42,
        "sel_env_num": 1,
        "eval_test": False,
        "analyst_workers": 1,
        "use_slow_update": True,
        "slow_update_samples": 1,
        "slow_update_gate_with_selection": False,
    }
    cfg.update(overrides)
    return cfg


def _read_json(out_root, *parts):
    with open(os.path.join(out_root, *parts), encoding="utf-8") as f:
        return json.load(f)


def _read_text(out_root, *parts):
    with open(os.path.join(out_root, *parts), encoding="utf-8") as f:
        return f.read()


_SLOW_AB = {"epoch_02": "SLOW-A guidance", "epoch_03": "SLOW-B guidance"}


def _expected_rollouts(slow_eval_dir):
    return [
        "selection_eval_baseline",
        "steps/step_0001/rollout",
        "steps/step_0001/selection_eval",
        "steps/step_0002/rollout",
        "steps/step_0002/selection_eval",
        "slow_update/epoch_02/rollout_prev",
        "slow_update/epoch_02/rollout_curr",
        f"slow_update/epoch_02/{slow_eval_dir}",
        "steps/step_0003/rollout",
        "steps/step_0003/selection_eval",
        "slow_update/epoch_03/rollout_prev",
        "slow_update/epoch_03/rollout_curr",
        f"slow_update/epoch_03/{slow_eval_dir}",
    ]


@pytest.mark.parametrize("use_gate", [True, False])
def test_force_write_remeasures_injected_skill(tmp_path, monkeypatch, use_gate):
    _stub_optimizer(monkeypatch, slow_contents=_SLOW_AB)
    cfg = _cfg(tmp_path, use_gate=use_gate)
    out_root = cfg["out_root"]
    calls: list[str] = []
    summary = trainer_mod.ReflACTTrainer(
        cfg, _make_adapter(out_root, calls),
    ).train()

    # Exactly one extra selection rollout per force-written epoch boundary.
    assert calls == _expected_rollouts("selection_eval_injected")

    slow2 = _read_json(out_root, "slow_update", "epoch_02", "slow_result.json")
    assert slow2["action"] == "force_accept"
    assert (slow2["selection_hard"], slow2["selection_soft"]) == (0.3, 0.9)
    assert slow2["current_score"] == 0.3

    # Step 3 is gated against the injected skill's 0.3, not the stale 0.8.
    step3 = _read_json(out_root, "steps", "step_0003", "step_record.json")
    assert step3["candidate_gate_score"] == 0.55
    assert step3["action"] == ("accept" if use_gate else "force_accept")
    assert step3["current_score"] == 0.55

    # The epoch-3 injection is re-measured too, and is the final current.
    state = _read_json(out_root, "runtime_state.json")
    assert state["current_score"] == 0.45
    assert state["current_origin"] == "slow_update_epoch_03"

    # best is untouched by force-injected content.
    assert (state["best_score"], state["best_step"]) == (0.8, 2)
    assert state["best_origin"] == "step_0002"
    best_skill = _read_text(out_root, "best_skill.md")
    assert "- rule 2" in best_skill
    assert "SLOW-A" not in best_skill and "SLOW-B" not in best_skill
    assert summary["best_selection_hard"] == 0.8
    assert summary["baseline_selection_hard"] == 0.5


def test_injected_skill_cache_holds_measured_hard_and_soft(tmp_path, monkeypatch):
    # After the epoch-2 injection the step-3 candidate is byte-identical to
    # it, and the epoch-3 slow update writes the same guidance again.
    _stub_optimizer(
        monkeypatch,
        slow_contents={"epoch_02": "SLOW-A guidance", "epoch_03": "SLOW-A guidance"},
        unchanged_after="SLOW-A",
    )
    cfg = _cfg(tmp_path, gate_metric="mixed", gate_mixed_weight=0.5)
    out_root = cfg["out_root"]
    calls: list[str] = []
    summary = trainer_mod.ReflACTTrainer(
        cfg, _make_adapter(out_root, calls),
    ).train()
    injected_score = select_gate_score(0.3, 0.9, "mixed", 0.5)

    slow2 = _read_json(out_root, "slow_update", "epoch_02", "slow_result.json")
    assert slow2["current_score"] == pytest.approx(injected_score)

    # Cache hit for the identical candidate returns the measured pair, not a
    # (gate_score, 0.0) placeholder.
    step3 = _read_json(out_root, "steps", "step_0003", "step_record.json")
    assert (step3["selection_hard"], step3["selection_soft"]) == (0.3, 0.9)
    assert step3["candidate_gate_score"] == pytest.approx(injected_score)
    assert step3["action"] == "reject"
    assert "steps/step_0003/selection_eval" not in calls

    # Same text injected again at epoch 3: genuine cache hit, no rollout.
    slow3 = _read_json(out_root, "slow_update", "epoch_03", "slow_result.json")
    assert (slow3["selection_hard"], slow3["selection_soft"]) == (0.3, 0.9)
    assert slow3["current_score"] == pytest.approx(injected_score)
    assert calls.count("slow_update/epoch_02/selection_eval_injected") == 1
    assert "slow_update/epoch_03/selection_eval_injected" not in calls

    assert summary["baseline_selection_hard"] == 0.5


@pytest.mark.parametrize(
    "slow_a, expected_action, expected_best",
    [
        ((0.3, 0.9), "reject", (0.8, 2, "step_0002")),
        ((0.9, 0.9), "accept_new_best", (0.9, 2, "slow_update_epoch_02")),
    ],
)
def test_gated_slow_update_is_unchanged(
    tmp_path, monkeypatch, slow_a, expected_action, expected_best,
):
    _stub_optimizer(monkeypatch, slow_contents=_SLOW_AB)
    cfg = _cfg(tmp_path, slow_update_gate_with_selection=True)
    out_root = cfg["out_root"]
    calls: list[str] = []
    trainer_mod.ReflACTTrainer(
        cfg, _make_adapter(out_root, calls, overrides={"SLOW-A": slow_a}),
    ).train()

    # One selection rollout per slow candidate, in the gated directory only.
    assert calls == _expected_rollouts("selection_eval")

    slow2 = _read_json(out_root, "slow_update", "epoch_02", "slow_result.json")
    assert slow2["action"] == expected_action
    assert (slow2["selection_hard"], slow2["selection_soft"]) == slow_a
    assert "current_score" not in slow2

    state = _read_json(out_root, "runtime_state.json")
    best_score, best_step, best_origin = expected_best
    assert (state["best_score"], state["best_step"]) == (best_score, best_step)
    assert state["best_origin"] == best_origin
    assert state["current_score"] == best_score
    assert ("SLOW-A" in _read_text(out_root, "best_skill.md")) == (
        expected_action == "accept_new_best"
    )
    step3 = _read_json(out_root, "steps", "step_0003", "step_record.json")
    assert step3["action"] == "reject"
    slow3 = _read_json(out_root, "slow_update", "epoch_03", "slow_result.json")
    assert slow3["action"] == "reject"


@pytest.mark.parametrize("stale_runtime_state", [False, True])
def test_resume_after_force_write_does_not_remeasure(
    tmp_path, monkeypatch, stale_runtime_state,
):
    slow_calls: list[str] = []
    _stub_optimizer(monkeypatch, slow_contents=_SLOW_AB, slow_calls=slow_calls)
    cfg = _cfg(tmp_path)
    out_root = cfg["out_root"]

    # First run stops at the start of epoch 3, after the epoch-2 slow update.
    first_calls: list[str] = []
    with pytest.raises(RuntimeError, match="simulated crash"):
        trainer_mod.ReflACTTrainer(
            dict(cfg),
            _make_adapter(out_root, first_calls,
                          crash_dir="steps/step_0003/rollout"),
        ).train()
    assert first_calls.count("slow_update/epoch_02/selection_eval_injected") == 1
    assert _read_json(out_root, "runtime_state.json")["current_score"] == 0.3

    if stale_runtime_state:
        # As if interrupted after slow_result.json but before the injected
        # skill and runtime_state.json were written.
        state = _read_json(out_root, "runtime_state.json")
        state["current_score"] = 0.8
        with open(os.path.join(out_root, "runtime_state.json"), "w") as f:
            json.dump(state, f)
        with open(os.path.join(out_root, "skills", "skill_v0002.md"), "w") as f:
            f.write(_read_text(out_root, "steps", "step_0002", "candidate_skill.md"))

    slow_calls.clear()
    resume_calls: list[str] = []
    trainer_mod.ReflACTTrainer(
        dict(cfg), _make_adapter(out_root, resume_calls),
    ).train()

    # No second rollout (or optimizer call) for the epoch-2 slow update.
    assert slow_calls == ["epoch_03"]
    assert resume_calls == [
        "steps/step_0003/rollout",
        "steps/step_0003/selection_eval",
        "slow_update/epoch_03/rollout_prev",
        "slow_update/epoch_03/rollout_curr",
        "slow_update/epoch_03/selection_eval_injected",
    ]
    # The resumed current skill carries the re-applied epoch-2 guidance and
    # is gated against its recorded 0.3, not the pre-injection 0.8.
    assert "SLOW-A" in _read_text(
        out_root, "steps", "step_0003", "candidate_skill.md",
    )
    step3 = _read_json(out_root, "steps", "step_0003", "step_record.json")
    assert step3["action"] == "accept"
    assert step3["current_score"] == 0.55
    assert _read_json(out_root, "runtime_state.json")["current_score"] == 0.45

    # A completed run resumes without any rollout at all.
    done_adapter = _make_adapter(out_root, [])
    trainer_mod.ReflACTTrainer(dict(cfg), done_adapter).train()
    done_adapter.rollout.assert_not_called()

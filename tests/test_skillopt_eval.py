"""Offline tests for experiments/skillopt-eval.

Deterministic: no network, no model calls, no POSIX requirement. Covers
splits, the validation gate (accept / reject / fail-closed), source-tree
non-mutation, artifact emission, and the A/A no-claim invariant required by
issue #132.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "skillopt-eval"
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from skillopt_eval.config import ConfigError, ExperimentConfig  # noqa: E402
from skillopt_eval.experiment import run_experiment  # noqa: E402
from skillopt_eval.mock_runner import MockRunner  # noqa: E402
from skillopt_eval.optimizer import FileOptimizer, RuleOptimizer  # noqa: E402
from skillopt_eval.splits import split_scenarios  # noqa: E402

SKILL = "verification-before-completion"
BASELINE = (
    "# Skill\n\nBe careful and write good code. "
    "Take your time to read the relevant files, make changes thoughtfully, "
    "and keep the codebase clean.\n"
)


def _cfg(tmp_path: Path, **overrides) -> ExperimentConfig:
    raw = {
        "name": "t",
        "skill": SKILL,
        "runner": "mock",
        "seed": 0,
        "train": 1,
        "validation": 2,
        "test": 2,
        "seeds_per_task": 1,
        "optimize_steps": 2,
        "baseline_skill": str(tmp_path / "baseline_SKILL.md"),
        "output_dir": str(tmp_path / "out"),
    }
    raw.update(overrides)
    return ExperimentConfig._from_dict(raw)


def _baseline_file(tmp_path: Path) -> Path:
    p = tmp_path / "baseline_SKILL.md"
    p.write_text(BASELINE, encoding="utf-8")
    return p


# ---- splits ---------------------------------------------------------------

def test_split_is_deterministic_and_disjoint():
    ids = ["a", "b", "c", "d", "e"]
    s1 = split_scenarios(ids, 1, 2, 2, seed=7)
    s2 = split_scenarios(ids, 1, 2, 2, seed=7)
    assert s1 == s2
    assert len(s1.train) == 1 and len(s1.validation) == 2 and len(s1.test) == 2
    assert not (set(s1.train) & set(s1.validation))
    assert not (set(s1.train) | set(s1.validation)) & set(s1.test)
    assert sorted(s1.train + s1.validation + s1.test) == ids


def test_split_rejects_bad_counts():
    with pytest.raises(ValueError, match="must equal"):
        split_scenarios(["a", "b"], 1, 1, 1, seed=0)


# ---- config ----------------------------------------------------------------

def test_config_requires_keys(tmp_path):
    with pytest.raises(ConfigError, match="missing required"):
        ExperimentConfig._from_dict({"name": "x"})


def test_config_rejects_unknown_runner(tmp_path):
    _baseline_file(tmp_path)
    with pytest.raises(ConfigError, match="runner must be one of"):
        _cfg(tmp_path, runner="bogus")


def test_mock_config_requires_baseline(tmp_path):
    with pytest.raises(ConfigError, match="baseline_skill"):
        _cfg(tmp_path, baseline_skill="")


# ---- optimization gate ------------------------------------------------------

def test_optimizer_accepts_then_rejects(tmp_path):
    base = _baseline_file(tmp_path)
    cfg = _cfg(tmp_path, optimize_steps=3)
    results = run_experiment(cfg, MockRunner(SKILL), BASELINE)
    attempts = results["optimization"]["attempts"]
    assert results["optimization"]["optimized"] is True
    assert attempts[0]["decision"] == "accepted"
    assert attempts[0]["validation_score"] > 0
    # later directives add no new covered behaviors on the val split
    assert all(a["decision"] == "rejected" for a in attempts[1:])
    # the source baseline file was never touched
    assert base.read_text(encoding="utf-8") == BASELINE


def test_validation_fails_closed_on_runner_error(tmp_path):
    """A candidate whose validation run errors can never be accepted."""
    _baseline_file(tmp_path)

    class ErrorRunner(MockRunner):
        def run(self, scenario_id, skill_text, seed, workdir):
            rec = super().run(scenario_id, skill_text, seed, workdir)
            if skill_text and "verify" in skill_text.lower():
                rec.passed = None
                rec.error = "TIMEOUT"
                rec.checks = []
            return rec

    cfg = _cfg(tmp_path, optimize_steps=1)
    results = run_experiment(cfg, ErrorRunner(SKILL), BASELINE)
    att = results["optimization"]["attempts"][0]
    assert att["decision"] == "rejected"
    assert "fail closed" in att["reason"]
    assert results["optimization"]["optimized"] is False


def test_learning_rate_bounds_edits(tmp_path):
    _baseline_file(tmp_path)
    # lr so small no directive can be appended -> no candidates proposed
    cfg = _cfg(tmp_path, optimize_steps=3, learning_rate=0.01)
    results = run_experiment(cfg, MockRunner(SKILL), BASELINE)
    assert results["optimization"]["attempts"] == []
    assert results["optimization"]["optimized"] is False


def test_rule_optimizer_determinism():
    opt = RuleOptimizer()
    c1 = opt.propose(BASELINE, 0, [])
    c2 = opt.propose(BASELINE, 0, [])
    assert c1 is not None and c1.text == c2.text
    assert c1.content_hash == c2.content_hash


# ---- file optimizer ---------------------------------------------------------

def test_file_optimizer_provenance_and_rejections(tmp_path):
    cand = tmp_path / "cand_SKILL.md"
    cand.write_text(BASELINE + "\n\nAlways verify your work.\n", encoding="utf-8")
    opt = FileOptimizer([str(cand)])
    edit = opt.propose(BASELINE, 0, [])
    assert edit is not None and edit.source == str(cand)
    assert opt.propose(BASELINE, 1, []) is None  # library exhausted
    with pytest.raises(FileNotFoundError):
        FileOptimizer([str(tmp_path / "missing.md")])


# ---- end-to-end artifacts ---------------------------------------------------

def test_end_to_end_artifacts(tmp_path):
    base = _baseline_file(tmp_path)
    cfg = _cfg(tmp_path, optimize_steps=1)
    results = run_experiment(cfg, MockRunner(SKILL), BASELINE)
    out = Path(cfg.output_dir)
    assert (out / "raw_results.csv").is_file()
    assert (out / "results.json").is_file()
    assert (out / "RESULTS.md").is_file()

    rows = list(csv.DictReader(open(out / "raw_results.csv", encoding="utf-8")))
    assert rows
    for key in ("tokens", "tool_calls", "turns", "latency_ms", "failure",
                    "trajectory_file", "skill_hash"):
        assert key in rows[0]
    traj = out / rows[0]["trajectory_file"]
    assert traj.is_file()

    # provenance is complete enough to reproduce
    prov = results["provenance"]
    assert prov["split"]["test"]
    assert prov["model"] and prov["seed"] == 0

    assert base.read_text(encoding="utf-8") == BASELINE


def test_aa_no_claim(tmp_path):
    """Identical conditions must produce delta 0 and no improvement claim."""
    _baseline_file(tmp_path)
    cfg = _cfg(tmp_path, optimize_steps=0, validation=0, test=5, train=0)
    results = run_experiment(cfg, MockRunner(SKILL), BASELINE)
    comp = results["test_split"]["comparison"]
    assert comp["delta"] == 0.0
    assert results["verdict"]["quality_improved"] is False
    assert "no candidate beats" in results["verdict"]["summary"]


def test_mock_runner_deterministic(tmp_path):
    r = MockRunner(SKILL)
    a = r.run("test-passes-verify", BASELINE + "\nverify\n", 0, tmp_path / "a")
    b = r.run("test-passes-verify", BASELINE + "\nverify\n", 0, tmp_path / "b")
    assert a.passed == b.passed and a.tokens == b.tokens
    assert a.checks == b.checks

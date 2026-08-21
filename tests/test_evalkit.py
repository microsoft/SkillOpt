"""Paired A/B evalkit: known-answer stats, A/A calibration, RESULTS replay."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

from skillopt_sleep.evalkit import (
    EvalkitError,
    bootstrap_delta_ci,
    compare,
    compare_aa,
    exact_mcnemar_p,
    format_markdown,
    main as evalkit_main,
    mcnemar_from_counts,
    mcnemar_paired,
    reconstruct_paired_from_rates,
)


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "evalkit")


def _load(name: str):
    with open(os.path.join(FIXTURE_DIR, name), encoding="utf-8") as f:
        return json.load(f)


class TestMcNemarKnownAnswer(unittest.TestCase):
    def test_textbook_2x2_chi2_and_exact(self):
        fx = _load("mcnemar_textbook.json")
        res = mcnemar_from_counts(
            fx["both_success"], fx["a_only"], fx["b_only"], fx["both_fail"],
        )
        self.assertAlmostEqual(res.chi2, fx["chi2"], places=12)
        self.assertAlmostEqual(res.p_chi2, fx["p_chi2"], places=12)
        self.assertAlmostEqual(res.p_exact, fx["p_exact"], places=12)
        self.assertTrue(res.significant)
        self.assertEqual(res.n, 100)

    def test_zero_discordants_is_not_significant(self):
        res = mcnemar_from_counts(20, 0, 0, 5)
        self.assertEqual(res.chi2, 0.0)
        self.assertEqual(res.p_chi2, 1.0)
        self.assertEqual(res.p_exact, 1.0)
        self.assertFalse(res.significant)

    def test_paired_vectors_match_counts(self):
        a = [1, 1, 1, 0, 0]
        b = [1, 0, 1, 1, 0]
        res = mcnemar_paired(a, b)
        self.assertEqual(res.both_success, 2)
        self.assertEqual(res.a_only, 1)
        self.assertEqual(res.b_only, 1)
        self.assertEqual(res.both_fail, 1)
        self.assertAlmostEqual(res.p_exact, exact_mcnemar_p(1, 1))


class TestBootstrapCoverage(unittest.TestCase):
    def test_identical_series_ci_collapses_to_zero(self):
        a = [1, 0, 1, 0, 1, 0, 1, 0]
        ci = bootstrap_delta_ci(a, a, n_boot=2000, seed=7)
        self.assertEqual(ci.low, 0.0)
        self.assertEqual(ci.high, 0.0)
        self.assertEqual(ci.mean, 0.0)

    def test_known_shift_ci_excludes_zero(self):
        # A always 0, B always 1: delta = 1 exactly, CI is [1, 1].
        a = [0] * 30
        b = [1] * 30
        ci = bootstrap_delta_ci(a, b, n_boot=1000, seed=1)
        self.assertEqual(ci.low, 1.0)
        self.assertEqual(ci.high, 1.0)

    def test_seed_is_deterministic(self):
        a = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        b = [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        x = bootstrap_delta_ci(a, b, n_boot=500, seed=99)
        y = bootstrap_delta_ci(a, b, n_boot=500, seed=99)
        self.assertEqual((x.low, x.high, x.mean), (y.low, y.high, y.mean))


class TestAACalibration(unittest.TestCase):
    def test_aa_does_not_reject(self):
        man = _load("aa_manifest.json")
        out = _load("aa_outcomes.json")
        report = compare_aa(man["ids"], out["outcomes"], n_boot=2000, seed=42)
        self.assertEqual(report.delta, 0.0)
        self.assertIsNotNone(report.mcnemar)
        self.assertFalse(report.mcnemar.significant)
        self.assertEqual(report.mcnemar.p_exact, 1.0)
        self.assertLessEqual(report.bootstrap.low, 0.0)
        self.assertGreaterEqual(report.bootstrap.high, 0.0)


class TestCompareContracts(unittest.TestCase):
    def test_mismatched_ids_are_refused(self):
        with self.assertRaises(EvalkitError) as ctx:
            compare(["t1", "t2"], {"t1": 1, "t2": 0}, {"t1": 1, "t3": 0})
        self.assertIn("must equal the manifest", str(ctx.exception))

    def test_duplicate_manifest_ids_refused(self):
        with self.assertRaises(EvalkitError):
            compare(["t1", "t1"], {"t1": 1}, {"t1": 0})

    def test_empty_manifest_refused(self):
        with self.assertRaises(EvalkitError):
            compare([], {}, {})

    def test_graded_refused_without_flag(self):
        with self.assertRaises(EvalkitError) as ctx:
            compare(["t1", "t2"], {"t1": 0.4, "t2": 0.9}, {"t1": 0.5, "t2": 0.8})
        self.assertIn("allow-graded", str(ctx.exception))

    def test_graded_bootstrap_only(self):
        report = compare(
            ["t1", "t2"],
            {"t1": 0.4, "t2": 0.9},
            {"t1": 0.5, "t2": 0.8},
            allow_graded=True,
            n_boot=500,
            seed=3,
        )
        self.assertIsNone(report.mcnemar)
        self.assertTrue(any("graded" in n for n in report.notes))
        self.assertAlmostEqual(report.delta, 0.0, places=12)

    def test_multi_seed_variance_band(self):
        report = compare(
            ["t1", "t2"],
            {"t1": [1, 0, 1], "t2": [0, 0, 1]},
            {"t1": [1, 1, 1], "t2": [1, 0, 1]},
            n_boot=400,
            seed=2,
        )
        self.assertEqual(len(report.per_seed), 3)
        self.assertIsNotNone(report.seed_mean_delta)
        self.assertGreaterEqual(report.seed_sd_delta, 0.0)
        self.assertAlmostEqual(report.rate_a, (2 / 3 + 1 / 3) / 2)
        self.assertAlmostEqual(report.rate_b, (1.0 + 2 / 3) / 2)


class TestResultsCellReplay(unittest.TestCase):
    def test_published_searchqa_nano_gated_delta(self):
        cell = _load("results_searchqa_nano_gated.json")
        a, b = reconstruct_paired_from_rates(cell["n"], cell["baseline"], cell["after"])
        self.assertEqual(len(a), cell["n"])
        self.assertAlmostEqual(sum(a) / cell["n"], cell["baseline"], places=3)
        self.assertAlmostEqual(sum(b) / cell["n"], cell["after"], places=3)
        ids = [f"q{i:04d}" for i in range(cell["n"])]
        report = compare(
            ids,
            dict(zip(ids, a)),
            dict(zip(ids, b)),
            n_boot=800,
            seed=42,
        )
        self.assertAlmostEqual(report.delta, cell["published_delta"], places=3)
        self.assertGreater(report.bootstrap.low, 0.0)
        self.assertTrue(report.mcnemar.significant)
        md = format_markdown(report)
        self.assertIn("delta (B-A)", md)
        self.assertIn("McNemar", md)


class TestCLI(unittest.TestCase):
    def test_aa_cli_exit_zero(self):
        rc = evalkit_main([
            "--manifest", os.path.join(FIXTURE_DIR, "aa_manifest.json"),
            "--a", os.path.join(FIXTURE_DIR, "aa_outcomes.json"),
            "--aa",
            "--boot", "300",
            "--json",
        ])
        self.assertEqual(rc, 0)

    def test_mismatch_cli_exit_two(self):
        with tempfile.TemporaryDirectory() as td:
            man = os.path.join(td, "m.json")
            a = os.path.join(td, "a.json")
            b = os.path.join(td, "b.json")
            with open(man, "w", encoding="utf-8") as f:
                json.dump(["t1", "t2"], f)
            with open(a, "w", encoding="utf-8") as f:
                json.dump({"t1": 1, "t2": 0}, f)
            with open(b, "w", encoding="utf-8") as f:
                json.dump({"t1": 1, "t3": 0}, f)
            rc = evalkit_main(["--manifest", man, "--a", a, "--b", b])
            self.assertEqual(rc, 2)

    def test_module_entrypoint(self):
        proc = subprocess.run(
            [
                sys.executable, "-m", "skillopt_sleep.evalkit",
                "--manifest", os.path.join(FIXTURE_DIR, "aa_manifest.json"),
                "--a", os.path.join(FIXTURE_DIR, "aa_outcomes.json"),
                "--aa",
                "--boot", "200",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("delta (B-A): +0.000000", proc.stdout)


if __name__ == "__main__":
    unittest.main()

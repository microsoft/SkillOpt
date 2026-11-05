"""POSIX-only end-to-end regression: the real bash pytest shim's emitted
``snap``/``result`` lines must be directly consumable by the order judge.

The offline scenario tests feed hand-authored logs to ``_pytest_reproduce_fix_order``,
so they never drive the actual bash producer (the shim). This test installs the shims,
drives ``pytest`` through them (reproduce -> edit -> verify), and asserts the judge
reaches True on the REAL emitted log. It fails if the shim's ``snap`` hash ever
diverges from the harness ``start``/``end`` snapshot (that divergence is exactly the
producer<->judge contract this guards). Skips on non-POSIX hosts (the shim is bash).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from skillopt_sleep.adapters.superpowers import (
    _pytest_reproduce_fix_order,
    _source_fingerprint,
    _write_pytest_shims,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="test executes POSIX pytest shims")


def _run_pytest(project: Path, bin_dir: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(
        ["pytest", "-q", str(project / "test_math.py")],
        cwd=project, env=env, capture_output=True, text=True,
    )


@posix_only
def test_real_shim_produces_order_the_judge_accepts(tmp_path: Path) -> None:
    project = tmp_path / "project"; project.mkdir()
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    audit_log = tmp_path / ".skillopt" / "pytest.log"; audit_log.parent.mkdir()
    nonce = "abc123"
    # Fingerprint is scoped to the scenario's source-under-test (setup minus protected).
    names = ["math_ops.py"]

    (project / "test_math.py").write_text(
        "from math_ops import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    # BUG: should be +. (test_math.py is the protected file, never hashed.)
    (project / "math_ops.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

    _write_pytest_shims(bin_dir, audit_log, nonce, project, names)

    start = _source_fingerprint(project, names)
    with open(audit_log, "a", encoding="utf-8") as fh:
        fh.write(f"{nonce} start {start}\n")

    # 1. Reproduce: failing run on the ORIGINAL source (shim snap must equal start).
    r1 = _run_pytest(project, bin_dir)
    assert r1.returncode == 1, f"expected reproduce to fail, got {r1.returncode}: {r1.stderr}"

    # 2. Fix the source (the only file the fingerprint hashes).
    (project / "math_ops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    # 3. Verify: passing run on the edited source.
    r2 = _run_pytest(project, bin_dir)
    assert r2.returncode == 0, f"expected verify to pass, got {r2.returncode}: {r2.stderr}"

    end = _source_fingerprint(project, names)
    with open(audit_log, "a", encoding="utf-8") as fh:
        fh.write(f"{nonce} end {end}\n")

    assert _pytest_reproduce_fix_order(audit_log, nonce) is True, (
        "real shim snap/result sequence did not satisfy reproduce-before-fix "
        "(producer<->judge contract broken)"
    )


@posix_only
def test_real_shim_rejects_edit_before_reproduction(tmp_path: Path) -> None:
    """The shim-driven judge must still fail-closed on edit-before-reproduce."""
    project = tmp_path / "project"; project.mkdir()
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    audit_log = tmp_path / ".skillopt" / "pytest.log"; audit_log.parent.mkdir()
    nonce = "abc123"
    names = ["math_ops.py"]

    (project / "test_math.py").write_text(
        "from math_ops import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    (project / "math_ops.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    _write_pytest_shims(bin_dir, audit_log, nonce, project, names)

    start = _source_fingerprint(project, names)
    with open(audit_log, "a", encoding="utf-8") as fh:
        fh.write(f"{nonce} start {start}\n")

    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

    # Edit BEFORE reproducing: the first run is already on a changed source.
    (project / "math_ops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    subprocess.run(["pytest", "-q", str(project / "test_math.py")], cwd=project, env=env,
                   capture_output=True, text=True)  # passes now

    end = _source_fingerprint(project, names)
    with open(audit_log, "a", encoding="utf-8") as fh:
        fh.write(f"{nonce} end {end}\n")

    # The judge sees only a passing run on the edited source (no fail on `start`).
    assert _pytest_reproduce_fix_order(audit_log, nonce) is False

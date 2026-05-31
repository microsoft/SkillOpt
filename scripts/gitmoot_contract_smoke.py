"""Run the local Gitmoot-SkillOpt contract smoke against a Gitmoot binary."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "examples" / "gitmoot" / "mvp-fixture"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-network Gitmoot-SkillOpt smoke: optimize the fixture, "
            "install a local Gitmoot template into a temp home, import the "
            "candidate artifacts, show the pending candidate, and reject it."
        )
    )
    parser.add_argument("--gitmoot-bin", default="gitmoot", help="Gitmoot executable to run")
    parser.add_argument("--home", default="", help="Gitmoot home; defaults to a temp directory")
    parser.add_argument("--out-root", default="", help="optimizer output root; defaults to a temp directory")
    parser.add_argument("--keep-temp", action="store_true", help="do not delete temp home/output directories")
    parser.add_argument("--training-package", default=str(FIXTURE_ROOT / "training.json"))
    parser.add_argument("--artifact-root", default=str(FIXTURE_ROOT / "blobs"))
    parser.add_argument("--template-file", default=str(FIXTURE_ROOT / "template.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    temp_dirs: list[Path] = []
    try:
        home = Path(args.home).expanduser() if args.home else _make_temp_dir("gitmoot-home-", temp_dirs)
        out_root = Path(args.out_root).expanduser() if args.out_root else _make_temp_dir("gitmoot-skillopt-out-", temp_dirs)
        candidate = out_root / "candidate.json"
        template_id = _template_id_from_package(Path(args.training_package))

        _run(
            [
                sys.executable,
                "-m",
                "gitmoot_skillopt",
                "optimize",
                "--training-package",
                args.training_package,
                "--artifact-root",
                args.artifact_root,
                "--out-root",
                str(out_root),
                "--candidate-output",
                str(candidate),
                "--dry-run",
            ],
            cwd=REPO_ROOT,
        )
        _run(
            [
                args.gitmoot_bin,
                "agent",
                "template",
                "add",
                "--home",
                str(home),
                "--file",
                args.template_file,
                template_id,
            ]
        )
        import_result = _run(
            [
                args.gitmoot_bin,
                "skillopt",
                "import",
                "--home",
                str(home),
                "--file",
                str(candidate),
                "--artifact-dir",
                str(out_root / "artifacts"),
            ]
        )
        version_id = _version_id_from_import(import_result.stdout)
        shown = _run([args.gitmoot_bin, "skillopt", "candidate", "show", "--home", str(home), version_id])
        _assert_contains(shown.stdout, "state: pending")
        _assert_contains(shown.stdout, "diff_artifact: candidate-diff")
        _run(
            [
                args.gitmoot_bin,
                "skillopt",
                "candidate",
                "reject",
                "--home",
                str(home),
                "--reason",
                "contract smoke",
                version_id,
            ]
        )
        rejected = _run([args.gitmoot_bin, "skillopt", "candidate", "show", "--home", str(home), version_id])
        _assert_contains(rejected.stdout, "state: rejected")
        print(f"contract smoke passed: {version_id}")
        return 0
    finally:
        if not args.keep_temp:
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir, ignore_errors=True)


def _make_temp_dir(prefix: str, temp_dirs: list[Path]) -> Path:
    path = Path(tempfile.mkdtemp(prefix=prefix))
    temp_dirs.append(path)
    return path


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not python_path else f"{REPO_ROOT}{os.pathsep}{python_path}"
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        quoted = " ".join(command)
        raise RuntimeError(
            f"command failed ({result.returncode}): {quoted}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _template_id_from_package(package_path: Path) -> str:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from gitmoot_skillopt.contracts import TrainingPackage

    return TrainingPackage.load(package_path).template.id


def _version_id_from_import(stdout: str) -> str:
    prefix = "imported pending candidate "
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    raise RuntimeError(f"could not parse imported version id from output:\n{stdout}")


def _assert_contains(text: str, expected: str) -> None:
    if expected not in text:
        raise RuntimeError(f"expected {expected!r} in output:\n{text}")


if __name__ == "__main__":
    raise SystemExit(main())

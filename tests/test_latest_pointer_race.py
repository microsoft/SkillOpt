"""The derived latest pointer retries replacement races, never unsafe files."""
import os
import stat
import sys
from unittest import mock

import pytest

from skillopt_sleep import staging


@pytest.mark.parametrize("scenario", [
    "two_zero_then_valid", "twenty_zero_then_valid", "persistent_zero",
    "hardlink", "multiple_hardlinks", "symlink", "broken_symlink",
    "directory", "fifo", "junction", "zero_link_fifo",
    "zero_then_hardlink", "zero_then_multiple_hardlinks",
    "zero_then_symlink", "zero_then_broken_symlink", "zero_then_directory",
    "zero_then_fifo", "zero_then_junction", "zero_then_missing",
])
def test_latest_pointer_revalidates_only_transient_zero_links(tmp_path, scenario):
    root = tmp_path / "staging"
    night = root / "20260815-010203"
    night.mkdir(parents=True)
    (night / "manifest.json").write_text("{}", encoding="utf-8")
    pointer = root / ".latest"
    original = b"20260814-010203\n"
    pointer.write_bytes(original)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside sentinel\n")
    aliases = [tmp_path / "hardlink", tmp_path / "second-hardlink"]
    junction = False

    def make_unsafe(kind):
        nonlocal junction
        if kind == "directory":
            pointer.unlink()
            pointer.mkdir()
        elif kind in {"hardlink", "multiple_hardlinks"}:
            try:
                for alias in aliases[:2 if kind == "multiple_hardlinks" else 1]:
                    os.link(pointer, alias)
            except OSError:
                pytest.skip("hard links unavailable")
        elif kind in {"symlink", "broken_symlink"}:
            pointer.unlink()
            try:
                pointer.symlink_to(outside if kind == "symlink" else tmp_path / "missing")
            except OSError:
                pytest.skip("symlinks unavailable")
        elif kind in {"fifo", "zero_link_fifo"}:
            if not hasattr(os, "mkfifo"):
                pytest.skip("FIFOs unavailable")
            pointer.unlink()
            os.mkfifo(pointer)
        elif kind == "junction":
            # Exercise the real link/junction guard via its platform hook;
            # leave the inode regular so S_ISREG cannot mask a missing check.
            junction = True
        elif kind == "missing":
            pointer.unlink()

    make_unsafe(scenario)
    real_lstat = os.lstat
    real_isjunction = getattr(os.path, "isjunction", lambda _path: False)
    publisher_code = staging._publish_latest.__code__
    snapshots = 0
    sleeps = []

    def replacement_snapshot(path, *args, **kwargs):
        nonlocal snapshots
        result = real_lstat(path, *args, **kwargs)
        # Only the publisher's direct lstat receives the observed race result.
        # lexists/islink/type checks and filesystem transitions remain real.
        if os.fspath(path) == str(pointer) and sys._getframe(1).f_code is publisher_code:
            snapshots += 1
            limit = {"two_zero_then_valid": 2, "twenty_zero_then_valid": 20}.get(scenario, 0)
            if (scenario in {"persistent_zero", "zero_link_fifo"}
                    or (scenario.startswith("zero_then_") and snapshots == 1)
                    or snapshots <= limit):
                fields = list(result)
                fields[3] = 0  # st_nlink; preserve the actual inode's type.
                return os.stat_result(fields)
        return result

    def isjunction(path):
        return (junction and os.fspath(path) == str(pointer)) or real_isjunction(path)

    def backoff(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1 and scenario.startswith("zero_then_"):
            make_unsafe(scenario.removeprefix("zero_then_"))

    with mock.patch.object(staging.os, "lstat", replacement_snapshot), mock.patch.object(
        staging.os.path, "isjunction", side_effect=isjunction, create=True
    ), mock.patch.object(staging.time, "sleep", side_effect=backoff), mock.patch.object(
        staging, "_write_atomic_bytes", wraps=staging._write_atomic_bytes
    ) as write:
        if scenario in {"two_zero_then_valid", "twenty_zero_then_valid"}:
            count = 2 if scenario == "two_zero_then_valid" else 20
            staging._publish_latest(str(root), str(night))
            assert pointer.read_bytes() == b"20260815-010203\n"
            assert snapshots == count + 1
            assert sleeps == pytest.approx([0.005 * (i + 1) for i in range(count)])
            write.assert_called_once_with(
                str(pointer), b"20260815-010203\n", mode=0o600,
                replace_permission_retries=20,
            )
        else:
            error = FileNotFoundError if scenario == "zero_then_missing" else staging.StagingError
            with pytest.raises(error):
                staging._publish_latest(str(root), str(night))
            write.assert_not_called()
            if scenario == "persistent_zero":
                assert snapshots == 21
                assert len(sleeps) == 20
                assert sum(sleeps) == pytest.approx(1.05)
                assert pointer.read_bytes() == original
            elif scenario.startswith("zero_then_"):
                assert sleeps == [0.005]
            else:
                assert sleeps == []

    for alias in aliases:
        if alias.exists():
            assert alias.read_bytes() == original
    if scenario.endswith("directory"):
        assert pointer.is_dir()
    elif scenario.endswith("fifo"):
        assert stat.S_ISFIFO(real_lstat(pointer).st_mode)
    elif scenario.endswith("symlink"):
        assert pointer.is_symlink()
    elif scenario.endswith("missing"):
        assert not os.path.lexists(pointer)
    elif scenario.endswith("junction"):
        assert pointer.read_bytes() == original
    assert outside.read_bytes() == b"outside sentinel\n"
    assert not list(root.glob(".tmp-*"))

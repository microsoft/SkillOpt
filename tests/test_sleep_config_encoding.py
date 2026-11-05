"""Sleep configuration stays readable independently of the process locale."""

import builtins
import json
import locale
import os
import subprocess
import sys
from pathlib import Path

import pytest

from skillopt_sleep import config


@pytest.mark.parametrize("suffix", ["json", "yaml", "yml"])
def test_utf8_config_survives_a_non_utf8_locale(tmp_path: Path, suffix: str) -> None:
    values = {
        "backend": "codex",
        "max_tokens_per_night": 1000,
        "max_tasks_per_night": 2,
        "preferences": "Use résumé examples and 中文说明.",
        "target_skill_path": "技能/SKILL.md",
    }
    path = tmp_path / f"config.{suffix}"
    if suffix == "json":
        text = json.dumps(values, ensure_ascii=False)
    else:
        yaml = pytest.importorskip("yaml")
        text = yaml.safe_dump(values, allow_unicode=True)
    path.write_text(text, encoding="utf-8")

    script = """
import json
import sys
from skillopt_sleep import config

config.HOME_STATE_DIR = sys.argv[1]
expected = json.loads(sys.argv[2])
loaded = config.load_config()
for key, value in expected.items():
    assert loaded.get(key) == value, (key, loaded.get(key), value)
assert set(expected) <= set(loaded.get("_user_config_keys"))
overridden = config.load_config(backend="claude", preferences=None)
assert overridden.backend == "claude"
assert overridden.preferences == expected["preferences"]
"""
    env = {
        **os.environ,
        "LC_ALL": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
    }
    root = str(Path(config.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), json.dumps(values)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(params=[(codec, suffix) for codec in ("gbk", "cp1252") for suffix in ("json", "yaml", "yml")])
def legacy_config(request, tmp_path, monkeypatch):
    """Simulate the old locale codec on Linux; this is not a native Windows test."""
    codec, suffix = request.param
    text = "中文规则" if codec == "gbk" else "résumé rules"
    values = {
        "backend": "mock",
        "max_tokens_per_night": 1000,
        "max_tasks_per_night": 2,
        "preferences": text,
        "target_skill_path": f"skills/{text}/SKILL.md",
    }
    path = tmp_path / f"config.{suffix}"
    if suffix == "json":
        content = json.dumps(values, ensure_ascii=False)
    else:
        yaml = pytest.importorskip("yaml")
        content = yaml.safe_dump(values, allow_unicode=True)
    path.write_bytes(content.encode(codec))
    monkeypatch.setattr(config, "HOME_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(locale, "getencoding", lambda: codec, raising=False)
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale=True: codec)

    # Also reproduce default open() for an unchanged, pre-UTF-8 loader.
    def locale_open(file, mode="r", **kwargs):
        if kwargs.get("encoding") is None and "b" not in mode:
            kwargs["encoding"] = codec
        return builtins.open(file, mode, **kwargs)

    monkeypatch.setattr(config, "open", locale_open, raising=False)
    return path, values, codec


def test_legacy_config_preserves_explicit_limits_and_targets(legacy_config, recwarn):
    path, values, codec = legacy_config
    original = path.read_bytes()
    loaded = config.load_config()
    for key, value in values.items():
        assert loaded.get(key) == value
    assert set(values) <= set(loaded.get("_user_config_keys"))
    assert path.read_bytes() == original
    assert any(codec in str(w.message) and "UTF-8" in str(w.message) for w in recwarn)


def test_cli_overrides_legacy_config_without_losing_other_limits(legacy_config, monkeypatch, recwarn):
    from skillopt_sleep import __main__ as cli

    path, values, _codec = legacy_config
    original = path.read_bytes()
    target = path.parent / "override" / "SKILL.md"
    captured = []

    # Use the real argument parser and config assembly, without starting a cycle.
    def capture_run(args, dry=False):
        captured.append(cli._cfg_from_args(args))
        return 0

    monkeypatch.setattr(cli, "cmd_run", capture_run)
    assert cli.main([
        "dry-run", "--max-tasks", "1", "--preferences", "CLI rules",
        "--target-skill-path", str(target), "--backend", "mock",
    ]) == 0
    loaded = captured[0]
    assert loaded.max_tasks_per_night == 1
    assert loaded.max_tokens_per_night == values["max_tokens_per_night"]
    assert loaded.preferences == "CLI rules"
    assert loaded.target_skill_path == str(target)
    assert set(values) <= set(loaded.get("_user_config_keys"))
    assert path.read_bytes() == original


@pytest.mark.parametrize("suffix", ["json", "yaml", "yml"])
@pytest.mark.parametrize("failure", ["encoding", "syntax", "shape"])
def test_unreadable_config_does_not_fall_back_to_defaults(tmp_path, monkeypatch, suffix, failure):
    path = tmp_path / f"config.{suffix}"
    if failure == "encoding":
        content = b"\xff\xfeinvalid"
    elif failure == "syntax":
        content = b'{"max_tokens_per_night":' if suffix == "json" else b"max_tokens_per_night: ["
    else:
        content = b"[]" if suffix == "json" else b"- not a mapping\n"
    path.write_bytes(content)
    monkeypatch.setattr(config, "HOME_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(locale, "getencoding", lambda: "ascii", raising=False)
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale=True: "ascii")
    with pytest.raises(ValueError, match="Cannot load sleep configuration") as exc:
        config.load_config(max_tasks_per_night=1)
    assert str(path) in str(exc.value)
    assert "UTF-8" in str(exc.value)
    assert path.read_bytes() == content


@pytest.mark.parametrize("suffix", ["json", "yaml", "yml"])
def test_cli_stops_before_a_cycle_on_unreadable_config(tmp_path, monkeypatch, capsys, suffix):
    from skillopt_sleep import __main__ as cli

    path = tmp_path / f"config.{suffix}"
    content = b"\xff\xfeinvalid"
    path.write_bytes(content)
    monkeypatch.setattr(config, "HOME_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(locale, "getencoding", lambda: "ascii", raising=False)
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale=True: "ascii")

    def unexpected_cycle(*args, **kwargs):
        pytest.fail("a configuration error must stop before running a cycle")

    monkeypatch.setattr(cli, "run_sleep_cycle", unexpected_cycle)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--backend", "mock", "--max-tasks", "1"])
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert str(path) in stderr
    assert "UTF-8" in stderr
    assert "Traceback" not in stderr
    assert path.read_bytes() == content
    assert list(tmp_path.iterdir()) == [path]


def test_missing_yaml_dependency_is_not_silently_ignored(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("max_tokens_per_night: 1000\n", encoding="utf-8")
    monkeypatch.setattr(config, "HOME_STATE_DIR", str(tmp_path))
    real_import = builtins.__import__

    def without_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("PyYAML unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_yaml)
    with pytest.raises(ValueError, match="PyYAML"):
        config.load_config()


def test_absent_config_keeps_defaults_and_explicit_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOME_STATE_DIR", str(tmp_path))
    loaded = config.load_config(max_tasks_per_night=1)
    assert loaded.max_tasks_per_night == 1
    assert loaded.max_tokens_per_night == config.DEFAULTS["max_tokens_per_night"]


def test_unreadable_config_path_does_not_fall_back_to_defaults(tmp_path, monkeypatch):
    (tmp_path / "config.json").mkdir()
    monkeypatch.setattr(config, "HOME_STATE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Cannot load sleep configuration"):
        config.load_config()

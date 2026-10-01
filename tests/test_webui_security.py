"""Tests for the SkillOpt WebUI security posture (bind default + public warning).

The WebUI is gradio-coupled, so we inject a minimal fake ``gradio`` module and
mock ``build_ui``/``launch`` to exercise ``main()``'s argparse + host-check
logic without the heavy ``webui`` extra.
"""

from __future__ import annotations

import itertools
import sys
import types
import unittest.mock as mock

import pytest


@pytest.fixture
def webui(monkeypatch):
    fake_gradio = types.ModuleType("gradio")
    fake_gradio.themes = types.SimpleNamespace(Soft=lambda **kw: mock.MagicMock())
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)
    import skillopt_webui.app as app

    return app


def test_main_defaults_host_to_localhost(webui, monkeypatch):
    """The server must not be publicly bound by default."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py"])

    webui_mod.main()

    launcher.assert_called_once()
    _args, kwargs = launcher.call_args
    assert kwargs["server_name"] == "127.0.0.1"


def test_main_warns_on_public_host(webui, monkeypatch, capsys):
    """An explicit public bind must emit an unauthenticated-exposure warning."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--host", "0.0.0.0"])

    webui_mod.main()

    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    _args, kwargs = launcher.call_args
    assert kwargs["server_name"] == "0.0.0.0"


def test_main_warns_on_share(webui, monkeypatch, capsys):
    """--share must emit a public-tunnel warning."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--share"])

    webui_mod.main()

    captured = capsys.readouterr()
    assert "share" in captured.err.lower()
    assert "public" in captured.err.lower() or "tunnel" in captured.err.lower()


def test_main_auth_via_cli_args(webui, monkeypatch):
    """--auth-user and --auth-pass must enable Gradio basic auth."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--auth-user", "admin", "--auth-pass", "s3cret"])

    webui_mod.main()

    _args, kwargs = launcher.call_args
    assert kwargs.get("auth") == ("admin", "s3cret")


def test_main_auth_via_env(webui, monkeypatch):
    """SKILLOPT_WEBUI_USER / SKILLOPT_WEBUI_PASS must enable auth without CLI args."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py"])
    monkeypatch.setenv("SKILLOPT_WEBUI_USER", "envuser")
    monkeypatch.setenv("SKILLOPT_WEBUI_PASS", "envpass")

    webui_mod.main()

    _args, kwargs = launcher.call_args
    assert kwargs.get("auth") == ("envuser", "envpass")


def test_main_no_auth_by_default(webui, monkeypatch):
    """Without auth args or env vars, no auth must be configured."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py"])
    monkeypatch.delenv("SKILLOPT_WEBUI_USER", raising=False)
    monkeypatch.delenv("SKILLOPT_WEBUI_PASS", raising=False)

    webui_mod.main()

    _args, kwargs = launcher.call_args
    assert "auth" not in kwargs or kwargs["auth"] is None


def test_scan_outputs_rejects_path_traversal(webui, tmp_path, monkeypatch):
    """The scan_outputs callback must not enumerate directories outside PROJECT_ROOT."""
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    (tmp_path / "outputs").mkdir()
    # Every traversal / escape form is denied at consumption: no rows, no reads.
    for bad in ("/../../etc/passwd", "../outside", "outputs/../../../etc", "C:\\Windows"):
        assert webui.scan_outputs(bad) == [], f"traversal {bad!r} must be denied"


def test_scan_outputs_allows_valid_subdir(webui, tmp_path, monkeypatch):
    """scan_outputs must accept directories within PROJECT_ROOT."""
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    (tmp_path / "outputs" / "bench1" / "run1").mkdir(parents=True)
    (tmp_path / "outputs" / "bench1" / "run1" / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    rows = webui.scan_outputs("outputs")
    assert rows, "valid in-tree output area must be digested"


def test_scan_outputs_callback_consumes_within_project(webui, tmp_path, monkeypatch):
    """The registered scan_outputs callback must digest data only inside PROJECT_ROOT
    at the point data is actually read (traversal denied, in-tree consumed)."""
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    # A traversal arg must be denied at consumption: no rows, no data read.
    assert webui.scan_outputs("/../../etc/passwd") == []
    assert webui.scan_outputs("../outside") == []
    # A valid in-tree output area is digested (config.yaml read per run dir).
    (tmp_path / "outputs/bench1/run1").mkdir(parents=True)
    (tmp_path / "outputs/bench1/run1/config.yaml").write_text("alpha: 1\n", encoding="utf-8")
    rows = webui.scan_outputs("outputs")
    assert rows, f"expected rows from a valid in-tree output area, got {rows!r}"


def test_config_preview_rejects_relative_traversal(webui, tmp_path, monkeypatch):
    """The config-preview callback must not read YAML outside PROJECT_ROOT."""
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    outside_dir = tmp_path.parent / (tmp_path.name + "_outside")
    outside_dir.mkdir()
    secret = outside_dir / "secret.yaml"
    secret.write_text("password: dummy-secret-value\n", encoding="utf-8")

    result = webui.config_preview(f"../{outside_dir.name}/secret.yaml")

    assert "dummy-secret-value" not in result


def test_config_preview_rejects_absolute_outside_path(webui, tmp_path, monkeypatch):
    """An absolute path escaping PROJECT_ROOT must be denied at consumption."""
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    outside_dir = tmp_path.parent / (tmp_path.name + "_outside_abs")
    outside_dir.mkdir()
    secret = outside_dir / "secret.yaml"
    secret.write_text("password: dummy-secret-value\n", encoding="utf-8")

    result = webui.config_preview(str(secret))

    assert "dummy-secret-value" not in result


def test_config_preview_allows_configs_under_project(webui, tmp_path, monkeypatch):
    """An in-tree config under configs/ must still preview normally."""
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "demo.yaml").write_text("name: demo\n", encoding="utf-8")

    result = webui.config_preview("configs/demo.yaml")

    assert "name: demo" in result


def test_validate_training_config_rejects_outside_path(webui, tmp_path, monkeypatch):
    """Launch preflight must reject a config path that escapes PROJECT_ROOT."""
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    outside_dir = tmp_path.parent / (tmp_path.name + "_outside_train")
    outside_dir.mkdir()
    (outside_dir / "train.yaml").write_text("name: demo\n", encoding="utf-8")

    result = webui.validate_training_config(
        f"../{outside_dir.name}/train.yaml",
        {},
    )

    assert result is not None, "path escaping PROJECT_ROOT must fail closed"


def test_main_rejects_incomplete_cli_auth_user_only(webui, monkeypatch):
    """--auth-user without --auth-pass must fail closed (never launch)."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--host", "0.0.0.0", "--auth-user", "admin"])
    with pytest.raises(SystemExit):
        webui_mod.main()
    launcher.assert_not_called()


def test_main_rejects_incomplete_cli_auth_pass_only(webui, monkeypatch):
    """--auth-pass without --auth-user must fail closed (never launch)."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--host", "0.0.0.0", "--auth-pass", "s3cret"])
    with pytest.raises(SystemExit):
        webui_mod.main()
    launcher.assert_not_called()


def test_main_rejects_incomplete_env_auth_user_only(webui, monkeypatch):
    """Only SKILLOPT_WEBUI_USER set must fail closed (never launch)."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--host", "0.0.0.0"])
    monkeypatch.setenv("SKILLOPT_WEBUI_USER", "envuser")
    monkeypatch.delenv("SKILLOPT_WEBUI_PASS", raising=False)
    with pytest.raises(SystemExit):
        webui_mod.main()
    launcher.assert_not_called()


def test_main_rejects_incomplete_env_auth_pass_only(webui, monkeypatch):
    """Only SKILLOPT_WEBUI_PASS set must fail closed (never launch)."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--host", "0.0.0.0"])
    monkeypatch.setenv("SKILLOPT_WEBUI_PASS", "envpass")
    monkeypatch.delenv("SKILLOPT_WEBUI_USER", raising=False)
    with pytest.raises(SystemExit):
        webui_mod.main()
    launcher.assert_not_called()


@pytest.fixture
def auth_probe(webui, monkeypatch):
    """Run the real ``main()`` with a mocked UI and report what it launched.

    Returns ``(launcher, exit_code, builder)``, where ``exit_code`` is ``None``
    when ``main()`` returned normally and the ``SystemExit`` code otherwise, so
    a test can tell "refused to start" from "started without auth". ``builder``
    is the mocked ``build_ui``, so a rejection test can also assert the UI was
    never constructed at all rather than only that it was not launched.
    """
    def _run(argv=(), env=None):
        launcher = mock.MagicMock()
        app_mock = mock.MagicMock()
        app_mock.launch = launcher
        builder = mock.MagicMock(return_value=app_mock)
        monkeypatch.setattr(webui, "build_ui", builder)
        monkeypatch.setattr(sys, "argv", ["app.py", *argv])
        for name in ("SKILLOPT_WEBUI_USER", "SKILLOPT_WEBUI_PASS"):
            monkeypatch.delenv(name, raising=False)
        for name, value in (env or {}).items():
            monkeypatch.setenv(name, value)
        code = None
        try:
            webui.main()
        except SystemExit as exc:
            code = exc.code
        return launcher, code, builder

    return _run


def test_main_auth_matrix_launches_with_complete_configuration(auth_probe):
    """CLI-only, env-only and field-mixed sources must each enable auth."""
    for argv, env, expected in (
        (["--auth-user", "admin", "--auth-pass", "s3cret"], {}, ("admin", "s3cret")),
        (
            [],
            {"SKILLOPT_WEBUI_USER": "envuser", "SKILLOPT_WEBUI_PASS": "envpass"},
            ("envuser", "envpass"),
        ),
        (["--auth-user", "admin"], {"SKILLOPT_WEBUI_PASS": "envpass"}, ("admin", "envpass")),
    ):
        launcher, code, _builder = auth_probe(argv, env)

        assert code is None, f"complete configuration must launch: {argv} {env}"
        assert launcher.call_args.kwargs.get("auth") == expected


def test_main_cli_credentials_take_precedence_over_env(auth_probe):
    """Precedence is per field: an explicit flag wins over its variable."""
    launcher, code, _builder = auth_probe(
        ["--auth-user", "cli", "--auth-pass", "clipass"],
        {"SKILLOPT_WEBUI_USER": "envuser", "SKILLOPT_WEBUI_PASS": "envpass"},
    )

    assert code is None
    assert launcher.call_args.kwargs["auth"] == ("cli", "clipass")


def _expected_auth_outcome(cli_user, cli_pass, env_user, env_pass):
    """The decision table, restated independently of the implementation.

    ``None`` means the source was not supplied at all, which is different from
    an empty string: a supplied-but-blank credential is a misconfiguration.
    """
    has_user = cli_user is not None or env_user is not None
    has_pass = cli_pass is not None or env_pass is not None
    if not has_user and not has_pass:
        return "no-auth"
    if not has_user or not has_pass:
        return "refuse"
    user = cli_user if cli_user is not None else env_user
    password = cli_pass if cli_pass is not None else env_pass
    if not user.strip() or not password.strip():
        return "refuse"
    return (user, password)


@pytest.mark.parametrize(
    "cli_user, cli_pass, env_user, env_pass",
    list(itertools.product(
        (None, "", "u"),  # absent, supplied-but-blank, supplied
        (None, "", "p"),
        (None, "", "u"),
        (None, "", "p"),
    )),
)
def test_main_auth_decision_table(auth_probe, cli_user, cli_pass, env_user, env_pass):
    """Every absent/blank/supplied combination across both sources."""
    argv = []
    if cli_user is not None:
        argv += ["--auth-user", cli_user]
    if cli_pass is not None:
        argv += ["--auth-pass", cli_pass]
    env = {}
    if env_user is not None:
        env["SKILLOPT_WEBUI_USER"] = env_user
    if env_pass is not None:
        env["SKILLOPT_WEBUI_PASS"] = env_pass

    launcher, code, builder = auth_probe(argv, env)
    expected = _expected_auth_outcome(cli_user, cli_pass, env_user, env_pass)

    if expected == "refuse":
        assert code == 1, "invalid auth must stop before the UI"
        assert builder.call_count == 0
        launcher.assert_not_called()
    elif expected == "no-auth":
        assert code is None
        assert "auth" not in launcher.call_args.kwargs
    else:
        assert code is None
        assert launcher.call_args.kwargs["auth"] == expected

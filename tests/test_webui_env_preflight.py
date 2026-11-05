import pytest
import yaml

pytest.importorskip("gradio")

from skillopt_webui import app as webui_app


def _write_config(tmp_path, model):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "demo.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "model": model,
            "env": {"name": "searchqa"},
        }),
        encoding="utf-8",
    )
    return "configs/demo.yaml"


def test_build_training_env_loads_project_dotenv(tmp_path, monkeypatch):
    monkeypatch.setattr(webui_app, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join([
            "export QWEN_CHAT_BASE_URL=http://qwen.example/v1",
            "QWEN_CHAT_MODEL=test-model",
            "QWEN_CHAT_API_KEY='secret-value'",
        ]),
        encoding="utf-8",
    )

    env = webui_app.build_training_env()

    assert env["QWEN_CHAT_BASE_URL"] == "http://qwen.example/v1"
    assert env["QWEN_CHAT_MODEL"] == "test-model"
    assert env["QWEN_CHAT_API_KEY"] == "secret-value"


def test_preflight_reports_missing_openai_chat_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(webui_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("OPTIMIZER_AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("TARGET_AZURE_OPENAI_ENDPOINT", raising=False)
    config_path = _write_config(
        tmp_path,
        {
            "backend": "qwen",
            "optimizer_backend": "openai_chat",
            "target_backend": "openai_chat",
        },
    )

    error = webui_app.validate_training_config(config_path, {})

    assert "missing Azure/OpenAI-compatible endpoint for optimizer, target" in error
    assert "model.backend is qwen" in error


def test_preflight_reports_unreachable_qwen_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(webui_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(webui_app, "_can_connect_to_url", lambda _url: False)
    config_path = _write_config(
        tmp_path,
        {
            "backend": "qwen",
            "optimizer_backend": "qwen_chat",
            "target_backend": "qwen_chat",
            "qwen_chat_base_url": "http://127.0.0.1:9/v1",
        },
    )

    error = webui_app.validate_training_config(config_path, {})

    assert "cannot connect to qwen_chat endpoint" in error
    assert "127.0.0.1:9" in error


def test_preflight_accepts_reachable_qwen_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(webui_app, "PROJECT_ROOT", tmp_path)
    seen_urls = []
    monkeypatch.setattr(webui_app, "_can_connect_to_url", lambda url: seen_urls.append(url) or True)
    config_path = _write_config(
        tmp_path,
        {
            "optimizer_backend": "qwen_chat",
            "target_backend": "qwen_chat",
            "qwen_chat_base_url": "http://qwen.example/v1",
        },
    )

    assert webui_app.validate_training_config(config_path, {}) is None
    assert seen_urls == ["http://qwen.example/v1", "http://qwen.example/v1"]

from pathlib import Path

import pytest

from src.config import load_config, validate_audio_baseline_config


def test_load_config_reads_yaml_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment:\n  name: test\n", encoding="utf-8")

    config = load_config(config_path)

    assert config == {"experiment": {"name": "test"}}


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_empty_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_config(config_path)


def test_validate_audio_baseline_config_accepts_project_config() -> None:
    config = load_config("configs/baseline_audio.yaml")

    validate_audio_baseline_config(config)


def test_validate_audio_baseline_config_rejects_missing_section() -> None:
    config = load_config("configs/baseline_audio.yaml")
    config.pop("model")

    with pytest.raises(ValueError, match="required sections"):
        validate_audio_baseline_config(config)


def test_validate_audio_baseline_config_rejects_unknown_model() -> None:
    config = load_config("configs/baseline_audio.yaml")
    config["model"]["name"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported audio model"):
        validate_audio_baseline_config(config)


def test_validate_audio_baseline_config_rejects_empty_metrics() -> None:
    config = load_config("configs/baseline_audio.yaml")
    config["evaluation"]["metrics"] = []

    with pytest.raises(ValueError, match="evaluation.metrics"):
        validate_audio_baseline_config(config)

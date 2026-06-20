"""Project configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.data.video import validate_video_preprocessing_config
from src.models.baselines.video import validate_video_model_config


Config = dict[str, Any]

__all__ = [
    "Config",
    "load_config",
    "validate_audio_baseline_config",
    "validate_video_baseline_config",
]


REQUIRED_AUDIO_BASELINE_SECTIONS = {
    "experiment",
    "data",
    "audio",
    "features",
    "model",
    "training",
    "validation",
    "evaluation",
    "checkpointing",
    "logging",
}

REQUIRED_VIDEO_BASELINE_SECTIONS = {
    "experiment",
    "data",
    "video",
    "model",
    "training",
    "validation",
    "evaluation",
    "checkpointing",
    "logging",
}


def load_config(config_path: str | Path) -> Config:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the YAML file is empty or does not contain a mapping.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def _require_sections(config: Config, required_sections: set[str]) -> None:
    """Validate that a config contains required top-level sections.

    Args:
        config: Configuration dictionary to validate.
        required_sections: Required top-level section names.

    Raises:
        ValueError: If at least one required section is missing.
    """
    missing_sections = required_sections - set(config)

    if missing_sections:
        raise ValueError(
            f"Config is missing required sections: {sorted(missing_sections)}"
        )


def _require_keys(section: Config, section_name: str, required_keys: set[str]) -> None:
    """Validate that a config section contains required keys.

    Args:
        section: Configuration section to validate.
        section_name: Name of the section used in the error message.
        required_keys: Required key names inside the section.

    Raises:
        ValueError: If ``section`` is not a mapping or if keys are missing.
    """
    if not isinstance(section, dict):
        raise ValueError(f"Config section must be a mapping: {section_name}")

    missing_keys = required_keys - set(section)

    if missing_keys:
        raise ValueError(
            f"Config section '{section_name}' is missing keys: {sorted(missing_keys)}"
        )


def _require_positive_number(value: int | float, key_path: str) -> None:
    """Validate that a numeric config value is positive.

    Args:
        value: Numeric value to validate.
        key_path: Human-readable config key path used in the error message.

    Raises:
        ValueError: If ``value`` is not numeric or is not strictly positive.
    """
    if not isinstance(value, int | float):
        raise ValueError(f"Config value must be numeric: {key_path}")

    if value <= 0:
        raise ValueError(f"Config value must be greater than 0: {key_path}")


def _validate_loss_config(loss_config: Config, num_classes: int) -> None:
    """Validate the supported cross-entropy and class-weight settings."""
    if loss_config.get("name") != "cross_entropy":
        raise ValueError(f"Unsupported loss: {loss_config.get('name')}")

    class_weights = loss_config.get("class_weights")
    if class_weights in (None, "none", "balanced"):
        return

    if not isinstance(class_weights, list) or len(class_weights) != num_classes:
        raise ValueError(
            "training.loss.class_weights must be 'balanced', 'none', "
            "or contain one weight per class"
        )

    for weight in class_weights:
        _require_positive_number(weight, "training.loss.class_weights")


def validate_audio_baseline_config(config: Config) -> None:
    """Validate the baseline audio experiment configuration.

    Args:
        config: Configuration dictionary loaded from YAML.

    Raises:
        ValueError: If the configuration is missing required sections, missing
            required keys, or contains unsupported baseline values.
    """
    _require_sections(config, REQUIRED_AUDIO_BASELINE_SECTIONS)

    _require_keys(config["experiment"], "experiment", {"name", "seed", "output_dir"})
    _require_keys(
        config["data"],
        "data",
        {"train_manifest", "val_manifest", "test_manifest", "label_mapping"},
    )
    _require_keys(
        config["audio"],
        "audio",
        {"source_file", "sample_rate", "duration_seconds", "mono", "normalize"},
    )
    _require_keys(
        config["features"],
        "features",
        {"type", "n_mels", "n_fft", "hop_length", "power", "log_scale"},
    )
    _require_keys(
        config["model"],
        "model",
        {"name", "input_channels", "num_classes", "conv_channels", "dropout"},
    )
    _require_keys(
        config["training"],
        "training",
        {"device", "epochs", "batch_size", "optimizer", "loss"},
    )
    _require_keys(config["validation"], "validation", {"batch_size"})
    _require_keys(config["evaluation"], "evaluation", {"batch_size", "metrics"})
    _require_keys(config["checkpointing"], "checkpointing", {"save_dir"})
    _require_keys(config["logging"], "logging", {"log_dir", "level"})

    if config["features"]["type"] != "mel_spectrogram":
        raise ValueError(f"Unsupported feature type: {config['features']['type']}")

    if config["model"]["name"] != "audio_cnn_baseline":
        raise ValueError(f"Unsupported audio model: {config['model']['name']}")

    expected_label_mapping = {"real": 0, "fake": 1}
    if config["data"]["label_mapping"] != expected_label_mapping:
        raise ValueError(
            "data.label_mapping must be {'real': 0, 'fake': 1} "
            "for the baseline audio task"
        )

    _require_positive_number(config["audio"]["sample_rate"], "audio.sample_rate")
    _require_positive_number(
        config["audio"]["duration_seconds"],
        "audio.duration_seconds",
    )
    _require_positive_number(config["features"]["n_mels"], "features.n_mels")
    _require_positive_number(config["features"]["n_fft"], "features.n_fft")
    _require_positive_number(config["features"]["hop_length"], "features.hop_length")
    _require_positive_number(config["features"]["power"], "features.power")
    _require_positive_number(config["model"]["input_channels"], "model.input_channels")
    _require_positive_number(config["model"]["num_classes"], "model.num_classes")
    _require_positive_number(config["training"]["epochs"], "training.epochs")
    _require_positive_number(config["training"]["batch_size"], "training.batch_size")
    _require_positive_number(config["validation"]["batch_size"], "validation.batch_size")
    _require_positive_number(config["evaluation"]["batch_size"], "evaluation.batch_size")
    _validate_loss_config(config["training"]["loss"], config["model"]["num_classes"])

    if not config["model"]["conv_channels"]:
        raise ValueError("model.conv_channels must contain at least one value")

    if not config["evaluation"]["metrics"]:
        raise ValueError("evaluation.metrics must contain at least one metric")


def validate_video_baseline_config(config: Config) -> None:
    """Validate the baseline video experiment configuration."""
    _require_sections(config, REQUIRED_VIDEO_BASELINE_SECTIONS)

    _require_keys(config["experiment"], "experiment", {"name", "seed", "output_dir"})
    _require_keys(
        config["data"],
        "data",
        {"train_manifest", "val_manifest", "test_manifest", "label_mapping"},
    )
    _require_keys(config["video"], "video", {"preprocessing"})
    _require_keys(config["model"], "model", {"name"})
    _require_keys(
        config["training"],
        "training",
        {"device", "epochs", "batch_size", "optimizer", "loss"},
    )
    _require_keys(config["validation"], "validation", {"batch_size"})
    _require_keys(config["evaluation"], "evaluation", {"batch_size", "metrics"})
    _require_keys(config["checkpointing"], "checkpointing", {"save_dir"})
    _require_keys(config["logging"], "logging", {"log_dir", "level"})

    expected_label_mapping = {"real": 0, "fake": 1}
    if config["data"]["label_mapping"] != expected_label_mapping:
        raise ValueError(
            "data.label_mapping must be {'real': 0, 'fake': 1} "
            "for the baseline video task"
        )

    validate_video_model_config(config["model"])
    validate_video_preprocessing_config(config["video"]["preprocessing"])
    _require_positive_number(config["training"]["epochs"], "training.epochs")
    _require_positive_number(config["training"]["batch_size"], "training.batch_size")
    _require_positive_number(config["validation"]["batch_size"], "validation.batch_size")
    _require_positive_number(config["evaluation"]["batch_size"], "evaluation.batch_size")
    _validate_loss_config(config["training"]["loss"], config["model"]["num_classes"])

    if not config["evaluation"]["metrics"]:
        raise ValueError("evaluation.metrics must contain at least one metric")

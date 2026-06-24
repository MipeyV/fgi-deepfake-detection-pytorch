"""Validation for video model configuration."""


def _require_keys(config: dict, required_keys: set[str]) -> None:
    """Require keys in a video model configuration mapping.

    Args:
        config: Mapping to inspect.
        required_keys: Keys that must be present.

    Raises:
        ValueError: If at least one required key is missing.
    """
    missing_keys = required_keys - set(config)
    if missing_keys:
        raise ValueError(f"Config section 'model' is missing keys: {sorted(missing_keys)}")


def _require_positive(value: int | float, key_path: str) -> None:
    """Require a positive numeric model configuration value.

    Args:
        value: Candidate numeric value.
        key_path: Configuration path included in validation errors.

    Raises:
        ValueError: If ``value`` is not numeric or is not greater than zero.
    """
    if not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"Config value must be greater than 0: {key_path}")


def validate_video_model_config(model_config: dict) -> None:
    """Validate one supported video classifier configuration.

    Args:
        model_config: Experiment ``model`` mapping. Supported names are
            ``video_cnn_baseline`` and ``r3d18``.

    Raises:
        ValueError: If required values are missing or invalid, or if the model
            name is unsupported.
    """
    _require_keys(model_config, {"name", "num_classes", "dropout"})
    _require_positive(model_config["num_classes"], "model.num_classes")

    model_name = model_config["name"]
    if model_name == "video_cnn_baseline":
        _require_keys(model_config, {"input_channels", "conv_channels"})
        _require_positive(model_config["input_channels"], "model.input_channels")
        if not model_config["conv_channels"]:
            raise ValueError("model.conv_channels must contain at least one value")
        return

    if model_name == "r3d18":
        weights = model_config.get("weights", "none")
        if weights not in {"none", "kinetics400_v1"}:
            raise ValueError("model.weights must be 'none' or 'kinetics400_v1' for r3d18")
        if not isinstance(model_config.get("normalize", True), bool):
            raise ValueError("model.normalize must be a boolean for r3d18")
        return

    raise ValueError(f"Unsupported video model: {model_name}")

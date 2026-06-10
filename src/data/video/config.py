"""Validation for video input pipeline configuration."""


def _require_keys(config: dict, required_keys: set[str]) -> None:
    missing_keys = required_keys - set(config)
    if missing_keys:
        raise ValueError(
            "Config section 'video.preprocessing' is missing keys: "
            f"{sorted(missing_keys)}"
        )


def _require_positive(value: int | float, key_path: str) -> None:
    if not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"Config value must be greater than 0: {key_path}")


def validate_video_preprocessing_config(preprocessing_config: dict) -> None:
    """Validate one supported video preprocessing strategy."""
    _require_keys(preprocessing_config, {"name"})
    preprocessing_name = preprocessing_config["name"]

    if preprocessing_name == "resize_square":
        _require_keys(preprocessing_config, {"frame_size"})
        _require_positive(
            preprocessing_config["frame_size"],
            "video.preprocessing.frame_size",
        )
        return

    if preprocessing_name == "resize_center_crop":
        _require_keys(preprocessing_config, {"resize_size", "crop_size"})
        resize_size = preprocessing_config["resize_size"]
        if not isinstance(resize_size, list) or len(resize_size) != 2:
            raise ValueError(
                "video.preprocessing.resize_size must contain height and width"
            )
        for size in resize_size:
            _require_positive(size, "video.preprocessing.resize_size")
        _require_positive(
            preprocessing_config["crop_size"],
            "video.preprocessing.crop_size",
        )
        if preprocessing_config["crop_size"] > min(resize_size):
            raise ValueError(
                "video.preprocessing.crop_size cannot exceed resize_size"
            )
        return

    raise ValueError(
        f"Unsupported video preprocessing: {preprocessing_name}"
    )

"""Factory for video classifiers sharing the project input/output contract."""

from torch import nn

from src.models.video.baseline import VideoCNNBaseline
from src.models.video.r3d18 import R3D18VideoClassifier


def build_video_model(model_config: dict) -> nn.Module:
    """Build a configured video classifier.

    Args:
        model_config: Experiment ``model`` mapping containing the architecture
            name and its constructor parameters.

    Returns:
        A classifier accepting clips shaped
        ``[batch, frames, channels, height, width]`` and returning logits
        shaped ``[batch, classes]``.

    Raises:
        ValueError: If ``model.name`` is unsupported or a constructor rejects
            one of its values.
    """
    model_name = model_config.get("name", "video_cnn_baseline")

    if model_name == "video_cnn_baseline":
        return VideoCNNBaseline(
            input_channels=model_config.get("input_channels", 3),
            num_classes=model_config.get("num_classes", 2),
            conv_channels=model_config.get("conv_channels", (16, 32, 64)),
            dense_channels=model_config.get("dense_channels", (128,)),
            dropout=model_config.get("dropout", 0.3),
        )

    if model_name == "r3d18":
        return R3D18VideoClassifier(
            num_classes=model_config.get("num_classes", 2),
            weights=model_config.get("weights", "none"),
            dropout=model_config.get("dropout", 0.3),
            normalize=model_config.get("normalize", True),
        )

    raise ValueError(f"Unsupported video model: {model_name}")

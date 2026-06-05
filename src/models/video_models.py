"""Video models for deepfake detection baselines."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


__all__ = ["VideoCNNBaseline", "VideoConvBlock", "build_video_model"]


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")


def _validate_positive_sequence(values: Sequence[int], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one value")

    for value in values:
        _validate_positive_int(value, name)


class VideoConvBlock(nn.Module):
    """3D convolution block for video frame sequences."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        _validate_positive_int(in_channels, "in_channels")
        _validate_positive_int(out_channels, "out_channels")

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class VideoCNNBaseline(nn.Module):
    """Simple 3D CNN classifier for video clips.

    The model expects batches produced by ``DeepFakeClipDataset``:
    ``[batch_size, num_frames, channels, height, width]``.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        conv_channels: Sequence[int] = (16, 32, 64),
        dense_channels: Sequence[int] = (128,),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        _validate_positive_int(input_channels, "input_channels")
        _validate_positive_int(num_classes, "num_classes")
        _validate_positive_sequence(conv_channels, "conv_channels")

        if dense_channels:
            _validate_positive_sequence(dense_channels, "dense_channels")

        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv_channels = tuple(conv_channels)
        self.dense_channels = tuple(dense_channels)
        self.dropout = dropout

        conv_layers: list[nn.Module] = []
        in_channels = input_channels

        for out_channels in self.conv_channels:
            conv_layers.append(VideoConvBlock(in_channels, out_channels))
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        classifier_layers: list[nn.Module] = []
        in_features = self.conv_channels[-1]

        for out_features in self.dense_channels:
            classifier_layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_features = out_features

        classifier_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on frame tensors.

        Args:
            inputs: Frame tensor with shape
                ``[batch_size, num_frames, channels, height, width]``.

        Returns:
            Raw class logits with shape ``[batch_size, num_classes]``.

        Raises:
            ValueError: If ``inputs`` does not have five dimensions.
        """
        if inputs.ndim != 5:
            raise ValueError(
                "VideoCNNBaseline expects inputs with shape "
                "[batch_size, num_frames, channels, height, width]"
            )

        inputs = inputs.permute(0, 2, 1, 3, 4)
        features = self.feature_extractor(inputs)
        features = self.pool(features)
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)


def build_video_model(model_config: dict) -> VideoCNNBaseline:
    """Build the baseline video model from a YAML model config section."""
    model_name = model_config.get("name", "video_cnn_baseline")

    if model_name != "video_cnn_baseline":
        raise ValueError(f"Unsupported video model: {model_name}")

    return VideoCNNBaseline(
        input_channels=model_config.get("input_channels", 3),
        num_classes=model_config.get("num_classes", 2),
        conv_channels=model_config.get("conv_channels", (16, 32, 64)),
        dense_channels=model_config.get("dense_channels", (128,)),
        dropout=model_config.get("dropout", 0.3),
    )

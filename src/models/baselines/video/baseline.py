"""Simple 3D CNN video baseline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.models.baselines.video._validation import (
    validate_positive_int,
    validate_positive_sequence,
)


class VideoConvBlock(nn.Module):
    """Spatial 3D convolution block that preserves the temporal resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize one convolution, normalization, activation, and pool block.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.

        Raises:
            ValueError: If either channel count is not positive.
        """
        super().__init__()

        validate_positive_int(in_channels, "in_channels")
        validate_positive_int(out_channels, "out_channels")

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the convolution block to ``[batch, channels, time, height, width]``."""
        return self.layers(inputs)


class VideoCNNBaseline(nn.Module):
    """Configurable 3D CNN baseline classifier for fixed-size video clips."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        conv_channels: Sequence[int] = (16, 32, 64),
        dense_channels: Sequence[int] = (128,),
        dropout: float = 0.3,
    ) -> None:
        """Initialize the convolutional feature extractor and dense classifier.

        Args:
            input_channels: Number of channels in each input frame.
            num_classes: Number of output classification logits.
            conv_channels: Output sizes of successive 3D convolution blocks.
            dense_channels: Hidden sizes in the dense classification head.
            dropout: Dropout probability between dense layers.

        Raises:
            ValueError: If channel or class sizes are invalid, if no
                convolution size is provided, or if dropout is outside
                ``[0, 1]``.
        """
        super().__init__()

        validate_positive_int(input_channels, "input_channels")
        validate_positive_int(num_classes, "num_classes")
        validate_positive_sequence(conv_channels, "conv_channels")

        if dense_channels:
            validate_positive_sequence(dense_channels, "dense_channels")

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
        """Classify clips shaped ``[batch, frames, channels, height, width]``.

        Args:
            inputs: Batch of frame sequences.

        Returns:
            Unnormalized class logits shaped ``[batch, num_classes]``.

        Raises:
            ValueError: If ``inputs`` is not a five-dimensional tensor.
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

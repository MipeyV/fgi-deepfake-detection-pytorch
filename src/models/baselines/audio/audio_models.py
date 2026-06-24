"""Audio models for deepfake detection baselines."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

__all__ = ["AudioCNNBaseline", "ConvBlock", "build_audio_model"]


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that an integer model parameter is positive.

    Args:
        value: Integer value to validate.
        name: Human-readable parameter name used in the error message.

    Raises:
        ValueError: If ``value`` is not strictly positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")


def _validate_positive_sequence(values: Sequence[int], name: str) -> None:
    """Validate that a sequence contains only positive integers.

    Args:
        values: Sequence of integer values to validate.
        name: Human-readable parameter name used in the error message.

    Raises:
        ValueError: If ``values`` is empty or contains a non-positive value.
    """
    if not values:
        raise ValueError(f"{name} must contain at least one value")

    for value in values:
        _validate_positive_int(value, name)


class ConvBlock(nn.Module):
    """Convolution block used by the audio CNN baseline.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels produced by the block.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize convolution, normalization, activation, and pooling layers.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels produced by the block.

        Raises:
            ValueError: If either channel count is not strictly positive.
        """
        super().__init__()

        _validate_positive_int(in_channels, "in_channels")
        _validate_positive_int(out_channels, "out_channels")

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the convolution block to an input tensor.

        Args:
            inputs: Input tensor with shape
                ``[batch_size, in_channels, height, width]``.

        Returns:
            Output tensor after convolution, batch normalization, activation,
            and max pooling.
        """
        return self.layers(inputs)


class AudioCNNBaseline(nn.Module):
    """Simple CNN classifier for mel-spectrogram audio features.

    Args:
        input_channels: Number of spectrogram input channels. Use ``1`` for a
            mono mel-spectrogram.
        num_classes: Number of output classes.
        conv_channels: Output channel sizes for each convolution block.
        dense_channels: Hidden layer sizes for the classifier head.
        dropout: Dropout probability used between dense layers.

    Raises:
        ValueError: If ``conv_channels`` is empty.
        ValueError: If a channel size is not strictly positive.
        ValueError: If ``dropout`` is outside the ``[0, 1]`` range.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        conv_channels: Sequence[int] = (32, 64, 128),
        dense_channels: Sequence[int] = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        """Initialize convolutional and dense classifier layers.

        Args:
            input_channels: Number of spectrogram input channels.
            num_classes: Number of output classes.
            conv_channels: Output channel sizes for each convolution block.
            dense_channels: Hidden layer sizes for the classifier head.
            dropout: Dropout probability used between dense layers.

        Raises:
            ValueError: If a numeric model parameter is invalid.
            ValueError: If ``dropout`` is outside the ``[0, 1]`` range.
        """
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
            conv_layers.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

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
        """Run a forward pass on mel-spectrogram features.

        Args:
            inputs: Mel-spectrogram tensor with shape
                ``[batch_size, input_channels, n_mels, time_steps]``.

        Returns:
            Raw class logits with shape ``[batch_size, num_classes]``.

        Raises:
            ValueError: If ``inputs`` does not have four dimensions.
        """
        if inputs.ndim != 4:
            raise ValueError(
                "AudioCNNBaseline expects inputs with shape "
                "[batch_size, channels, n_mels, time_steps]"
            )

        features = self.feature_extractor(inputs)
        features = self.pool(features)
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)


def build_audio_model(model_config: dict) -> AudioCNNBaseline:
    """Build the baseline audio model from a YAML model config section.

    Args:
        model_config: Configuration dictionary, typically the ``model`` section
            from ``configs/baseline_audio.yaml``. Supported keys are
            ``input_channels``, ``num_classes``, ``conv_channels``,
            ``dense_channels``, and ``dropout``.

    Returns:
        Configured audio CNN baseline model.

    Raises:
        ValueError: If ``conv_channels`` is present but empty.
        ValueError: If ``name`` is provided and is not ``audio_cnn_baseline``.
    """
    model_name = model_config.get("name", "audio_cnn_baseline")

    if model_name != "audio_cnn_baseline":
        raise ValueError(f"Unsupported audio model: {model_name}")

    return AudioCNNBaseline(
        input_channels=model_config.get("input_channels", 1),
        num_classes=model_config.get("num_classes", 2),
        conv_channels=model_config.get("conv_channels", (32, 64, 128)),
        dense_channels=model_config.get("dense_channels", (256, 128)),
        dropout=model_config.get("dropout", 0.3),
    )

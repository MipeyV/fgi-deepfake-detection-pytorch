"""Audio and video encoders for the FGI-inspired model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _validate_positive_int(value: int, name: str) -> None:
    """Require a strictly positive integer-like encoder parameter."""
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")


class FGIConv3DBlock(nn.Module):
    """3D convolution block used by the compact FGI video encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
    ) -> None:
        """Initialize convolution, normalization, activation, and residual path."""
        super().__init__()
        _validate_positive_int(in_channels, "in_channels")
        _validate_positive_int(out_channels, "out_channels")
        if len(stride) != 3 or any(value <= 0 for value in stride):
            raise ValueError("stride must contain three positive values")

        self.layers = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        )
        if in_channels == out_channels and stride == (1, 1, 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode one feature map shaped ``[batch, channels, time, H, W]``."""
        return self.activation(self.layers(inputs) + self.residual(inputs))


class FGIVideoEncoder(nn.Module):
    """Encode face clips into local spatio-temporal feature maps."""

    def __init__(
        self,
        embedding_dim: int = 128,
        temporal_size: int = 15,
        spatial_size: int = 28,
        stem_channels: int = 32,
    ) -> None:
        """Initialize the compact 3D convolutional video encoder.

        Args:
            embedding_dim: Output feature channels shared with audio.
            temporal_size: Output temporal positions.
            spatial_size: Output feature-map height and width.
            stem_channels: Width of the first convolution stage.

        Raises:
            ValueError: If a dimension is not positive.
        """
        super().__init__()
        for value, name in (
            (embedding_dim, "embedding_dim"),
            (temporal_size, "temporal_size"),
            (spatial_size, "spatial_size"),
            (stem_channels, "stem_channels"),
        ):
            _validate_positive_int(value, name)

        self.embedding_dim = embedding_dim
        self.temporal_size = temporal_size
        self.spatial_size = spatial_size
        self.encoder = nn.Sequential(
            FGIConv3DBlock(3, stem_channels, stride=(1, 2, 2)),
            FGIConv3DBlock(
                stem_channels,
                stem_channels * 2,
                stride=(2, 2, 2),
            ),
            FGIConv3DBlock(
                stem_channels * 2,
                embedding_dim,
                stride=(1, 2, 2),
            ),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode face frames into ``[batch, D, time, height, width]``.

        Args:
            frames: Tensor shaped ``[batch, time, 3, height, width]``.

        Returns:
            Local video features shaped
            ``[batch, embedding_dim, temporal_size, spatial_size, spatial_size]``.

        Raises:
            ValueError: If frames are not five-dimensional or not RGB.
        """
        if frames.ndim != 5:
            raise ValueError("FGIVideoEncoder expects [batch, time, channels, height, width]")
        if frames.shape[2] != 3:
            raise ValueError("FGIVideoEncoder expects 3-channel RGB frames")

        features = self.encoder(frames.permute(0, 2, 1, 3, 4))
        return F.adaptive_avg_pool3d(
            features,
            (self.temporal_size, self.spatial_size, self.spatial_size),
        )


class FGIAudioEncoder(nn.Module):
    """Encode raw mono waveforms into temporal features aligned with video."""

    def __init__(
        self,
        embedding_dim: int = 128,
        temporal_size: int = 15,
        hidden_channels: int = 128,
    ) -> None:
        """Initialize the raw-waveform convolutional audio encoder.

        Args:
            embedding_dim: Output feature channels shared with video.
            temporal_size: Output temporal positions.
            hidden_channels: Width of the raw-audio convolutional stem.

        Raises:
            ValueError: If a dimension is not positive.
        """
        super().__init__()
        for value, name in (
            (embedding_dim, "embedding_dim"),
            (temporal_size, "temporal_size"),
            (hidden_channels, "hidden_channels"),
        ):
            _validate_positive_int(value, name)

        self.embedding_dim = embedding_dim
        self.temporal_size = temporal_size
        self.encoder = nn.Sequential(
            nn.Conv1d(
                1,
                hidden_channels,
                kernel_size=80,
                stride=8,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_channels,
                embedding_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio into ``[batch, D, time]``.

        Args:
            audio: Mono waveform tensor shaped ``[batch, samples]``.

        Returns:
            Temporal audio features shaped
            ``[batch, embedding_dim, temporal_size]``.

        Raises:
            ValueError: If audio is not a two-dimensional waveform batch.
        """
        if audio.ndim != 2:
            raise ValueError("FGIAudioEncoder expects [batch, samples]")
        features = self.encoder(audio.unsqueeze(1))
        return F.adaptive_avg_pool1d(features, self.temporal_size)


class FGIEncoderPair(nn.Module):
    """Run aligned audio and video encoders with a shared feature dimension."""

    def __init__(
        self,
        video_encoder: FGIVideoEncoder,
        audio_encoder: FGIAudioEncoder,
    ) -> None:
        """Initialize and validate an aligned encoder pair."""
        super().__init__()
        if video_encoder.embedding_dim != audio_encoder.embedding_dim:
            raise ValueError("Audio and video embedding dimensions must match")
        if video_encoder.temporal_size != audio_encoder.temporal_size:
            raise ValueError("Audio and video temporal sizes must match")
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder

    def forward(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aligned local video and audio features."""
        return self.video_encoder(frames), self.audio_encoder(audio)


def build_fgi_encoders(model_config: dict) -> FGIEncoderPair:
    """Build aligned FGI audio and video encoders from model configuration.

    Args:
        model_config: ``model`` section from ``configs/fgi_inspired.yaml``.

    Returns:
        Configured encoder pair.

    Raises:
        ValueError: If the model name or encoder configuration is invalid.
    """
    if model_config.get("name") != "fgi_inspired":
        raise ValueError("FGI encoders require model.name=fgi_inspired")
    encoder_config = model_config.get("encoders")
    if not isinstance(encoder_config, dict):
        raise ValueError("model.encoders must be a mapping")

    embedding_dim = encoder_config.get("embedding_dim", 128)
    temporal_size = encoder_config.get("temporal_size", 15)
    spatial_size = encoder_config.get("spatial_size", 28)
    video_encoder = FGIVideoEncoder(
        embedding_dim=embedding_dim,
        temporal_size=temporal_size,
        spatial_size=spatial_size,
        stem_channels=encoder_config.get("video_stem_channels", 32),
    )
    audio_encoder = FGIAudioEncoder(
        embedding_dim=embedding_dim,
        temporal_size=temporal_size,
        hidden_channels=encoder_config.get("audio_hidden_channels", 128),
    )
    return FGIEncoderPair(video_encoder, audio_encoder)

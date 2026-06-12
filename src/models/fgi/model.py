"""Complete FGI-inspired inconsistency classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.models.fgi.encoders import FGIEncoderPair, build_fgi_encoders


@dataclass(frozen=True)
class FGIModelOutput:
    """Inspectable outputs produced by the FGI-inspired classifier."""

    logits: torch.Tensor
    inconsistency_map: torch.Tensor
    attention_map: torch.Tensor | None
    video_features: torch.Tensor
    audio_features: torch.Tensor


class FGILocalInconsistency(nn.Module):
    """Compute spatial audio-video distances over channels and time."""

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the numerically stable Euclidean distance.

        Args:
            eps: Positive value added before the square root.

        Raises:
            ValueError: If ``eps`` is not positive.
        """
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be greater than 0")
        self.eps = eps

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return an inconsistency map shaped ``[batch, height, width]``.

        Args:
            video_features: Local video features shaped ``[B, D, T, H, W]``.
            audio_features: Audio features shaped ``[B, D, T]``.

        Returns:
            Euclidean audio-video distances aggregated over ``D`` and ``T``.

        Raises:
            ValueError: If feature dimensions or aligned axes are invalid.
        """
        if video_features.ndim != 5 or audio_features.ndim != 3:
            raise ValueError(
                "FGI features must have shapes [B,D,T,H,W] and [B,D,T]"
            )
        if video_features.shape[:3] != audio_features.shape:
            raise ValueError("Audio and video feature axes B, D, and T must match")

        audio_grid = audio_features.unsqueeze(-1).unsqueeze(-1)
        squared_distance = (video_features - audio_grid).pow(2)
        return torch.sqrt(
            squared_distance.sum(dim=(1, 2)) + self.eps
        )


class FGISpatialAttention(nn.Module):
    """Estimate spatial relevance from aligned audio and video embeddings."""

    def __init__(self, embedding_dim: int, attention_dim: int = 32) -> None:
        """Initialize modality projections used for spatial attention.

        Args:
            embedding_dim: Shared input feature dimension.
            attention_dim: Projected feature dimension.

        Raises:
            ValueError: If a dimension is not positive.
        """
        super().__init__()
        if embedding_dim <= 0 or attention_dim <= 0:
            raise ValueError("Attention dimensions must be positive")
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.video_projection = nn.Conv3d(
            embedding_dim,
            attention_dim,
            kernel_size=1,
        )
        self.audio_projection = nn.Conv1d(
            embedding_dim,
            attention_dim,
            kernel_size=1,
        )

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return a normalized spatial attention map ``[batch, height, width]``."""
        if video_features.shape[:3] != audio_features.shape:
            raise ValueError("Attention requires aligned audio-video features")

        video_embedding = self.video_projection(video_features)
        audio_embedding = self.audio_projection(audio_features)
        scores = torch.einsum(
            "bcthw,bct->bhw",
            video_embedding,
            audio_embedding,
        )
        scores = scores / (
            self.attention_dim * video_features.shape[2]
        )
        batch_size, height, width = scores.shape
        return F.softmax(
            scores.reshape(batch_size, height * width),
            dim=1,
        ).reshape(batch_size, height, width)


class FGIInspiredClassifier(nn.Module):
    """Classify clips from fine-grained audio-visual inconsistencies."""

    def __init__(
        self,
        encoders: FGIEncoderPair,
        num_classes: int = 2,
        attention_enabled: bool = True,
        attention_dim: int = 32,
        attention_mode: str = "residual",
        dropout: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        """Initialize encoders, inconsistency computation, and classifier.

        Args:
            encoders: Aligned FGI audio and video encoders.
            num_classes: Number of output logits.
            attention_enabled: Whether to estimate spatial attention.
            attention_dim: Internal attention projection width.
            attention_mode: ``multiply`` or ``residual`` weighting.
            dropout: Dropout probability before classification.
            eps: Stability value used by local Euclidean distances.

        Raises:
            ValueError: If classifier or attention parameters are invalid.
        """
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if attention_mode not in {"multiply", "residual"}:
            raise ValueError(
                "attention_mode must be 'multiply' or 'residual'"
            )
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

        self.encoders = encoders
        self.num_classes = num_classes
        self.attention_enabled = attention_enabled
        self.attention_mode = attention_mode
        self.dropout = dropout
        self.inconsistency = FGILocalInconsistency(eps=eps)
        if attention_enabled:
            self.attention = FGISpatialAttention(
                embedding_dim=encoders.video_encoder.embedding_dim,
                attention_dim=attention_dim,
            )
        else:
            self.attention = None

        spatial_size = encoders.video_encoder.spatial_size
        map_size = spatial_size * spatial_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(map_size),
            nn.Dropout(dropout),
            nn.Linear(map_size, num_classes),
        )

    def forward(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
    ) -> FGIModelOutput:
        """Return logits and intermediate FGI maps for one multimodal batch."""
        video_features, audio_features = self.encoders(frames, audio)
        inconsistency_map = self.inconsistency(
            video_features,
            audio_features,
        )

        attention_map = None
        classified_map = inconsistency_map
        if self.attention is not None:
            attention_map = self.attention(video_features, audio_features)
            if self.attention_mode == "residual":
                classified_map = inconsistency_map * (1 + attention_map)
            else:
                classified_map = inconsistency_map * attention_map

        logits = self.classifier(classified_map.flatten(start_dim=1))
        return FGIModelOutput(
            logits=logits,
            inconsistency_map=inconsistency_map,
            attention_map=attention_map,
            video_features=video_features,
            audio_features=audio_features,
        )


def build_fgi_model(model_config: dict) -> FGIInspiredClassifier:
    """Build the complete FGI-inspired classifier from configuration.

    Args:
        model_config: ``model`` section from ``configs/fgi_inspired.yaml``.

    Returns:
        Configured FGI-inspired classifier.

    Raises:
        ValueError: If model or attention configuration is invalid.
    """
    encoders = build_fgi_encoders(model_config)
    attention_config = model_config.get("attention", {})
    if not isinstance(attention_config, dict):
        raise ValueError("model.attention must be a mapping")
    return FGIInspiredClassifier(
        encoders=encoders,
        num_classes=model_config.get("num_classes", 2),
        attention_enabled=attention_config.get("enabled", True),
        attention_dim=attention_config.get("embedding_dim", 32),
        attention_mode=attention_config.get("mode", "residual"),
        dropout=model_config.get("dropout", 0.5),
        eps=model_config.get("distance_eps", 1e-8),
    )

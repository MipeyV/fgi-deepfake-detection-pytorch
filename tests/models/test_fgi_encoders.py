import pytest
import torch

from src.config import load_config
from src.models.fgi import (
    FGIAudioEncoder,
    FGIEncoderPair,
    FGIVideoEncoder,
    build_fgi_encoders,
)


def test_fgi_encoders_return_aligned_local_features() -> None:
    encoders = FGIEncoderPair(
        video_encoder=FGIVideoEncoder(
            embedding_dim=8,
            temporal_size=3,
            spatial_size=4,
            stem_channels=4,
        ),
        audio_encoder=FGIAudioEncoder(
            embedding_dim=8,
            temporal_size=3,
            hidden_channels=4,
        ),
    )

    video_features, audio_features = encoders(
        torch.randn(2, 6, 3, 32, 32),
        torch.randn(2, 2048),
    )

    assert video_features.shape == (2, 8, 3, 4, 4)
    assert audio_features.shape == (2, 8, 3)


def test_fgi_encoders_propagate_gradients() -> None:
    encoders = FGIEncoderPair(
        video_encoder=FGIVideoEncoder(
            embedding_dim=4,
            temporal_size=2,
            spatial_size=2,
            stem_channels=2,
        ),
        audio_encoder=FGIAudioEncoder(
            embedding_dim=4,
            temporal_size=2,
            hidden_channels=2,
        ),
    )
    frames = torch.randn(1, 4, 3, 16, 16, requires_grad=True)
    audio = torch.randn(1, 1024, requires_grad=True)

    video_features, audio_features = encoders(frames, audio)
    (video_features.mean() + audio_features.mean()).backward()

    assert frames.grad is not None
    assert audio.grad is not None
    assert any(parameter.grad is not None for parameter in encoders.parameters())


def test_build_fgi_encoders_uses_project_config() -> None:
    config = load_config("configs/fgi_inspired.yaml")

    encoders = build_fgi_encoders(config["model"])

    assert encoders.video_encoder.embedding_dim == 128
    assert encoders.video_encoder.temporal_size == 15
    assert encoders.video_encoder.spatial_size == 28
    assert encoders.audio_encoder.embedding_dim == 128


def test_fgi_encoder_pair_rejects_misaligned_dimensions() -> None:
    with pytest.raises(ValueError, match="embedding dimensions"):
        FGIEncoderPair(
            video_encoder=FGIVideoEncoder(
                embedding_dim=8,
                stem_channels=4,
            ),
            audio_encoder=FGIAudioEncoder(
                embedding_dim=4,
                hidden_channels=4,
            ),
        )


def test_fgi_video_encoder_rejects_non_rgb_frames() -> None:
    encoder = FGIVideoEncoder(stem_channels=4)

    with pytest.raises(ValueError, match="3-channel RGB"):
        encoder(torch.randn(1, 4, 1, 16, 16))


def test_fgi_audio_encoder_rejects_channel_dimension() -> None:
    encoder = FGIAudioEncoder(hidden_channels=4)

    with pytest.raises(ValueError, match="batch, samples"):
        encoder(torch.randn(1, 1, 1024))

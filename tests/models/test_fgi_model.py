import pytest
import torch

from src.config import load_config
from src.models.fgi import (
    FGIAudioEncoder,
    FGIEncoderPair,
    FGIInspiredClassifier,
    FGILocalInconsistency,
    FGIVideoEncoder,
    build_fgi_model,
)


def make_small_model(
    attention_enabled: bool = True,
    attention_mode: str = "residual",
) -> FGIInspiredClassifier:
    """Build a lightweight FGI classifier for unit tests."""
    encoders = FGIEncoderPair(
        video_encoder=FGIVideoEncoder(
            embedding_dim=4,
            temporal_size=2,
            spatial_size=3,
            stem_channels=2,
        ),
        audio_encoder=FGIAudioEncoder(
            embedding_dim=4,
            temporal_size=2,
            hidden_channels=2,
        ),
    )
    return FGIInspiredClassifier(
        encoders=encoders,
        num_classes=2,
        attention_enabled=attention_enabled,
        attention_dim=2,
        attention_mode=attention_mode,
        dropout=0.0,
    )


def test_local_inconsistency_matches_euclidean_distance() -> None:
    video = torch.ones(1, 2, 3, 2, 2)
    audio = torch.zeros(1, 2, 3)

    distance = FGILocalInconsistency(eps=1e-12)(video, audio)

    assert distance.shape == (1, 2, 2)
    assert torch.allclose(
        distance,
        torch.full((1, 2, 2), 6**0.5),
    )


def test_fgi_model_returns_logits_maps_and_aligned_features() -> None:
    model = make_small_model()

    output = model(
        torch.randn(2, 4, 3, 16, 16),
        torch.randn(2, 1024),
    )

    assert output.logits.shape == (2, 2)
    assert output.inconsistency_map.shape == (2, 3, 3)
    assert output.attention_map is not None
    assert output.attention_map.shape == (2, 3, 3)
    assert torch.allclose(
        output.attention_map.sum(dim=(1, 2)),
        torch.ones(2),
    )
    assert output.video_features.shape == (2, 4, 2, 3, 3)
    assert output.audio_features.shape == (2, 4, 2)


def test_fgi_model_can_disable_attention() -> None:
    output = make_small_model(attention_enabled=False)(
        torch.randn(1, 4, 3, 16, 16),
        torch.randn(1, 1024),
    )

    assert output.logits.shape == (1, 2)
    assert output.attention_map is None


def test_fgi_model_propagates_classification_gradients() -> None:
    model = make_small_model(attention_mode="multiply")
    frames = torch.randn(1, 4, 3, 16, 16, requires_grad=True)
    audio = torch.randn(1, 1024, requires_grad=True)

    model(frames, audio).logits.sum().backward()

    assert frames.grad is not None
    assert audio.grad is not None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_build_fgi_model_uses_project_config() -> None:
    config = load_config("configs/fgi_inspired.yaml")

    model = build_fgi_model(config["model"])

    assert model.num_classes == 2
    assert model.attention_enabled
    assert model.attention_mode == "residual"
    assert model.encoders.video_encoder.spatial_size == 28


def test_fgi_model_rejects_unknown_attention_mode() -> None:
    with pytest.raises(ValueError, match="attention_mode"):
        make_small_model(attention_mode="unknown")

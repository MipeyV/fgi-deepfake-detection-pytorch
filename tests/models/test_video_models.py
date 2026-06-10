import pytest
import torch

from src.models.video import (
    R3D18VideoClassifier,
    VideoCNNBaseline,
    build_video_model,
)


def test_video_cnn_baseline_returns_class_logits() -> None:
    model = VideoCNNBaseline(
        input_channels=3,
        num_classes=2,
        conv_channels=(4,),
        dense_channels=(8,),
        dropout=0.0,
    )

    logits = model(torch.randn(2, 3, 3, 8, 8))

    assert logits.shape == (2, 2)


def test_build_video_model_uses_config_values() -> None:
    model = build_video_model(
        {
            "name": "video_cnn_baseline",
            "input_channels": 3,
            "num_classes": 2,
            "conv_channels": [4, 8],
            "dense_channels": [16],
            "dropout": 0.1,
        }
    )

    assert model.conv_channels == (4, 8)
    assert model.dense_channels == (16,)
    assert model.dropout == 0.1


def test_video_cnn_baseline_rejects_non_video_inputs() -> None:
    model = VideoCNNBaseline(conv_channels=(4,), dense_channels=(), dropout=0.0)

    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 3, 8, 8))


def test_build_video_model_rejects_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="Unsupported video model"):
        build_video_model({"name": "unknown"})


def test_build_video_model_builds_r3d18_classifier() -> None:
    model = build_video_model(
        {
            "name": "r3d18",
            "num_classes": 2,
            "weights": "none",
            "normalize": True,
            "dropout": 0.1,
        }
    )

    assert isinstance(model, R3D18VideoClassifier)
    assert model.backbone.fc[-1].out_features == 2
    assert model.dropout == 0.1


def test_r3d18_rejects_non_rgb_inputs() -> None:
    model = R3D18VideoClassifier(weights="none")

    with pytest.raises(ValueError, match="3-channel RGB"):
        model(torch.randn(1, 4, 1, 32, 32))


def test_r3d18_rejects_unknown_weights() -> None:
    with pytest.raises(ValueError, match="weights"):
        R3D18VideoClassifier(weights="unknown")

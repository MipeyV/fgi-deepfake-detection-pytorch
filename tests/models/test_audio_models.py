import pytest
import torch

from src.models.baselines.audio.audio_models import AudioCNNBaseline, build_audio_model


def test_audio_cnn_baseline_returns_class_logits() -> None:
    model = AudioCNNBaseline()
    inputs = torch.randn(4, 1, 128, 94)

    logits = model(inputs)

    assert logits.shape == (4, 2)


def test_build_audio_model_uses_config_values() -> None:
    model = build_audio_model(
        {
            "input_channels": 1,
            "num_classes": 2,
            "conv_channels": [16, 32],
            "dense_channels": [64],
            "dropout": 0.2,
        }
    )
    inputs = torch.randn(2, 1, 128, 94)

    logits = model(inputs)

    assert logits.shape == (2, 2)


def test_audio_cnn_baseline_rejects_non_spectrogram_inputs() -> None:
    model = AudioCNNBaseline()

    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(4, 48000))


def test_audio_cnn_baseline_rejects_empty_conv_channels() -> None:
    with pytest.raises(ValueError, match="conv_channels"):
        AudioCNNBaseline(conv_channels=())


def test_audio_cnn_baseline_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        AudioCNNBaseline(dropout=1.5)


def test_build_audio_model_rejects_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="Unsupported audio model"):
        build_audio_model({"name": "unknown_model"})

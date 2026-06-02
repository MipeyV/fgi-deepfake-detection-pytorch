import pytest
import torch

from src.data.audio_feature import (
    MelSpectrogramExtractor,
    build_audio_feature_extractor,
    create_mel_filterbank,
)


def test_create_mel_filterbank_returns_expected_shape() -> None:
    filterbank = create_mel_filterbank(
        sample_rate=16000,
        n_fft=512,
        n_mels=40,
        f_min=20,
        f_max=8000,
    )

    assert filterbank.shape == (40, 257)
    assert torch.all(filterbank >= 0)


def test_mel_spectrogram_extractor_handles_single_waveform() -> None:
    extractor = MelSpectrogramExtractor(
        sample_rate=16000,
        n_mels=40,
        n_fft=512,
        hop_length=160,
        win_length=400,
    )
    waveform = torch.randn(1, 16000)

    features = extractor(waveform)

    assert features.ndim == 3
    assert features.shape[0] == 1
    assert features.shape[1] == 40


def test_mel_spectrogram_extractor_handles_batched_waveforms() -> None:
    extractor = MelSpectrogramExtractor(
        sample_rate=16000,
        n_mels=64,
        n_fft=512,
        hop_length=160,
        win_length=400,
    )
    waveforms = torch.randn(3, 1, 16000)

    features = extractor(waveforms)

    assert features.ndim == 4
    assert features.shape[0] == 3
    assert features.shape[1] == 1
    assert features.shape[2] == 64


def test_build_audio_feature_extractor_uses_config_values() -> None:
    extractor = build_audio_feature_extractor(
        {
            "type": "mel_spectrogram",
            "sample_rate": 16000,
            "n_mels": 32,
            "n_fft": 512,
            "hop_length": 128,
            "win_length": 512,
            "f_min": 20,
            "f_max": 8000,
            "power": 2.0,
            "log_scale": True,
        }
    )

    assert extractor.sample_rate == 16000
    assert extractor.n_mels == 32
    assert extractor.n_fft == 512


def test_build_audio_feature_extractor_can_use_audio_sample_rate() -> None:
    extractor = build_audio_feature_extractor(
        {
            "type": "mel_spectrogram",
            "n_mels": 32,
            "n_fft": 512,
            "hop_length": 128,
            "win_length": 512,
        },
        sample_rate=22050,
    )

    assert extractor.sample_rate == 22050


def test_mel_spectrogram_extractor_rejects_invalid_waveform_shape() -> None:
    extractor = MelSpectrogramExtractor()

    with pytest.raises(ValueError, match="waveform shape"):
        extractor(torch.randn(2, 1, 1, 48000))


def test_build_audio_feature_extractor_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported audio feature type"):
        build_audio_feature_extractor({"type": "mfcc"})

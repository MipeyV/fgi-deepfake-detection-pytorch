"""Audio feature extraction utilities."""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "MelSpectrogramExtractor",
    "build_audio_feature_extractor",
    "create_mel_filterbank",
]


def _hz_to_mel(frequency_hz: torch.Tensor) -> torch.Tensor:
    """Convert frequencies from Hertz to the mel scale.

    Args:
        frequency_hz: Frequencies in Hertz.

    Returns:
        Frequencies converted to mel units.
    """
    return 2595.0 * torch.log10(1.0 + frequency_hz / 700.0)


def _mel_to_hz(frequency_mel: torch.Tensor) -> torch.Tensor:
    """Convert frequencies from the mel scale to Hertz.

    Args:
        frequency_mel: Frequencies in mel units.

    Returns:
        Frequencies converted to Hertz.
    """
    return 700.0 * (10.0 ** (frequency_mel / 2595.0) - 1.0)


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that an integer value is strictly positive.

    Args:
        value: Integer value to validate.
        name: Human-readable parameter name used in the error message.

    Raises:
        ValueError: If ``value`` is not strictly positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Create a triangular mel filterbank matrix.

    Args:
        sample_rate: Audio sampling rate in Hertz.
        n_fft: FFT window size.
        n_mels: Number of mel bands to produce.
        f_min: Minimum frequency in Hertz.
        f_max: Maximum frequency in Hertz. Defaults to the Nyquist frequency.

    Returns:
        Mel filterbank with shape ``[n_mels, n_fft // 2 + 1]``.

    Raises:
        ValueError: If frequency bounds or shape parameters are invalid.
    """
    _validate_positive_int(sample_rate, "sample_rate")
    _validate_positive_int(n_fft, "n_fft")
    _validate_positive_int(n_mels, "n_mels")

    max_frequency = float(sample_rate // 2) if f_max is None else float(f_max)

    if f_min < 0:
        raise ValueError("f_min must be greater than or equal to 0")

    if max_frequency <= f_min:
        raise ValueError("f_max must be greater than f_min")

    if max_frequency > sample_rate / 2:
        raise ValueError("f_max must be less than or equal to sample_rate / 2")

    min_mel = _hz_to_mel(torch.tensor(float(f_min)))
    max_mel = _hz_to_mel(torch.tensor(max_frequency))
    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fft_bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    fft_bins = torch.clamp(fft_bins, min=0, max=n_fft // 2)

    filterbank = torch.zeros(n_mels, n_fft // 2 + 1)

    for mel_idx in range(n_mels):
        left_bin = fft_bins[mel_idx].item()
        center_bin = fft_bins[mel_idx + 1].item()
        right_bin = fft_bins[mel_idx + 2].item()

        if center_bin > left_bin:
            filterbank[mel_idx, left_bin:center_bin] = torch.linspace(
                0.0,
                1.0,
                center_bin - left_bin,
            )

        if right_bin > center_bin:
            filterbank[mel_idx, center_bin:right_bin] = torch.linspace(
                1.0,
                0.0,
                right_bin - center_bin,
            )

    return filterbank


class MelSpectrogramExtractor(nn.Module):
    """Convert waveform tensors into mel-spectrogram features.

    Args:
        sample_rate: Audio sampling rate in Hertz.
        n_mels: Number of mel bands to produce.
        n_fft: FFT window size.
        hop_length: Number of samples between two adjacent STFT windows.
        win_length: Window length in samples. Defaults to ``n_fft``.
        f_min: Minimum mel filter frequency in Hertz.
        f_max: Maximum mel filter frequency in Hertz. Defaults to Nyquist.
        power: Exponent applied to the magnitude spectrogram.
        log_scale: Whether to apply logarithmic compression.
        eps: Small value used for numerical stability in log compression.

    Raises:
        ValueError: If numeric parameters are invalid.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        power: float = 2.0,
        log_scale: bool = True,
        eps: float = 1e-10,
    ) -> None:
        """Initialize mel filterbank, STFT window, and validation settings.

        Args:
            sample_rate: Audio sampling rate in Hertz.
            n_mels: Number of mel bands to produce.
            n_fft: FFT window size.
            hop_length: Number of samples between adjacent STFT windows.
            win_length: Window length in samples. Defaults to ``n_fft``.
            f_min: Minimum mel filter frequency in Hertz.
            f_max: Maximum mel filter frequency in Hertz. Defaults to Nyquist.
            power: Exponent applied to the magnitude spectrogram.
            log_scale: Whether to apply logarithmic compression.
            eps: Small value used for numerical stability in log compression.

        Raises:
            ValueError: If numeric parameters are invalid.
        """
        super().__init__()

        window_length = n_fft if win_length is None else win_length

        _validate_positive_int(hop_length, "hop_length")
        _validate_positive_int(window_length, "win_length")

        if window_length > n_fft:
            raise ValueError("win_length must be less than or equal to n_fft")

        if power <= 0:
            raise ValueError("power must be greater than 0")

        if eps <= 0:
            raise ValueError("eps must be greater than 0")

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = window_length
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.log_scale = log_scale
        self.eps = eps

        filterbank = create_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        window = torch.hann_window(window_length)

        self.register_buffer("mel_filterbank", filterbank)
        self.register_buffer("window", window)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram features from a waveform tensor.

        Args:
            waveform: Audio tensor with shape ``[channels, samples]`` or
                ``[batch_size, channels, samples]``.

        Returns:
            Mel-spectrogram tensor with shape ``[channels, n_mels, time_steps]``
            for a single example, or
            ``[batch_size, channels, n_mels, time_steps]`` for batched input.

        Raises:
            ValueError: If ``waveform`` does not have 2 or 3 dimensions.
        """
        if waveform.ndim not in (2, 3):
            raise ValueError(
                "MelSpectrogramExtractor expects waveform shape "
                "[channels, samples] or [batch_size, channels, samples]"
            )

        is_single_example = waveform.ndim == 2

        if is_single_example:
            waveform = waveform.unsqueeze(0)

        batch_size, channels, num_samples = waveform.shape
        flattened_waveform = waveform.reshape(batch_size * channels, num_samples)

        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        mel_filterbank = self.mel_filterbank.to(
            device=waveform.device,
            dtype=waveform.dtype,
        )

        stft = torch.stft(
            flattened_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )

        spectrogram = stft.abs().pow(self.power)
        mel_spectrogram = torch.matmul(mel_filterbank, spectrogram)

        if self.log_scale:
            mel_spectrogram = torch.log(mel_spectrogram + self.eps)

        mel_spectrogram = mel_spectrogram.reshape(
            batch_size,
            channels,
            self.n_mels,
            mel_spectrogram.shape[-1],
        )

        if is_single_example:
            return mel_spectrogram.squeeze(0)

        return mel_spectrogram


def build_audio_feature_extractor(
    features_config: dict,
    sample_rate: int | None = None,
) -> MelSpectrogramExtractor:
    """Build an audio feature extractor from a YAML feature config section.

    Args:
        features_config: Configuration dictionary, typically the ``features``
            section from ``configs/baseline_audio.yaml``.
        sample_rate: Optional sampling rate from the YAML ``audio`` section.
            If omitted, the builder uses ``features_config["sample_rate"]`` when
            present, otherwise ``48000``.

    Returns:
        Configured mel-spectrogram extractor.

    Raises:
        ValueError: If ``type`` is provided and is not ``mel_spectrogram``.
    """
    feature_type = features_config.get("type", "mel_spectrogram")

    if feature_type != "mel_spectrogram":
        raise ValueError(f"Unsupported audio feature type: {feature_type}")

    return MelSpectrogramExtractor(
        sample_rate=sample_rate or features_config.get("sample_rate", 48000),
        n_mels=features_config.get("n_mels", 128),
        n_fft=features_config.get("n_fft", 2048),
        hop_length=features_config.get("hop_length", 512),
        win_length=features_config.get("win_length"),
        f_min=features_config.get("f_min", 0.0),
        f_max=features_config.get("f_max"),
        power=features_config.get("power", 2.0),
        log_scale=features_config.get("log_scale", True),
    )

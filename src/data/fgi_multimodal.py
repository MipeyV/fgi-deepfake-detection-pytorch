"""Strict synchronized audio-video dataset for FGI-inspired models."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.video.transforms import FrameTransform


REQUIRED_MANIFEST_COLUMNS = {"clip_path", "label", "video_id", "clip_id"}


class FGIMultimodalDataset(Dataset):
    """Load synchronized face frames and raw audio with a strict FGI contract."""

    def __init__(
        self,
        manifest_path: str | Path,
        frame_transform: FrameTransform,
        expected_frames: int = 30,
        sample_rate: int = 48000,
        audio_samples: int = 48000,
        audio_normalization: str = "minmax",
    ) -> None:
        """Initialize the synchronized multimodal dataset.

        Args:
            manifest_path: CSV manifest referencing face-cropped clips.
            frame_transform: Transform applied independently to every frame.
            expected_frames: Exact number of JPEG frames required per clip.
            sample_rate: Required WAV sample rate in Hertz.
            audio_samples: Exact number of mono samples required per clip.
            audio_normalization: ``minmax`` for per-clip scaling to ``[-1, 1]``
                or ``pcm16`` for fixed signed-16-bit scaling.

        Raises:
            FileNotFoundError: If the manifest does not exist.
            ValueError: If configuration or manifest values are invalid.
        """
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest file does not exist: {self.manifest_path}"
            )
        if expected_frames <= 0 or sample_rate <= 0 or audio_samples <= 0:
            raise ValueError("FGI shape and sample-rate values must be positive")
        if audio_normalization not in {"minmax", "pcm16"}:
            raise ValueError(
                "audio_normalization must be 'minmax' or 'pcm16'"
            )

        self.samples = pd.read_csv(
            self.manifest_path,
            dtype={
                "clip_path": str,
                "label": str,
                "video_id": str,
                "clip_id": str,
            },
        )
        missing_columns = REQUIRED_MANIFEST_COLUMNS - set(self.samples.columns)
        if missing_columns:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing_columns)}"
            )
        invalid_labels = set(self.samples["label"]) - {"real", "fake"}
        if invalid_labels:
            raise ValueError(
                f"Manifest contains invalid labels: {sorted(invalid_labels)}"
            )

        self.frame_transform = frame_transform
        self.expected_frames = expected_frames
        self.sample_rate = sample_rate
        self.audio_samples = audio_samples
        self.audio_normalization = audio_normalization
        self.label_to_idx = {"real": 0, "fake": 1}

    def __len__(self) -> int:
        """Return the number of synchronized clips in the manifest."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        """Load one synchronized FGI sample.

        Args:
            index: Manifest row index.

        Returns:
            Mapping containing frames ``[time, 3, height, width]``, audio
            ``[samples]``, a scalar class label, and clip metadata.
        """
        row = self.samples.iloc[index]
        clip_path = Path(row["clip_path"])
        return {
            "frames": self._load_frames(clip_path),
            "audio": self._load_audio(clip_path),
            "label": torch.tensor(
                self.label_to_idx[row["label"]],
                dtype=torch.long,
            ),
            "clip_path": clip_path,
            "video_id": row["video_id"],
            "clip_id": row["clip_id"],
        }

    def _load_frames(self, clip_path: Path) -> torch.Tensor:
        """Load and transform the exact configured number of face frames."""
        if not clip_path.is_dir():
            raise FileNotFoundError(f"Clip directory does not exist: {clip_path}")
        frame_paths = sorted(clip_path.glob("*.jpg"))
        if len(frame_paths) != self.expected_frames:
            raise ValueError(
                f"Expected {self.expected_frames} frames in {clip_path}, "
                f"found {len(frame_paths)}"
            )

        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frames.append(self.frame_transform(image.convert("RGB")))
        return torch.stack(frames)

    def _load_audio(self, clip_path: Path) -> torch.Tensor:
        """Load, validate, and normalize one synchronized mono waveform."""
        audio_path = clip_path / "audio.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        with wave.open(str(audio_path), "rb") as audio_file:
            channels = audio_file.getnchannels()
            sample_width = audio_file.getsampwidth()
            sample_rate = audio_file.getframerate()
            num_samples = audio_file.getnframes()
            raw_audio = audio_file.readframes(num_samples)

        if channels != 1:
            raise ValueError(f"FGI audio must be mono: {audio_path}")
        if sample_width != 2:
            raise ValueError(f"FGI audio must use 16-bit PCM: {audio_path}")
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected audio sample rate {self.sample_rate}, "
                f"found {sample_rate}: {audio_path}"
            )
        if num_samples != self.audio_samples:
            raise ValueError(
                f"Expected {self.audio_samples} audio samples, "
                f"found {num_samples}: {audio_path}"
            )

        audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
        if self.audio_normalization == "pcm16":
            audio /= 32768.0
        else:
            audio_min = float(audio.min())
            audio_max = float(audio.max())
            if audio_max == audio_min:
                audio = np.zeros_like(audio)
            else:
                audio = 2.0 * (audio - audio_min) / (audio_max - audio_min) - 1.0
        return torch.from_numpy(audio)


def collate_fgi_multimodal_batch(batch: list[dict]) -> dict:
    """Stack strict FGI samples and preserve their metadata as lists.

    Args:
        batch: Samples returned by ``FGIMultimodalDataset``.

    Returns:
        Batch with frames shaped ``[batch, time, 3, height, width]``, audio
        shaped ``[batch, samples]``, and labels shaped ``[batch]``.
    """
    return {
        "frames": torch.stack([sample["frames"] for sample in batch]),
        "audio": torch.stack([sample["audio"] for sample in batch]),
        "label": torch.stack([sample["label"] for sample in batch]),
        "clip_path": [sample["clip_path"] for sample in batch],
        "video_id": [sample["video_id"] for sample in batch],
        "clip_id": [sample["clip_id"] for sample in batch],
    }


def create_fgi_multimodal_dataloader(
    manifest_path: str | Path,
    frame_transform: FrameTransform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    expected_frames: int = 30,
    sample_rate: int = 48000,
    audio_samples: int = 48000,
    audio_normalization: str = "minmax",
) -> DataLoader:
    """Create a dataloader for strict synchronized FGI clips.

    Args:
        manifest_path: CSV manifest referencing face-cropped clips.
        frame_transform: Transform applied to each face frame.
        batch_size: Number of clips per batch.
        shuffle: Whether to randomize sample order.
        num_workers: Number of PyTorch loader workers.
        expected_frames: Exact number of frames per clip.
        sample_rate: Required WAV sample rate.
        audio_samples: Exact waveform length.
        audio_normalization: ``minmax`` or ``pcm16``.

    Returns:
        Configured PyTorch dataloader.

    Raises:
        ValueError: If ``batch_size`` or ``num_workers`` is invalid.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    dataset = FGIMultimodalDataset(
        manifest_path=manifest_path,
        frame_transform=frame_transform,
        expected_frames=expected_frames,
        sample_rate=sample_rate,
        audio_samples=audio_samples,
        audio_normalization=audio_normalization,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fgi_multimodal_batch,
    )


def validate_fgi_multimodal_config(config: dict) -> None:
    """Validate the data contract required by the future FGI model.

    Args:
        config: Full FGI experiment configuration.

    Raises:
        ValueError: If required sections or synchronized shape values are
            missing or invalid.
    """
    required_sections = {"data", "video", "audio", "model", "training"}
    missing_sections = required_sections - set(config)
    if missing_sections:
        raise ValueError(
            f"FGI config is missing sections: {sorted(missing_sections)}"
        )

    preprocessing = config["video"].get("preprocessing", {})
    if preprocessing.get("name") != "resize_normalize":
        raise ValueError(
            "FGI video preprocessing must use resize_normalize"
        )
    for key in ("frame_size", "mean", "std"):
        if key not in preprocessing:
            raise ValueError(
                f"FGI video preprocessing is missing: {key}"
            )

    expected_frames = config["video"].get("expected_frames")
    if not isinstance(expected_frames, int) or expected_frames <= 0:
        raise ValueError("video.expected_frames must be a positive integer")

    audio_config = config["audio"]
    for key in ("sample_rate", "num_samples", "normalization"):
        if key not in audio_config:
            raise ValueError(f"FGI audio config is missing: {key}")
    if audio_config["sample_rate"] <= 0 or audio_config["num_samples"] <= 0:
        raise ValueError("FGI audio dimensions must be positive")
    if audio_config["normalization"] not in {"minmax", "pcm16"}:
        raise ValueError("Unsupported FGI audio normalization")

    if config["model"].get("name") != "fgi_inspired":
        raise ValueError("FGI config model.name must be fgi_inspired")
    implementation_status = config["model"].get("implementation_status")
    if implementation_status not in {
        "pending",
        "encoders_ready",
        "model_ready",
    }:
        raise ValueError(
            "FGI model status must be pending, encoders_ready, or model_ready"
        )
    encoder_config = config["model"].get("encoders")
    if not isinstance(encoder_config, dict):
        raise ValueError("model.encoders must be a mapping")
    for key in (
        "embedding_dim",
        "temporal_size",
        "spatial_size",
        "video_stem_channels",
        "audio_hidden_channels",
    ):
        value = encoder_config.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"model.encoders.{key} must be a positive integer")
    if implementation_status == "model_ready":
        attention_config = config["model"].get("attention")
        if not isinstance(attention_config, dict):
            raise ValueError("model.attention must be a mapping")
        if attention_config.get("mode") not in {"multiply", "residual"}:
            raise ValueError("Unsupported FGI attention mode")

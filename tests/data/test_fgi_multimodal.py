from pathlib import Path

import pytest
import torch
from tests.data.helpers import create_clip, write_manifest

from src.config import load_config
from src.data.fgi import build_fgi_input_pipeline
from src.data.fgi_multimodal import (
    FGIMultimodalDataset,
    validate_fgi_multimodal_config,
)
from src.data.video import build_resize_normalize_transform


def make_transform():
    """Build a small FGI-style transform for dataset tests."""
    return build_resize_normalize_transform(
        frame_size=6,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )


def test_fgi_multimodal_dataset_returns_strict_synchronized_sample(
    tmp_path: Path,
) -> None:
    clip_path = tmp_path / "processed_fgi" / "real" / "video_001" / "clips" / "000000"
    create_clip(
        clip_path,
        num_frames=30,
        audio_samples=48000,
        varying_audio=True,
    )
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path)

    sample = FGIMultimodalDataset(
        manifest_path=manifest_path,
        frame_transform=make_transform(),
    )[0]

    assert sample["frames"].shape == (30, 3, 6, 6)
    assert sample["audio"].shape == (48000,)
    assert sample["label"].shape == ()
    assert sample["audio"].min().item() == pytest.approx(-1.0)
    assert sample["audio"].max().item() == pytest.approx(1.0)


def test_fgi_pipeline_builds_expected_batch(tmp_path: Path) -> None:
    clip_path = tmp_path / "processed_fgi" / "fake" / "video_001" / "clips" / "000000"
    create_clip(clip_path, num_frames=3, audio_samples=12)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path, label="fake")
    config = {
        "video": {
            "expected_frames": 3,
            "preprocessing": {
                "name": "resize_normalize",
                "frame_size": 6,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
        },
        "audio": {
            "sample_rate": 48000,
            "num_samples": 12,
            "normalization": "minmax",
        },
    }

    batch = next(
        iter(
            build_fgi_input_pipeline(config).create_dataloader(
                manifest_path=manifest_path,
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        )
    )

    assert batch["frames"].shape == (1, 3, 3, 6, 6)
    assert batch["audio"].shape == (1, 12)
    assert batch["label"].tolist() == [1]
    assert torch.all(batch["frames"] >= -1)
    assert torch.all(batch["frames"] <= 1)


def test_fgi_dataset_rejects_wrong_frame_count(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip"
    create_clip(clip_path, num_frames=2)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path)

    dataset = FGIMultimodalDataset(
        manifest_path=manifest_path,
        frame_transform=make_transform(),
        expected_frames=3,
    )

    with pytest.raises(ValueError, match="Expected 3 frames"):
        dataset[0]


def test_fgi_dataset_rejects_wrong_audio_sample_rate(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip"
    create_clip(clip_path, num_frames=3, sample_rate=16000)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path)

    dataset = FGIMultimodalDataset(
        manifest_path=manifest_path,
        frame_transform=make_transform(),
        expected_frames=3,
    )

    with pytest.raises(ValueError, match="sample rate 48000"):
        dataset[0]


def test_validate_fgi_multimodal_config_accepts_project_config() -> None:
    config = load_config("configs/fgi_inspired.yaml")

    validate_fgi_multimodal_config(config)


def test_validate_fgi_multimodal_config_rejects_unknown_status() -> None:
    config = load_config("configs/fgi_inspired.yaml")
    config["model"]["implementation_status"] = "unknown"

    with pytest.raises(ValueError, match="pending, encoders_ready, or model_ready"):
        validate_fgi_multimodal_config(config)

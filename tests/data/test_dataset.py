from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.dataset import DeepFakeClipDataset


def create_clip(clip_path: Path, num_frames: int = 3) -> None:
    clip_path.mkdir(parents=True)

    for index in range(num_frames):
        image = Image.new("RGB", (8, 8), color=(index * 30, 10, 20))
        image.save(clip_path / f"{index + 1:06d}.jpg")


def write_manifest(manifest_path: Path, clip_path: Path, label: str = "real") -> None:
    manifest_path.write_text(
        "\n".join(
            [
                "clip_path,label,video_id,clip_id",
                f"{clip_path},{label},video_001,000000",
            ]
        ),
        encoding="utf-8",
    )


def test_dataset_loads_clip_frames_and_metadata(tmp_path: Path) -> None:
    clip_path = tmp_path / "preprocessed" / "real" / "video_001" / "clips" / "000000"
    create_clip(clip_path, num_frames=3)

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path)

    dataset = DeepFakeClipDataset(manifest_path)
    sample = dataset[0]

    assert len(dataset) == 1
    assert isinstance(sample["frames"], torch.Tensor)
    assert sample["frames"].shape == (3, 3, 8, 8)
    assert sample["label"].item() == 0
    assert sample["clip_path"] == clip_path
    assert sample["video_id"] == "video_001"
    assert sample["clip_id"] == "000000"


def test_dataset_raises_for_missing_manifest_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "clip_path,label,video_id",
                "some/path,real,video_001",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Manifest is missing required columns"):
        DeepFakeClipDataset(manifest_path)


def test_dataset_raises_for_invalid_label(tmp_path: Path) -> None:
    clip_path = tmp_path / "preprocessed" / "unknown" / "video_001" / "clips" / "000000"
    create_clip(clip_path)

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path, label="unknown")

    with pytest.raises(ValueError, match="Manifest contains invalid labels"):
        DeepFakeClipDataset(manifest_path)


def test_dataset_raises_when_clip_directory_is_missing(tmp_path: Path) -> None:
    clip_path = tmp_path / "missing_clip"
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path)

    dataset = DeepFakeClipDataset(manifest_path)

    with pytest.raises(FileNotFoundError, match="Clip directory does not exist"):
        dataset[0]


def test_dataset_raises_when_clip_has_no_frames(tmp_path: Path) -> None:
    clip_path = tmp_path / "preprocessed" / "real" / "video_001" / "clips" / "000000"
    clip_path.mkdir(parents=True)

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path)

    dataset = DeepFakeClipDataset(manifest_path)

    with pytest.raises(FileNotFoundError, match="No frames found"):
        dataset[0]

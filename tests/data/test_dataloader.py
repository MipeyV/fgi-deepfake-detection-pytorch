from pathlib import Path

import pytest
import torch

from src.data.dataloader import create_dataloader
from tests.data.helpers import create_clip, write_manifest


def test_create_data_loader(tmp_path: Path) -> None:
    clip_path = tmp_path / "preprocessed" / "real" / "video_001" / "clips" / "000000"
    create_clip(clip_path, num_frames=3)

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path)

    dataloader = create_dataloader(manifest_path, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    assert isinstance(batch["frames"], torch.Tensor)
    assert batch["frames"].shape == (1, 3, 3, 8, 8)
    assert isinstance(batch["audio"], torch.Tensor)
    assert batch["audio"].ndim == 3
    assert batch["audio"].shape == (1, 1, 48000)
    assert batch["audio"].dtype == torch.float32
    assert batch["label"].shape == (1,)
    assert batch["label"].item() == 0
    assert batch["clip_path"][0] == clip_path
    assert batch["video_id"][0] == "video_001"
    assert batch["clip_id"][0] == "000000"


def test_create_audio_only_dataloader_skips_frames(tmp_path: Path) -> None:
    clip_path = tmp_path / "preprocessed" / "real" / "video_001" / "clips" / "000000"
    create_clip(clip_path, num_frames=3)

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path)

    dataloader = create_dataloader(
        manifest_path,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        include_frames=False,
    )
    batch = next(iter(dataloader))

    assert "frames" not in batch
    assert isinstance(batch["audio"], torch.Tensor)
    assert batch["audio"].shape == (1, 1, 48000)
    assert batch["label"].shape == (1,)


def test_create_dataloader_can_resize_frames(tmp_path: Path) -> None:
    clip_path = tmp_path / "preprocessed" / "real" / "video_001" / "clips" / "000000"
    create_clip(clip_path, num_frames=3)

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path)

    from src.data.dataset import build_frame_resize_transform

    dataloader = create_dataloader(
        manifest_path,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        frame_transform=build_frame_resize_transform(4),
    )
    batch = next(iter(dataloader))

    assert batch["frames"].shape == (1, 3, 3, 4, 4)


def test_create_dataloader_raises_for_missing_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"

    with pytest.raises(FileNotFoundError, match="Manifest file does not exist"):
        create_dataloader(manifest_path)

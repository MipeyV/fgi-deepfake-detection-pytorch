from pathlib import Path

import torch
from tests.data.helpers import create_clip, write_manifest

from src.data.video import build_video_input_pipeline


def test_resize_square_pipeline_builds_expected_frame_batch(
    tmp_path: Path,
) -> None:
    clip_path = tmp_path / "real" / "video" / "clips" / "000000"
    create_clip(clip_path, num_frames=3)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path)

    pipeline = build_video_input_pipeline(
        {
            "preprocessing": {
                "name": "resize_square",
                "frame_size": 6,
            }
        }
    )
    batch = next(
        iter(
            pipeline.create_dataloader(
                manifest_path=manifest_path,
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        )
    )

    assert batch["frames"].shape == (1, 3, 3, 6, 6)


def test_resize_normalize_pipeline_uses_configured_rgb_statistics(
    tmp_path: Path,
) -> None:
    clip_path = tmp_path / "real" / "video" / "clips" / "000000"
    create_clip(clip_path, num_frames=3)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, clip_path)

    pipeline = build_video_input_pipeline(
        {
            "preprocessing": {
                "name": "resize_normalize",
                "frame_size": 6,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            }
        }
    )
    batch = next(
        iter(
            pipeline.create_dataloader(
                manifest_path=manifest_path,
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        )
    )

    assert batch["frames"].shape == (1, 3, 3, 6, 6)
    assert torch.all(batch["frames"] >= -1)
    assert torch.all(batch["frames"] <= 1)


def test_video_pipeline_rejects_unknown_preprocessing() -> None:
    try:
        build_video_input_pipeline({"preprocessing": {"name": "unknown"}})
    except ValueError as error:
        assert "Unsupported video preprocessing" in str(error)
    else:
        raise AssertionError("Expected unsupported preprocessing to fail")

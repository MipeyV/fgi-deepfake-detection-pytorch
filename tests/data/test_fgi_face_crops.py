from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data.fgi_face_crops import (
    FaceBox,
    create_fgi_face_crop_dataset,
    crop_face_clip,
    select_longest_face_track,
    stabilize_face_box,
    write_face_crop_contact_sheet,
)
from tests.data.helpers import create_clip, write_manifest


def fixed_detector(image: Image.Image) -> list[FaceBox]:
    """Return a deterministic centered face box for fixture images."""
    return [FaceBox(x=2, y=1, width=4, height=5)]


def test_stabilize_face_box_returns_clamped_square_crop() -> None:
    crop = stabilize_face_box(
        [
            FaceBox(x=1, y=2, width=4, height=4),
            FaceBox(x=2, y=1, width=4, height=4),
        ],
        image_size=(8, 8),
        margin=0.25,
    )

    left, top, right, bottom = crop
    assert right - left == bottom - top
    assert 0 <= left < right <= 8
    assert 0 <= top < bottom <= 8


def test_select_longest_face_track_ignores_one_frame_distractor() -> None:
    tracked_face = FaceBox(x=1, y=1, width=3, height=3)
    distractor = FaceBox(x=4, y=4, width=4, height=4)

    track = select_longest_face_track(
        [
            [tracked_face],
            [tracked_face, distractor],
            [tracked_face],
        ],
        iou_threshold=0.5,
    )

    assert track == [tracked_face, tracked_face, tracked_face]


def test_crop_face_clip_writes_resized_frames_and_audio(tmp_path: Path) -> None:
    source_clip = tmp_path / "source"
    create_clip(source_clip, num_frames=3)
    destination_clip = tmp_path / "destination"

    crop_face_clip(
        source_clip=source_clip,
        destination_clip=destination_clip,
        detector=fixed_detector,
        output_size=6,
    )

    frame_paths = sorted(destination_clip.glob("*.jpg"))
    assert len(frame_paths) == 3
    with Image.open(frame_paths[0]) as image:
        assert image.size == (6, 6)
    assert (destination_clip / "audio.wav").is_file()


def test_create_fgi_face_crop_dataset_preserves_manifest_columns(
    tmp_path: Path,
) -> None:
    source_clip = tmp_path / "source" / "real" / "video_001" / "clips" / "000000"
    create_clip(source_clip, num_frames=3)
    source_manifest = tmp_path / "train_manifest.csv"
    write_manifest(source_manifest, source_clip)
    samples = pd.read_csv(
        source_manifest,
        dtype={"clip_id": str, "video_id": str},
    )
    samples["video_id_hashmod10"] = 2
    samples["split"] = "train"
    samples.to_csv(source_manifest, index=False)

    output_manifest = tmp_path / "manifests" / "train_manifest.csv"
    result = create_fgi_face_crop_dataset(
        manifest_path=source_manifest,
        output_dir=tmp_path / "processed_fgi",
        output_manifest_path=output_manifest,
        detector=fixed_detector,
        output_size=6,
    )

    output_samples = pd.read_csv(output_manifest, dtype={"clip_id": str})
    assert result.processed_clips == 1
    assert result.skipped_clips == 0
    assert list(output_samples.columns) == list(samples.columns)
    assert output_samples.loc[0, "split"] == "train"
    assert output_samples.loc[0, "clip_id"] == "000000"
    assert Path(output_samples.loc[0, "clip_path"]).is_dir()


def test_create_fgi_face_crop_dataset_can_skip_missing_faces(
    tmp_path: Path,
) -> None:
    source_clip = tmp_path / "source" / "real" / "video_001" / "clips" / "000000"
    create_clip(source_clip, num_frames=3)
    source_manifest = tmp_path / "manifest.csv"
    write_manifest(source_manifest, source_clip)

    result = create_fgi_face_crop_dataset(
        manifest_path=source_manifest,
        output_dir=tmp_path / "processed_fgi",
        output_manifest_path=tmp_path / "fgi_manifest.csv",
        detector=lambda image: [],
        missing_face_policy="skip",
    )

    assert result.processed_clips == 0
    assert result.skipped_clips == 1


def test_crop_face_clip_rejects_insufficient_face_detections(
    tmp_path: Path,
) -> None:
    source_clip = tmp_path / "source"
    create_clip(source_clip, num_frames=3)

    with pytest.raises(ValueError, match="Insufficient face detections"):
        crop_face_clip(
            source_clip=source_clip,
            destination_clip=tmp_path / "destination",
            detector=lambda image: [],
        )


def test_write_face_crop_contact_sheet_creates_png(tmp_path: Path) -> None:
    source_clip = tmp_path / "source" / "real" / "video_001" / "clips" / "000000"
    create_clip(source_clip, num_frames=3)
    source_manifest = tmp_path / "manifest.csv"
    write_manifest(source_manifest, source_clip)
    cropped_manifest = tmp_path / "fgi_manifest.csv"
    create_fgi_face_crop_dataset(
        manifest_path=source_manifest,
        output_dir=tmp_path / "processed_fgi",
        output_manifest_path=cropped_manifest,
        detector=fixed_detector,
        output_size=6,
    )

    output_path = write_face_crop_contact_sheet(
        manifest_path=cropped_manifest,
        output_path=tmp_path / "plots" / "contact_sheet.png",
        thumbnail_size=16,
    )

    assert output_path.is_file()
    with Image.open(output_path) as image:
        assert image.format == "PNG"

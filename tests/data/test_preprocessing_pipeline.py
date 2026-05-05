from pathlib import Path

import pytest

from src.data.preprocessing_pipeline import discover_videos, write_manifest


def test_discover_videos_finds_real_and_fake_videos(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()

    (real_dir / "real_video.mp4").touch()
    (real_dir / "notes.txt").touch()
    (fake_dir / "fake_video.avi").touch()

    videos = discover_videos(real_dir, fake_dir)

    assert len(videos) == 2
    assert videos[0].label == "real"
    assert videos[0].video_id == "real_video"
    assert videos[1].label == "fake"
    assert videos[1].video_id == "fake_video"


def test_discover_videos_raises_for_missing_directory(tmp_path: Path) -> None:
    real_dir = tmp_path / "missing_real"
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        discover_videos(real_dir, fake_dir)


def test_write_manifest_creates_csv(tmp_path: Path) -> None:
    clip_dir = tmp_path / "preprocessed" / "real" / "video_001" / "clips" / "000000"
    clip_dir.mkdir(parents=True)

    manifest_path = tmp_path / "manifest.csv"

    write_manifest([clip_dir], manifest_path)

    content = manifest_path.read_text(encoding="utf-8")

    assert "clip_path,label,video_id,clip_id" in content
    assert "real" in content
    assert "video_001" in content
    assert "000000" in content
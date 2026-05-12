from pathlib import Path

import pytest

from src.data.preprocessing_pipeline import (
    VideoItem,
    cleanup_frames_dir,
    discover_videos,
    preprocess_video,
    write_manifest,
)


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


def test_cleanup_frames_dir_removes_extracted_frames(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "000001.jpg").touch()

    cleanup_frames_dir(frames_dir)

    assert not frames_dir.exists()


def test_preprocess_video_removes_temporary_frames_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.touch()
    output_dir = tmp_path / "preprocessed"
    video_item = VideoItem(path=input_video, label="real", video_id="video_001")

    def fake_normalize_video(input_path: Path, output_path: Path, fps: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()

    def fake_extract_frames(video_path: Path, frames_dir: Path) -> None:
        frames_dir.mkdir(parents=True)
        for index in range(2):
            (frames_dir / f"{index + 1:06d}.jpg").write_text("frame", encoding="utf-8")

    def fake_extract_audio(video_path: Path, audio_path: Path, sample_rate: int) -> None:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.touch()

    def fake_create_audio_clip(
        audio_path: Path,
        clip_path: Path,
        start_time: float,
        clip_size: int,
        fps: int,
        sample_rate: int,
    ) -> None:
        (clip_path / "audio.wav").touch()

    monkeypatch.setattr(
        "src.data.preprocessing_pipeline.normalize_video", fake_normalize_video
    )
    monkeypatch.setattr(
        "src.data.preprocessing_pipeline.extract_frames", fake_extract_frames
    )
    monkeypatch.setattr(
        "src.data.preprocessing_pipeline.extract_audio", fake_extract_audio
    )
    monkeypatch.setattr(
        "src.data.preprocessing_pipeline.create_audio_clip", fake_create_audio_clip
    )

    clip_paths = preprocess_video(
        video_item=video_item,
        output_dir=output_dir,
        fps=30,
        clip_size=2,
        sample_rate=48000,
    )

    frames_dir = output_dir / "real" / "video_001" / "frames"

    assert len(clip_paths) == 1
    assert not frames_dir.exists()
    assert (clip_paths[0] / "000001.jpg").is_file()
    assert (clip_paths[0] / "000002.jpg").is_file()
    assert (clip_paths[0] / "audio.wav").is_file()

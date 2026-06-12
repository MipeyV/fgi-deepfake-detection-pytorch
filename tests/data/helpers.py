import numpy as np
import wave
from PIL import Image
from pathlib import Path


def create_clip(
    clip_path: Path,
    num_frames: int = 3,
    sample_rate: int = 48000,
    audio_samples: int = 48000,
    varying_audio: bool = False,
) -> None:
    """Create a synthetic synchronized frame/audio clip for tests."""
    clip_path.mkdir(parents=True)

    for index in range(num_frames):
        image = Image.new("RGB", (8, 8), color=(index * 30, 10, 20))
        image.save(clip_path / f"{index + 1:06d}.jpg")

    audio_path = clip_path / "audio.wav"
    if varying_audio:
        samples = np.linspace(
            -16000,
            16000,
            audio_samples,
            dtype=np.int16,
        )
    else:
        samples = np.zeros(audio_samples, dtype=np.int16)

    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(samples.tobytes())


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


def write_split_test_manifest(manifest_path: Path) -> None:
    manifest_path.write_text(
        "\n".join(
            [
                "clip_path,label,video_id,clip_id",
                "clips/video_a/000000,real,video_a,000000",
                "clips/video_a/000001,real,video_a,000001",
                "clips/video_b/000000,fake,video_b,000000",
                "clips/video_c/000000,real,video_c,000000",
            ]
        ),
        encoding="utf-8",
    )

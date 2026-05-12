from pathlib import Path
from dataclasses import dataclass

import subprocess
import shutil
import csv

# === GLOBAL CONSTANTS ===
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# === DATA CLASSES ===
@dataclass(frozen=True)
class VideoItem:
    path: Path
    label: str
    video_id: str


# === UTILITY FUNCTIONS ===
def discover_videos(real_dir: Path, fake_dir: Path) -> list[VideoItem]:
    """Discover video files in the given directories and create a list of VideoItem instances.
    
    Args:
        real_dir (Path): Path to the directory containing real videos.
        fake_dir (Path): Path to the directory containing fake videos.
    
    Returns:
        list[VideoItem]: A list of VideoItem instances representing the discovered videos.
    
    Raises:
        FileNotFoundError: If either of the provided directories does not exist or is not a directory.
    """
    if not real_dir.is_dir():
        raise FileNotFoundError(f"Real directory '{real_dir}' does not exist or is not a directory.")
    if not fake_dir.is_dir():
        raise FileNotFoundError(f"Fake directory '{fake_dir}' does not exist or is not a directory.")

    video_items: list[VideoItem] = []

    for video_path in sorted(real_dir.rglob("*")):
        if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
            video_items.append(
                VideoItem(
                    path=video_path,
                    label="real", 
                    video_id=video_path.stem
                )
            )

    for video_path in sorted(fake_dir.rglob("*")):
        if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
            video_items.append(
                VideoItem(
                    path=video_path,
                    label="fake", 
                    video_id=video_path.stem
                )
            )

    return video_items


def run_command(command: list[str]) -> None:
    """Run a shell command and raise an error if it fails.
    
    Args:
        command (list[str]): The command to run, as a list of strings.

    Raises:
        RuntimeError: If the command fails (i.e., returns a non-zero exit code).
    """
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with errors:\n"
            f"Exit code: {result.returncode}\n"
            f"Command: {' '.join(command)}\n"
            f"Errors: {result.stderr}"
        )


def check_ffmpeg_available() -> None:
    """Check if ffmpeg is available in the system PATH.
    
    Raises:
        RuntimeError: If ffmpeg is not found in the system PATH.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise RuntimeError(
            "ffmpeg is not available in the system PATH. Please install ffmpeg to proceed."
        ) from error


def normalize_video(input_path: Path, output_path: Path, fps: int = 30) -> None:
    """Normalize a video by converting it to a standard format and frame rate using ffmpeg.
    
    Args:
        input_path (Path): The path to the input video file.
        output_path (Path): The path where the normalized video will be saved.
        fps (int): The frame rate for the normalized video.
    
    Raises:
        FileNotFoundError: If the input video file does not exist.
        FileExistsError: If the output video file already exists.
        RuntimeError: If the ffmpeg command fails.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video file '{input_path}' does not exist.")
    
    if output_path.exists():
        raise FileExistsError(
            f"Output video file '{output_path}' already exists. " 
            "Please choose a different path or remove the existing file."
        )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-qscale:v",
        "2",
        "-async",
        "1",
        "-r",
        str(fps),
        str(output_path),
    ]

    run_command(command)


def extract_frames(video_path: Path, frames_dir: Path) -> None:
    """Extract frames from a video using ffmpeg and save them as JPEG images.
    
    Args:
        video_path (Path): The path to the input video file.
        frames_dir (Path): The directory where the extracted frames will be saved.
    
    Raises:
        FileNotFoundError: If the input video file does not exist.
        FileExistsError: If the frames directory already exists and is not empty.
        RuntimeError: If the ffmpeg command fails.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video file '{video_path}' does not exist.")
    
    if frames_dir.exists() and any(frames_dir.iterdir()):
        raise FileExistsError(
            f"Frames directory '{frames_dir}' already exists and is not empty. " 
            "Please choose a different path or remove the existing directory."
        )
    
    frames_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-qscale:v",
        "2",
        "-threads",
        "1",
        "-f",
        "image2",
        str(frames_dir / "%06d.jpg"),
    ]

    run_command(command)


def extract_audio(video_path: Path, audio_path: Path, sample_rate: int = 48000) -> None:
    """Extract audio from a video using ffmpeg and save it as a WAV file.
    
    Args:
        video_path (Path): The path to the input video file.
        audio_path (Path): The path where the extracted audio will be saved.
        sample_rate (int): The sample rate for the extracted audio.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        FileExistsError: If the output audio file already exists.
        RuntimeError: If the ffmpeg command fails.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video file '{video_path}' does not exist.")
    
    if audio_path.exists():
        raise FileExistsError(
            f"Output audio file '{audio_path}' already exists. " 
            "Please choose a different path or remove the existing file."
        )
    
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        str(audio_path),
    ]

    run_command(command)


def create_frame_clips(frame_dir: Path, clips_dir: Path, clip_size: int = 30) -> list[Path]:
    """Create clips of frames from a directory of extracted frames.
    
    Args:
        frame_dir (Path): The directory containing the extracted frames.
        clips_dir (Path): The directory where the created clips will be saved.
        clip_size (int): The number of frames per clip.

    Returns:
        list[Path]: A list of paths to the created clip files.

    Raises:
        FileNotFoundError: If the frame directory does not exist or is empty or if the frame directory does not contain any JPEG images.
        FileExistsError: If the clips directory already exists and is not empty.
    """
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")

    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"Frame directory is empty: {frame_dir}")

    if clips_dir.exists() and any(clips_dir.iterdir()):
        raise FileExistsError(
            f"Clips directory '{clips_dir}' already exists and is not empty. "
            "Please choose a different path or remove the existing directory."
        )

    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[Path] = []

    for start_index in range(0, len(frame_paths), clip_size):
        clip_frames = frame_paths[start_index:start_index + clip_size]

        if len(clip_frames) < clip_size:
            break

        clip_index = start_index // clip_size
        clip_dir = clips_dir / f"{clip_index:06d}"
        clip_dir.mkdir(parents=True, exist_ok=False)

        for frame_path in clip_frames:
            destination_path = clip_dir / frame_path.name
            shutil.copy2(frame_path, destination_path)

        clip_paths.append(clip_dir)

    return clip_paths


def create_audio_clip(
    audio_path: Path,
    clip_path: Path,
    start_time: float,
    clip_size: int = 30,
    fps: int = 30,
    sample_rate: int = 48000,
) -> None:
    """Create an audio clip from a WAV file corresponding to a video clip.
    
    Args:
        audio_path (Path): The path to the input audio WAV file.
        clip_path (Path): The path to the directory of the corresponding video clip frames.
        start_time (float): The start time in seconds for the audio clip.
        clip_size (int): The number of frames in the video clip (default is 30).
        fps (int): The frame rate of the video (default is 30).
        sample_rate (int): The sample rate for the output audio clip (default is 48000).

    Raises:
        FileNotFoundError: If the input audio file does not exist or if the clip directory does not exist.
        FileExistsError: If the output audio clip file already exists.
        RuntimeError: If the ffmpeg command fails.
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"Input audio file '{audio_path}' does not exist.")

    if not clip_path.is_dir():
        raise FileNotFoundError(f"Clip directory '{clip_path}' does not exist.")

    output_audio_path = clip_path / "audio.wav"
    if output_audio_path.exists():
        raise FileExistsError(
            f"Output audio clip file '{output_audio_path}' already exists. "
            "Please choose a different path or remove the existing file."
        )

    duration_seconds = clip_size / fps

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        str(audio_path),
        "-t",
        str(duration_seconds),
        "-ac",
        "1",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        str(output_audio_path),
    ]

    run_command(command)


def cleanup_frames_dir(frames_dir: Path) -> None:
    """Remove the temporary extracted frames directory after clips are created.

    Args:
        frames_dir (Path): The directory containing extracted frames.
    """
    if frames_dir.exists():
        shutil.rmtree(frames_dir)


def preprocess_video(
    video_item: VideoItem, 
    output_dir: Path,
    fps: int = 30,
    clip_size: int = 30,
    sample_rate: int = 48000,
) -> list[Path]:
    """Preprocess a video by normalizing it, extracting frames and audio, and creating clips.
    
    Args:
        video_item (VideoItem): The VideoItem instance representing the video to preprocess.
        output_dir (Path): The directory where the preprocessed data will be saved.
        fps (int): The frame rate for normalization and clip creation (default is 30).
        clip_size (int): The number of frames per clip (default is 30).
        sample_rate (int): The sample rate for audio extraction and clips (default is 48000).

    Returns:
        list[Path]: A list of paths to the created clip directories.

    Raises:
        RuntimeError: If any of the preprocessing steps fail.
    """
    video_output_dir = output_dir / video_item.label / video_item.video_id

    normalized_video_path = video_output_dir / "video.avi"
    frames_dir = video_output_dir / "frames"
    audio_path = video_output_dir / "audio.wav"
    clips_dir = video_output_dir / "clips"

    normalize_video(video_item.path, normalized_video_path, fps)
    extract_frames(normalized_video_path, frames_dir)
    extract_audio(normalized_video_path, audio_path, sample_rate)
    clip_paths = create_frame_clips(frames_dir, clips_dir, clip_size)
    cleanup_frames_dir(frames_dir)

    for index, clip_path in enumerate(clip_paths):
        start_time = index * clip_size / fps
        create_audio_clip(
            audio_path=audio_path,
            clip_path=clip_path,
            start_time=start_time,
            clip_size=clip_size,
            fps=fps,
            sample_rate=sample_rate,
        )

    return clip_paths


def preprocess_dataset(
    real_dir: Path, 
    fake_dir: Path, 
    output_dir: Path,
    fps: int = 30,
    clip_size: int = 30,
    sample_rate: int = 48000,
) -> list[Path]:
    """Preprocess a dataset of videos by discovering videos and applying preprocessing to each video.
    
    Args:
        real_dir (Path): The directory containing real videos.
        fake_dir (Path): The directory containing fake videos.
        output_dir (Path): The directory where the preprocessed dataset will be saved.
        fps (int): The frame rate for normalization and clip creation (default is 30).
        clip_size (int): The number of frames per clip (default is 30).
        sample_rate (int): The sample rate for audio extraction and clips (default is 48000).

    Returns:
        list[Path]: A list of paths to all the created clip directories for the entire dataset.
    
    Raises:
        RuntimeError: If any of the preprocessing steps fail for any video in the dataset.
    """
    check_ffmpeg_available()

    video_items = discover_videos(real_dir, fake_dir)

    all_clip_paths: list[Path] = []

    for video_item in video_items:
        clip_paths = preprocess_video(
            video_item=video_item,
            output_dir=output_dir,
            fps=fps,
            clip_size=clip_size,
            sample_rate=sample_rate,
        )
        all_clip_paths.extend(clip_paths)
    
    return all_clip_paths


def write_manifest(clip_paths: list[Path], manifest_path: Path) -> None:
    """Write a manifest file containing the paths and labels of the preprocessed clips.
    
    Args:
        clip_paths (list[Path]): A list of paths to the clip directories.
        manifest_path (Path): The path where the manifest file will be saved.

    Raises:
        FileExistsError: If the manifest file already exists.
        RuntimeError: If there is an error writing the manifest file.
    """
    if manifest_path.exists():
        raise FileExistsError(
            f"Manifest file '{manifest_path}' already exists. "
            "Please choose a different path or remove the existing file."
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=["clip_path", "label", "video_id", "clip_id"],
            )
            writer.writeheader()

            for clip_path in clip_paths:
                writer.writerow(
                    {
                        "clip_path": str(clip_path),
                        "label": clip_path.parents[2].name,
                        "video_id": clip_path.parents[1].name,
                        "clip_id": clip_path.name,
                    }
                )
    except Exception as error:
        raise RuntimeError(f"Failed to write manifest file: {error}") from error

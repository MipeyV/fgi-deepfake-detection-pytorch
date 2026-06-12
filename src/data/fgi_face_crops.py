"""Offline face-crop preprocessing for FGI-style audio-visual experiments."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class FaceBox:
    """Axis-aligned face bounding box in source-image pixels."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class FaceCropResult:
    """Summary of one manifest-level face-crop preprocessing run."""

    manifest_path: Path
    processed_clips: int
    skipped_clips: int


FaceDetector = Callable[[Image.Image], Sequence[FaceBox]]


class OpenCVHaarFaceDetector:
    """Fallback frontal-face detector backed by OpenCV's Haar cascade.

    Haar detection is dependency-free but less reliable than YuNet. It should
    mainly be used for development and explicit comparisons.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: int = 40,
        max_detection_size: int = 640,
    ) -> None:
        """Initialize and load the OpenCV frontal-face cascade.

        Args:
            scale_factor: Pyramid scale used by ``detectMultiScale``.
            min_neighbors: Minimum neighboring detections required by OpenCV.
            min_face_size: Minimum face width and height in source pixels.
            max_detection_size: Maximum image side used during detection.

        Raises:
            ValueError: If one of the detector parameters is invalid.
            RuntimeError: If OpenCV cannot load its bundled cascade.
        """
        if scale_factor <= 1:
            raise ValueError("scale_factor must be greater than 1")
        if min_neighbors < 0:
            raise ValueError("min_neighbors must be non-negative")
        if min_face_size <= 0 or max_detection_size <= 0:
            raise ValueError("face and detection sizes must be positive")

        cascade_path = (
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
        classifier = cv2.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            raise RuntimeError(f"Failed to load face cascade: {cascade_path}")

        self.classifier = classifier
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.max_detection_size = max_detection_size

    def __call__(self, image: Image.Image) -> list[FaceBox]:
        """Detect frontal faces in one RGB image.

        Args:
            image: Source RGB image.

        Returns:
            Detected boxes expressed in source-image coordinates.
        """
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        resize_scale = min(
            1.0,
            self.max_detection_size / max(height, width),
        )

        if resize_scale < 1:
            detection_width = max(1, round(width * resize_scale))
            detection_height = max(1, round(height * resize_scale))
            rgb = cv2.resize(
                rgb,
                (detection_width, detection_height),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        scaled_min_size = max(1, round(self.min_face_size * resize_scale))
        detections = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(scaled_min_size, scaled_min_size),
        )

        inverse_scale = 1 / resize_scale
        return [
            FaceBox(
                x=round(x * inverse_scale),
                y=round(y * inverse_scale),
                width=round(box_width * inverse_scale),
                height=round(box_height * inverse_scale),
            )
            for x, y, box_width, box_height in detections
        ]


class OpenCVYuNetFaceDetector:
    """Face detector backed by OpenCV's YuNet ONNX implementation."""

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_detection_size: int = 640,
    ) -> None:
        """Initialize a YuNet detector from an ONNX model.

        Args:
            model_path: Path to ``face_detection_yunet_2023mar.onnx`` or a
                compatible YuNet model.
            score_threshold: Minimum confidence retained by YuNet.
            nms_threshold: Non-maximum suppression overlap threshold.
            top_k: Maximum candidate count considered before suppression.
            max_detection_size: Maximum image side used during detection.

        Raises:
            FileNotFoundError: If the ONNX model does not exist.
            ValueError: If one of the detector parameters is invalid.
        """
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"YuNet model does not exist: {model_path}")
        if not 0 < score_threshold <= 1:
            raise ValueError("score_threshold must be in (0, 1]")
        if not 0 < nms_threshold <= 1:
            raise ValueError("nms_threshold must be in (0, 1]")
        if top_k <= 0 or max_detection_size <= 0:
            raise ValueError("top_k and max_detection_size must be positive")

        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )
        self.max_detection_size = max_detection_size

    def __call__(self, image: Image.Image) -> list[FaceBox]:
        """Detect faces in one RGB image using YuNet.

        Args:
            image: Source RGB image.

        Returns:
            Detected boxes expressed in source-image coordinates.
        """
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        resize_scale = min(
            1.0,
            self.max_detection_size / max(height, width),
        )
        if resize_scale < 1:
            detection_width = max(1, round(width * resize_scale))
            detection_height = max(1, round(height * resize_scale))
            rgb = cv2.resize(
                rgb,
                (detection_width, detection_height),
                interpolation=cv2.INTER_AREA,
            )

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, detections = self.detector.detect(bgr)
        if detections is None:
            return []

        inverse_scale = 1 / resize_scale
        return [
            FaceBox(
                x=round(row[0] * inverse_scale),
                y=round(row[1] * inverse_scale),
                width=round(row[2] * inverse_scale),
                height=round(row[3] * inverse_scale),
            )
            for row in detections
        ]


def select_largest_face(boxes: Sequence[FaceBox]) -> FaceBox | None:
    """Select the largest detected face by bounding-box area.

    Args:
        boxes: Face candidates detected in one frame.

    Returns:
        The largest box, or ``None`` when no face was detected.
    """
    if not boxes:
        return None
    return max(boxes, key=lambda box: box.width * box.height)


def face_box_iou(first: FaceBox, second: FaceBox) -> float:
    """Compute intersection over union for two face boxes.

    Args:
        first: First face box.
        second: Second face box.

    Returns:
        Intersection-over-union value in ``[0, 1]``.
    """
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = first.width * first.height
    second_area = second.width * second.height
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def select_longest_face_track(
    frame_detections: Sequence[Sequence[FaceBox]],
    iou_threshold: float = 0.3,
    max_frame_gap: int = 5,
) -> list[FaceBox]:
    """Associate detections over time and return the longest face track.

    This mirrors the IoU-based tracking principle used by the original FGI
    preprocessing while remaining lightweight and clip-local.

    Args:
        frame_detections: Detected face boxes for each frame in temporal order.
        iou_threshold: Minimum IoU required to extend an existing track.
        max_frame_gap: Maximum number of missing frames tolerated by a track.

    Returns:
        Face boxes belonging to the longest consistent track. Track area is
        used as a deterministic tie-breaker.

    Raises:
        ValueError: If tracking parameters are invalid.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    if max_frame_gap < 0:
        raise ValueError("max_frame_gap must be non-negative")

    tracks: list[list[tuple[int, FaceBox]]] = []
    for frame_index, detections in enumerate(frame_detections):
        updated_tracks: set[int] = set()
        for detection in sorted(
            detections,
            key=lambda box: box.width * box.height,
            reverse=True,
        ):
            best_track_index = None
            best_iou = iou_threshold
            for track_index, track in enumerate(tracks):
                if track_index in updated_tracks:
                    continue
                last_frame, last_box = track[-1]
                if frame_index - last_frame > max_frame_gap + 1:
                    continue
                overlap = face_box_iou(detection, last_box)
                if overlap >= best_iou:
                    best_iou = overlap
                    best_track_index = track_index

            if best_track_index is None:
                tracks.append([(frame_index, detection)])
                updated_tracks.add(len(tracks) - 1)
            else:
                tracks[best_track_index].append((frame_index, detection))
                updated_tracks.add(best_track_index)

    if not tracks:
        return []
    longest_track = max(
        tracks,
        key=lambda track: (
            len(track),
            sum(box.width * box.height for _, box in track),
        ),
    )
    return [box for _, box in longest_track]


def stabilize_face_box(
    boxes: Sequence[FaceBox],
    image_size: tuple[int, int],
    margin: float = 0.3,
) -> tuple[int, int, int, int]:
    """Aggregate frame detections into one stable square crop.

    Median centers and dimensions reduce frame-to-frame detector jitter. The
    configured margin is then added around the face before clamping the square
    crop to the source-image boundaries.

    Args:
        boxes: One selected face box per successfully detected frame.
        image_size: Source image size as ``(width, height)``.
        margin: Fractional padding added to the stabilized face size.

    Returns:
        PIL-compatible crop coordinates ``(left, top, right, bottom)``.

    Raises:
        ValueError: If no boxes are provided, dimensions are invalid, or the
            margin is negative.
    """
    if not boxes:
        raise ValueError("At least one face box is required")
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_size dimensions must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    center_x = float(np.median([box.x + box.width / 2 for box in boxes]))
    center_y = float(np.median([box.y + box.height / 2 for box in boxes]))
    face_size = float(
        max(
            np.median([box.width for box in boxes]),
            np.median([box.height for box in boxes]),
        )
    )
    crop_size = min(
        max(1, round(face_size * (1 + 2 * margin))),
        image_width,
        image_height,
    )

    left = round(center_x - crop_size / 2)
    top = round(center_y - crop_size / 2)
    left = min(max(0, left), image_width - crop_size)
    top = min(max(0, top), image_height - crop_size)
    return left, top, left + crop_size, top + crop_size


def crop_face_clip(
    source_clip: str | Path,
    destination_clip: str | Path,
    detector: FaceDetector,
    output_size: int = 256,
    margin: float = 0.3,
    min_detection_fraction: float = 0.5,
) -> Path:
    """Create one synchronized clip containing stable face crops and audio.

    Args:
        source_clip: Directory containing JPEG frames and ``audio.wav``.
        destination_clip: New clip directory to create.
        detector: Callable returning face boxes for one frame.
        output_size: Width and height of saved square crops.
        margin: Fractional padding around the stabilized face.
        min_detection_fraction: Minimum fraction of frames that must contain a
            detected face.

    Returns:
        The created destination clip path.

    Raises:
        FileNotFoundError: If frames or synchronized audio are missing.
        FileExistsError: If the destination already exists.
        ValueError: If frames have inconsistent sizes or no face is detected.
    """
    source_clip = Path(source_clip)
    destination_clip = Path(destination_clip)
    if output_size <= 0:
        raise ValueError("output_size must be greater than 0")
    if not 0 < min_detection_fraction <= 1:
        raise ValueError("min_detection_fraction must be in (0, 1]")
    if destination_clip.exists():
        raise FileExistsError(
            f"Destination clip already exists: {destination_clip}"
        )

    frame_paths = sorted(source_clip.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No JPEG frames found in clip: {source_clip}")
    audio_path = source_clip / "audio.wav"
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    images: list[Image.Image] = []
    frame_detections: list[Sequence[FaceBox]] = []
    image_size: tuple[int, int] | None = None
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            rgb_image = image.convert("RGB")
        if image_size is None:
            image_size = rgb_image.size
        elif rgb_image.size != image_size:
            raise ValueError(f"Inconsistent frame size in clip: {source_clip}")

        images.append(rgb_image)
        frame_detections.append(detector(rgb_image))

    selected_boxes = select_longest_face_track(frame_detections)
    detection_fraction = len(selected_boxes) / len(frame_paths)
    if image_size is None or detection_fraction < min_detection_fraction:
        raise ValueError(
            "Insufficient face detections in clip "
            f"({len(selected_boxes)}/{len(frame_paths)}): {source_clip}"
        )

    crop_box = stabilize_face_box(selected_boxes, image_size, margin)
    destination_clip.mkdir(parents=True)
    for frame_path, image in zip(frame_paths, images, strict=True):
        face_crop = image.crop(crop_box).resize(
            (output_size, output_size),
            Image.Resampling.LANCZOS,
        )
        face_crop.save(destination_clip / frame_path.name, quality=95)

    shutil.copy2(audio_path, destination_clip / "audio.wav")
    return destination_clip


def create_fgi_face_crop_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    output_manifest_path: str | Path,
    detector: FaceDetector,
    output_size: int = 256,
    margin: float = 0.3,
    min_detection_fraction: float = 0.5,
    missing_face_policy: str = "error",
) -> FaceCropResult:
    """Create stable face crops for every clip referenced by a manifest.

    Existing manifest metadata, including split columns, is preserved while
    ``clip_path`` is rewritten to the new FGI cache.

    Args:
        manifest_path: Source clip manifest.
        output_dir: Root directory for cropped clips.
        output_manifest_path: Destination CSV manifest.
        detector: Face detector used for every source frame.
        output_size: Width and height of saved square crops.
        margin: Fractional padding around the stabilized face.
        min_detection_fraction: Minimum fraction of frames with detections.
        missing_face_policy: ``error`` to stop or ``skip`` to omit clips where
            no face is detected.

    Returns:
        Processing counts and the written manifest path.

    Raises:
        FileNotFoundError: If the source manifest does not exist.
        FileExistsError: If the destination manifest already exists.
        ValueError: If the manifest or missing-face policy is invalid.
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_manifest_path = Path(output_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
    if output_manifest_path.exists():
        raise FileExistsError(
            f"Output manifest already exists: {output_manifest_path}"
        )
    if missing_face_policy not in {"error", "skip"}:
        raise ValueError("missing_face_policy must be 'error' or 'skip'")

    samples = pd.read_csv(
        manifest_path,
        dtype={
            "clip_path": str,
            "label": str,
            "video_id": str,
            "clip_id": str,
        },
    )
    required_columns = {"clip_path", "label", "video_id", "clip_id"}
    missing_columns = required_columns - set(samples.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing_columns)}"
        )

    processed_rows: list[dict] = []
    skipped_clips = 0
    for row in samples.to_dict(orient="records"):
        destination_clip = (
            output_dir
            / row["label"]
            / row["video_id"]
            / "clips"
            / row["clip_id"]
        )
        try:
            crop_face_clip(
                source_clip=row["clip_path"],
                destination_clip=destination_clip,
                detector=detector,
                output_size=output_size,
                margin=margin,
                min_detection_fraction=min_detection_fraction,
            )
        except ValueError as error:
            if (
                missing_face_policy == "skip"
                and "Insufficient face detections" in str(error)
            ):
                skipped_clips += 1
                continue
            raise

        processed_row = dict(row)
        processed_row["clip_path"] = str(destination_clip)
        processed_rows.append(processed_row)

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(processed_rows, columns=samples.columns).to_csv(
        output_manifest_path,
        index=False,
        encoding="utf-8",
    )
    return FaceCropResult(
        manifest_path=output_manifest_path,
        processed_clips=len(processed_rows),
        skipped_clips=skipped_clips,
    )


def write_face_crop_contact_sheet(
    manifest_path: str | Path,
    output_path: str | Path,
    max_clips: int = 16,
    columns: int = 4,
    thumbnail_size: int = 192,
) -> Path:
    """Write a labeled contact sheet from the first frame of cropped clips.

    Args:
        manifest_path: Manifest produced by ``create_fgi_face_crop_dataset``.
        output_path: Destination PNG path.
        max_clips: Maximum number of clips represented.
        columns: Number of thumbnails per row.
        thumbnail_size: Square thumbnail width and height.

    Returns:
        The written contact-sheet path.

    Raises:
        FileNotFoundError: If the manifest or a sampled frame is missing.
        ValueError: If layout parameters are invalid or the manifest is empty.
    """
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
    if max_clips <= 0 or columns <= 0 or thumbnail_size <= 0:
        raise ValueError("Contact-sheet layout values must be positive")

    samples = pd.read_csv(
        manifest_path,
        dtype={
            "clip_path": str,
            "label": str,
            "video_id": str,
            "clip_id": str,
        },
    ).head(max_clips)
    if samples.empty:
        raise ValueError("Cannot create a contact sheet from an empty manifest")

    label_height = 28
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumbnail_size, rows * (thumbnail_size + label_height)),
        color="white",
    )
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(samples.to_dict(orient="records")):
        frame_paths = sorted(Path(row["clip_path"]).glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(
                f"No JPEG frames found in clip: {row['clip_path']}"
            )
        sampled_indices = [0, len(frame_paths) // 2, len(frame_paths) - 1]
        panel_width = thumbnail_size // 3
        thumbnail = Image.new(
            "RGB",
            (thumbnail_size, thumbnail_size),
            color="white",
        )
        for panel_index, frame_index in enumerate(sampled_indices):
            with Image.open(frame_paths[frame_index]) as image:
                panel = image.convert("RGB").resize(
                    (panel_width, panel_width),
                    Image.Resampling.LANCZOS,
                )
            panel_top = (thumbnail_size - panel_width) // 2
            thumbnail.paste(
                panel,
                (panel_index * panel_width, panel_top),
            )

        column = index % columns
        sheet_row = index // columns
        left = column * thumbnail_size
        top = sheet_row * (thumbnail_size + label_height)
        sheet.paste(thumbnail, (left, top))
        draw.text(
            (left + 4, top + thumbnail_size + 4),
            f"{row['label']} {row['video_id']}/{row['clip_id']}",
            fill="black",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path

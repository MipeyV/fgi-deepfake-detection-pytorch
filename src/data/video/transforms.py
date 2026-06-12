"""Frame-level transforms for video input pipelines."""

from collections.abc import Callable

from PIL import Image
from torch import Tensor, tensor

from src.data.dataset import pil_to_tensor


FrameTransform = Callable[[Image.Image], Tensor]


def build_resize_square_transform(frame_size: int) -> FrameTransform:
    """Resize each frame directly to a square."""
    if frame_size <= 0:
        raise ValueError("frame_size must be greater than 0")

    def transform(image: Image.Image) -> Tensor:
        resized = image.resize((frame_size, frame_size), Image.BILINEAR)
        return pil_to_tensor(resized)

    return transform


def build_resize_normalize_transform(
    frame_size: int,
    mean: tuple[float, float, float] | list[float],
    std: tuple[float, float, float] | list[float],
) -> FrameTransform:
    """Resize each frame to a square and normalize its RGB channels."""
    if frame_size <= 0:
        raise ValueError("frame_size must be greater than 0")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must contain three RGB values")
    if any(value <= 0 for value in std):
        raise ValueError("std values must be greater than 0")

    channel_mean = tensor(mean).view(3, 1, 1)
    channel_std = tensor(std).view(3, 1, 1)

    def transform(image: Image.Image) -> Tensor:
        resized = image.resize((frame_size, frame_size), Image.BILINEAR)
        return (pil_to_tensor(resized) - channel_mean) / channel_std

    return transform


def build_resize_center_crop_transform(
    resize_size: tuple[int, int] | list[int],
    crop_size: int,
) -> FrameTransform:
    """Resize to ``[height, width]`` and take a centered square crop."""
    if len(resize_size) != 2 or any(size <= 0 for size in resize_size):
        raise ValueError("resize_size must contain two positive values")
    if crop_size <= 0:
        raise ValueError("crop_size must be greater than 0")

    resize_height, resize_width = resize_size
    if crop_size > min(resize_height, resize_width):
        raise ValueError("crop_size cannot exceed resize_size")

    def transform(image: Image.Image) -> Tensor:
        resized = image.resize((resize_width, resize_height), Image.BILINEAR)
        left = (resize_width - crop_size) // 2
        top = (resize_height - crop_size) // 2
        cropped = resized.crop((left, top, left + crop_size, top + crop_size))
        return pil_to_tensor(cropped)

    return transform

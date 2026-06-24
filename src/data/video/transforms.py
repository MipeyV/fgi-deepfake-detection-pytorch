"""Frame-level transforms for video input pipelines."""

from collections.abc import Callable

from PIL import Image
from torch import Tensor, tensor

from src.data.dataset import pil_to_tensor

FrameTransform = Callable[[Image.Image], Tensor]


def build_resize_square_transform(frame_size: int) -> FrameTransform:
    """Build a transform that resizes an RGB frame directly to a square.

    Args:
        frame_size: Height and width of the output frame in pixels.

    Returns:
        A callable converting one PIL image to a float tensor in ``[0, 1]``
        with shape ``[channels, frame_size, frame_size]``.

    Raises:
        ValueError: If ``frame_size`` is not greater than zero.
    """
    if frame_size <= 0:
        raise ValueError("frame_size must be greater than 0")

    def transform(image: Image.Image) -> Tensor:
        """Resize one PIL image and convert it to a channel-first tensor."""
        resized = image.resize((frame_size, frame_size), Image.BILINEAR)
        return pil_to_tensor(resized)

    return transform


def build_resize_normalize_transform(
    frame_size: int,
    mean: tuple[float, float, float] | list[float],
    std: tuple[float, float, float] | list[float],
) -> FrameTransform:
    """Build a square resize followed by channel-wise RGB normalization.

    Args:
        frame_size: Height and width of the output frame in pixels.
        mean: Three RGB means subtracted after conversion to ``[0, 1]``.
        std: Three positive RGB standard deviations used for scaling.

    Returns:
        A callable converting one PIL image to a normalized float tensor with
        shape ``[3, frame_size, frame_size]``.

    Raises:
        ValueError: If the size is invalid, if ``mean`` or ``std`` does not
            contain three values, or if one standard deviation is not positive.
    """
    if frame_size <= 0:
        raise ValueError("frame_size must be greater than 0")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must contain three RGB values")
    if any(value <= 0 for value in std):
        raise ValueError("std values must be greater than 0")

    channel_mean = tensor(mean).view(3, 1, 1)
    channel_std = tensor(std).view(3, 1, 1)

    def transform(image: Image.Image) -> Tensor:
        """Resize and normalize one RGB PIL image."""
        resized = image.resize((frame_size, frame_size), Image.BILINEAR)
        return (pil_to_tensor(resized) - channel_mean) / channel_std

    return transform


def build_resize_center_crop_transform(
    resize_size: tuple[int, int] | list[int],
    crop_size: int,
) -> FrameTransform:
    """Build a resize followed by a centered square crop.

    Args:
        resize_size: Intermediate ``[height, width]`` in pixels.
        crop_size: Height and width of the final square crop.

    Returns:
        A callable converting one PIL image to a float tensor in ``[0, 1]``
        with shape ``[channels, crop_size, crop_size]``.

    Raises:
        ValueError: If a dimension is not positive or if the crop is larger
            than the resized frame.
    """
    if len(resize_size) != 2 or any(size <= 0 for size in resize_size):
        raise ValueError("resize_size must contain two positive values")
    if crop_size <= 0:
        raise ValueError("crop_size must be greater than 0")

    resize_height, resize_width = resize_size
    if crop_size > min(resize_height, resize_width):
        raise ValueError("crop_size cannot exceed resize_size")

    def transform(image: Image.Image) -> Tensor:
        """Resize, center-crop, and tensorize one PIL image."""
        resized = image.resize((resize_width, resize_height), Image.BILINEAR)
        left = (resize_width - crop_size) // 2
        top = (resize_height - crop_size) // 2
        cropped = resized.crop((left, top, left + crop_size, top + crop_size))
        return pil_to_tensor(cropped)

    return transform

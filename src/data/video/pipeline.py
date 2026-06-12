"""Video input pipeline construction from experiment configuration."""

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataloader import create_dataloader
from src.data.video.transforms import (
    FrameTransform,
    build_resize_center_crop_transform,
    build_resize_normalize_transform,
    build_resize_square_transform,
)


@dataclass(frozen=True)
class VideoInputPipeline:
    """Frame preprocessing and dataloader construction for one experiment."""

    frame_transform: FrameTransform

    def create_dataloader(
        self,
        manifest_path: str | Path,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:
        return create_dataloader(
            manifest_path=manifest_path,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            frame_transform=self.frame_transform,
            include_frames=True,
        )


def build_video_input_pipeline(video_config: dict) -> VideoInputPipeline:
    """Build a video input pipeline independently from the model architecture."""
    preprocessing_config = video_config["preprocessing"]
    preprocessing_name = preprocessing_config["name"]

    if preprocessing_name == "resize_square":
        frame_transform = build_resize_square_transform(
            preprocessing_config["frame_size"]
        )
    elif preprocessing_name == "resize_normalize":
        frame_transform = build_resize_normalize_transform(
            frame_size=preprocessing_config["frame_size"],
            mean=preprocessing_config["mean"],
            std=preprocessing_config["std"],
        )
    elif preprocessing_name == "resize_center_crop":
        frame_transform = build_resize_center_crop_transform(
            resize_size=preprocessing_config["resize_size"],
            crop_size=preprocessing_config["crop_size"],
        )
    else:
        raise ValueError(
            f"Unsupported video preprocessing: {preprocessing_name}"
        )

    return VideoInputPipeline(frame_transform=frame_transform)

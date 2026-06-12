"""Configuration-driven construction of the FGI multimodal input pipeline."""

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.fgi_multimodal import create_fgi_multimodal_dataloader
from src.data.video.transforms import build_resize_normalize_transform


@dataclass(frozen=True)
class FGIInputPipeline:
    """Build strict synchronized FGI dataloaders from one configuration."""

    config: dict

    def create_dataloader(
        self,
        manifest_path: str | Path,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:
        """Create one FGI dataloader for a configured dataset split."""
        video_config = self.config["video"]
        audio_config = self.config["audio"]
        preprocessing = video_config["preprocessing"]
        frame_transform = build_resize_normalize_transform(
            frame_size=preprocessing["frame_size"],
            mean=preprocessing["mean"],
            std=preprocessing["std"],
        )
        return create_fgi_multimodal_dataloader(
            manifest_path=manifest_path,
            frame_transform=frame_transform,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            expected_frames=video_config["expected_frames"],
            sample_rate=audio_config["sample_rate"],
            audio_samples=audio_config["num_samples"],
            audio_normalization=audio_config["normalization"],
        )


def build_fgi_input_pipeline(config: dict) -> FGIInputPipeline:
    """Build a configuration-driven synchronized FGI input pipeline."""
    return FGIInputPipeline(config=config)

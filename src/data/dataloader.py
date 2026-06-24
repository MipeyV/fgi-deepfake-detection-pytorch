from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DeepFakeClipDataset


def collate_deepfake_batch(batch: list[dict]) -> dict:
    """Collate deepfake samples into a batch.

    Tensor fields are stacked, while metadata fields are kept as lists.

    Args:
        batch: Samples where each dictionary contains:
            - "frames": Tensor of shape [num_frames, C, H, W]
            - "audio": Tensor of shape [1, num_samples]
            - "label": Tensor scalar (0 for real, 1 for fake)
            - "clip_path": Path to the clip directory
            - "video_id": ID of the source video
            - "clip_id": ID of the clip within the video

    Returns:
        Dictionary containing the collated batch with the same keys as the input
        samples.
    """
    collated_batch = {
        "audio": torch.stack([sample["audio"] for sample in batch]),
        "label": torch.stack([sample["label"] for sample in batch]),
        "clip_path": [sample["clip_path"] for sample in batch],
        "video_id": [sample["video_id"] for sample in batch],
        "clip_id": [sample["clip_id"] for sample in batch],
    }

    if "frames" in batch[0]:
        collated_batch["frames"] = torch.stack([sample["frames"] for sample in batch])

    return collated_batch


def create_dataloader(
    manifest_path: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    frame_transform=None,
    include_frames: bool = True,
    collate_fn=collate_deepfake_batch,
) -> DataLoader:
    """Create a dataloader for preprocessed deepfake clips.

    Args:
        manifest_path: Path to the manifest CSV file.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data.
        num_workers: Number of subprocesses to use for data loading.
        frame_transform: Optional transform applied to loaded PIL frames.
        include_frames: Whether to load video frames for each sample.
        collate_fn: Function used to merge samples into batches.

    Returns:
        PyTorch dataloader for the deepfake clip dataset.

    Raises:
        FileNotFoundError: If the manifest file does not exist.

    """
    if not Path(manifest_path).is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    dataset = DeepFakeClipDataset(
        manifest_path,
        frame_transform=frame_transform,
        include_frames=include_frames,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
